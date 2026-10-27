"""Build and evaluate versioned datasets. Model modes call live Qwen APIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import get_args

import numpy as np

from app.evaluation.lexical import BM25
from app.evaluation.metrics import score
from app.evaluation.scifact import build
from app.schemas.audit import EvidencePassage, ExtractedClaim, Verdict
from app.services.document_parser import SourceChunk
from app.services.qwen_client import MODEL_NAME, evaluate_claims
from app.services.retrieval import (
    EMBEDDING_MODEL, RERANK_MODEL, RetrievalResult, embed_texts, retrieve_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def read_dataset(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 3 or not dataset.get("cases"):
        raise ValueError("Expected a nonempty schema-version-3 dataset.")
    seen_claims, seen_cases = set(), set()
    for case in dataset["cases"]:
        if case["id"] in seen_cases:
            raise ValueError("Duplicate case ID.")
        seen_cases.add(case["id"])
        passages = {item["id"] for item in case["passages"]}
        if not passages or len(passages) != len(case["passages"]):
            raise ValueError("Empty passages or duplicate passage IDs.")
        if not case["claims"]:
            raise ValueError("A case must contain claims.")
        for item in case["claims"]:
            ExtractedClaim.model_validate(item)
            if item["id"] in seen_claims:
                raise ValueError("Claim IDs must be unique across cases for batched evaluation.")
            seen_claims.add(item["id"])
            if item["expected_verdict"] not in get_args(Verdict):
                raise ValueError("Invalid gold verdict.")
            if not set(item["expected_source_ids"]) <= passages:
                raise ValueError("Gold source is not present in its case.")
    return dataset


class VectorCache:
    """Explicit public-benchmark cache, never used for uploaded user documents."""
    def __init__(self, directory: Path):
        self.directory = directory / hashlib.sha256(EMBEDDING_MODEL.encode()).hexdigest()[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors = {}

    def prepare(self, texts: list[str], workers: int):
        missing = []
        for text in dict.fromkeys(texts):
            path = self.directory / (hashlib.sha256(text.encode()).hexdigest() + ".npy")
            if path.exists():
                vector = np.load(path, allow_pickle=False)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError("Invalid benchmark embedding cache entry.")
                self.vectors[text] = vector
            else:
                missing.append(text)
        batches = [missing[i:i + 10] for i in range(0, len(missing), 10)]
        def encode(batch):
            return batch, embed_texts(batch)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(encode, batch) for batch in batches]
            for n, future in enumerate(as_completed(futures), 1):
                batch, matrix = future.result()
                for text, vector in zip(batch, matrix, strict=True):
                    path = self.directory / (hashlib.sha256(text.encode()).hexdigest() + ".npy")
                    np.save(path, vector, allow_pickle=False)
                    self.vectors[text] = vector
                if n % 10 == 0 or n == len(batches):
                    print(f"Embeddings: {n}/{len(batches)} batches", file=sys.stderr, flush=True)

    def get(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.vectors[text] for text in texts])


def lexical_retrieval(claims, chunks):
    index = BM25([chunk.text for chunk in chunks])
    evidence, candidate_ids = {}, {}
    for claim in claims:
        ranked = index.rank(claim.text)[:8]
        candidate_ids[claim.id] = [chunks[i].id for i in ranked]
        evidence[claim.id] = [EvidencePassage(
            source_id=chunks[i].id, filename=chunks[i].filename,
            page_number=chunks[i].page_number, text=chunks[i].text,
            retrieval_score=0.0,
        ) for i in ranked[:4]]
    return RetrievalResult(evidence, False, candidate_ids)


def run(dataset_path: Path, mode: str, verdicts: bool, workers: int,
        limit: int | None, cache_path: Path, verdict_batch_size: int = 6) -> dict:
    if not 1 <= verdict_batch_size <= 6:
        raise ValueError("Verdict batch size must be between one and six.")
    dataset = read_dataset(dataset_path)
    cases = dataset["cases"][:limit] if limit else dataset["cases"]
    modes = ["cosine", "rerank"] if mode == "compare" else [mode]
    started = perf_counter()
    cache = None
    if mode != "lexical":
        cache = VectorCache(cache_path)
        cache.prepare([p["text"] for case in cases for p in case["passages"]]
                      + [c["text"] for case in cases for c in case["claims"]], workers)
    reports = {}
    for strategy in modes:
        rows, evidence_by_id, timings = [], {}, []
        def retrieve(case):
            claims = [ExtractedClaim.model_validate(c) for c in case["claims"]]
            chunks = [SourceChunk(p["id"], p["text"], p["page"], p.get("filename", case["id"] + ".txt"))
                      for p in case["passages"]]
            begin = perf_counter()
            result = (lexical_retrieval(claims, chunks) if strategy == "lexical" else
                      retrieve_evidence(claims, chunks, use_reranker=strategy == "rerank", embedding_fn=cache.get))
            return case, result, perf_counter() - begin
        # executor.map preserves dataset order, including with concurrent requests.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, (case, result, seconds) in enumerate(executor.map(retrieve, cases), 1):
                timings.append(seconds)
                evidence_by_id.update(result.evidence_by_claim)
                for item in case["claims"]:
                    selected = [p.source_id for p in result.evidence_by_claim.get(item["id"], [])]
                    candidates = result.candidate_source_ids.get(item["id"], selected)
                    # Final top four, then remaining first-stage candidates, for Recall@8.
                    ranking = selected + [sid for sid in candidates if sid not in selected]
                    rows.append({
                        "case_id": case["id"], "claim_id": item["id"], "claim": item["text"],
                        "type": item["type"], "importance": item["importance"],
                        "expected_verdict": item["expected_verdict"],
                        "expected_source_ids": item["expected_source_ids"],
                        "ranked_source_ids": ranking,
                        "reranker_fallback": item["id"] in result.fallback_claim_ids,
                    })
                if i % 25 == 0 or i == len(cases):
                    print(f"{strategy} retrieval: {i}/{len(cases)} cases", file=sys.stderr, flush=True)
        verdict_timings, errors = [], []
        if verdicts:
            def judge(batch):
                begin = perf_counter()
                claims = [ExtractedClaim(id=r["claim_id"], text=r["claim"], type=r["type"], importance=r["importance"]) for r in batch]
                try:
                    audits = evaluate_claims(claims, {c.id: evidence_by_id[c.id] for c in claims})
                    returned = [a.claim.id for a in audits]
                    if len(returned) != len(set(returned)) or set(returned) - {c.id for c in claims}:
                        raise ValueError("Invalid returned verdict IDs")
                    return batch, {a.claim.id: a for a in audits}, perf_counter() - begin, None
                except Exception as exc:
                    # Do not print provider exception bodies, which may echo input.
                    return batch, {}, perf_counter() - begin, type(exc).__name__
            batches = [rows[i:i + verdict_batch_size] for i in range(0, len(rows), verdict_batch_size)]
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
                for n, (batch, audits, seconds, error) in enumerate(executor.map(judge, batches), 1):
                    verdict_timings.append(seconds)
                    if error:
                        errors.append({"claim_ids": [r["claim_id"] for r in batch], "error_type": error})
                    for row in batch:
                        audit = audits.get(row["claim_id"])
                        row.update({
                            "actual_verdict": audit.verdict if audit else None,
                            "evidence_quote": audit.evidence_quote if audit else "",
                            "cited_source_ids": [p.source_id for p in audit.citations] if audit else [],
                        })
                    print(f"{strategy} verdicts: {n}/{len(batches)} batches", file=sys.stderr, flush=True)
        reports[strategy] = {
            "metrics": score(rows, verdicts=verdicts),
            "reranker_fallback_claims": sum(r["reranker_fallback"] for r in rows),
            "retrieval_seconds_p50": round(float(np.percentile(timings, 50)), 4),
            "retrieval_seconds_p95": round(float(np.percentile(timings, 95)), 4),
            "verdict_batch_seconds_p50": round(float(np.percentile(verdict_timings, 50)), 4) if verdict_timings else None,
            "errors": errors, "claim_results": rows,
        }
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "schema_version": 3, "run_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset["dataset_id"], "split": dataset["split"],
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_total_cases": len(dataset["cases"]), "evaluated_cases": len(cases),
        "partial_run": len(cases) != len(dataset["cases"]),
        "annotation_status": dataset["annotation_status"],
        "git_revision": revision.stdout.strip(),
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), ROOT / "backend/app/services/retrieval.py",
                                     ROOT / "backend/app/services/qwen_client.py"]},
        "verdicts_evaluated": verdicts,
        "verdict_batch_size": verdict_batch_size,
        "models": {"embedding": EMBEDDING_MODEL if mode != "lexical" else None,
                   "rerank": RERANK_MODEL if mode in {"rerank", "compare"} else None,
                   "verdict": MODEL_NAME if verdicts else None},
        "wall_seconds": round(perf_counter() - started, 3),
        "latency_note": "Retrieval timings exclude embedding precomputation; verdict timings are for batches of up to six claims. Neither is end-to-end application latency.",
        "limitations": dataset.get("limitations", []), "strategies": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build-scifact")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=ROOT / "evaluation/datasets/scifact-v2")
    prepare.add_argument("--development-size", type=int, default=100)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--mode", choices=["lexical", "cosine", "rerank", "compare"], default="lexical")
    evaluate.add_argument("--verdicts", action="store_true", help="Also run live model verdict evaluation.")
    evaluate.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--cache", type=Path, default=ROOT / "evaluation/cache")
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-scifact":
        print(json.dumps(build(args.archive, args.output, args.development_size), indent=2))
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve recorded runs.")
    report = run(args.dataset, args.mode, args.verdicts, args.workers, args.limit, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evaluated_cases": report["evaluated_cases"],
                      "strategies": {k: v["metrics"] for k, v in report["strategies"].items()}}, indent=2))


if __name__ == "__main__":
    main()
