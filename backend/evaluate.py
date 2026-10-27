"""Evaluate retrieval and verdicts on specified claims, not claim extraction."""

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import get_args

from app.schemas.audit import ExtractedClaim, Verdict
from app.services.document_parser import SourceChunk
from app.services.qwen_client import MODEL_NAME, evaluate_claims
from app.services.retrieval import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    RETRIEVAL_TOP_K,
    retrieve_evidence,
)

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "benchmark.json"
HISTORICAL_RESULTS_PATH = BENCHMARK_PATH.with_name("results.json")
VERDICTS = get_args(Verdict)


def _validate_cases(cases: list[dict]) -> None:
    """Reject ambiguous gold labels before making provider requests."""
    if not cases:
        raise ValueError("The benchmark must contain at least one case.")
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Benchmark case IDs must be unique.")
    for case in cases:
        passages, claims = case["passages"], case["claims"]
        if not passages or not claims:
            raise ValueError(f"{case['id']}: passages and claims must be nonempty.")
        passage_ids = {item["id"] for item in passages}
        claim_ids = {item["id"] for item in claims}
        if len(passage_ids) != len(passages) or len(claim_ids) != len(claims):
            raise ValueError(f"{case['id']}: duplicate passage or claim IDs.")
        for claim in claims:
            if claim["expected_verdict"] not in VERDICTS:
                raise ValueError(f"{case['id']}: invalid expected verdict.")
            if claim["expected_source_id"] not in passage_ids:
                raise ValueError(f"{case['id']}: expected source is not in passages.")
            ExtractedClaim.model_validate(claim)


def summarize(rows: list[dict]) -> dict:
    """Every gold claim stays in the denominator, including missing outputs."""
    if not rows:
        raise ValueError("Cannot score an empty set of claims.")
    total = len(rows)
    gold_counts = Counter(row["expected_verdict"] for row in rows)
    per_verdict = {}
    confusion = {label: {} for label in VERDICTS}
    for row in rows:
        label = row["expected_verdict"]
        prediction = row["actual_verdict"] or "missing"
        confusion[label][prediction] = confusion[label].get(prediction, 0) + 1
    for label in VERDICTS:
        true_positive = sum(
            row["expected_verdict"] == label and row["actual_verdict"] == label
            for row in rows
        )
        predicted = sum(row["actual_verdict"] == label for row in rows)
        support = gold_counts[label]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_verdict[label] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4) if support else None,
            "f1": round(f1, 4) if support else None,
        }
    return {
        "claims": total,
        "verdict_accuracy": round(sum(row["verdict_correct"] for row in rows) / total, 4),
        "verdict_coverage": round(sum(row["actual_verdict"] is not None for row in rows) / total, 4),
        "missing_verdicts": sum(row["actual_verdict"] is None for row in rows),
        "retrieval_recall_at_1": round(sum(row["retrieval_hit_at_1"] for row in rows) / total, 4),
        "retrieval_recall_at_4": round(sum(row["retrieval_hit_at_4"] for row in rows) / total, 4),
        "per_verdict": per_verdict,
        "confusion_matrix": confusion,
    }


def evaluate(benchmark_path: Path = BENCHMARK_PATH) -> dict:
    benchmark_bytes = benchmark_path.read_bytes()
    cases = json.loads(benchmark_bytes)
    _validate_cases(cases)
    rows = []
    case_runs = []
    for case in cases:
        chunks = [
            SourceChunk(
                id=passage["id"], text=passage["text"],
                page_number=passage["page"], filename=f"{case['id']}.txt",
            )
            for passage in case["passages"]
        ]
        claims = [ExtractedClaim.model_validate(item) for item in case["claims"]]
        started = perf_counter()
        retrieval = retrieve_evidence(claims, chunks)
        audits = evaluate_claims(claims, retrieval.evidence_by_claim)
        expected_ids = {claim.id for claim in claims}
        returned_ids = [audit.claim.id for audit in audits]
        if len(set(returned_ids)) != len(returned_ids):
            raise ValueError(f"{case['id']}: duplicate verdict IDs; run is invalid.")
        if set(returned_ids) - expected_ids:
            raise ValueError(f"{case['id']}: unknown verdict IDs; run is invalid.")
        audit_by_id = {audit.claim.id: audit for audit in audits}
        case_runs.append({
            "case_id": case["id"],
            "elapsed_seconds": round(perf_counter() - started, 3),
            "reranker_used": retrieval.reranker_used,
        })
        for expected in case["claims"]:
            claim_id = expected["id"]
            audit = audit_by_id.get(claim_id)
            ranked_ids = [
                passage.source_id
                for passage in retrieval.evidence_by_claim.get(claim_id, [])
            ]
            actual = audit.verdict if audit else None
            rows.append({
                "case_id": case["id"], "claim_id": claim_id,
                "claim": expected["text"],
                "expected_verdict": expected["expected_verdict"],
                "actual_verdict": actual,
                "verdict_correct": actual == expected["expected_verdict"],
                "expected_source_id": expected["expected_source_id"],
                "ranked_source_ids": ranked_ids,
                "retrieval_hit_at_1": expected["expected_source_id"] in ranked_ids[:1],
                "retrieval_hit_at_4": expected["expected_source_id"] in ranked_ids[:4],
                "evidence_quote": audit.evidence_quote if audit else "",
                "cited_source_ids": [p.source_id for p in audit.citations] if audit else [],
            })
            print(
                f"{case['id']}/{claim_id}: expected={expected['expected_verdict']} "
                f"actual={actual or 'missing'} top={ranked_ids[:4]}",
                file=sys.stderr,
            )
    return {
        "schema_version": 2,
        "run_at": datetime.now(UTC).isoformat(),
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "verdict_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "cases": len(cases),
        "passages": sum(len(case["passages"]) for case in cases),
        **summarize(rows),
        "case_runs": case_runs,
        "claim_results": rows,
        "limitations": [
            "Synthetic benchmark; not a held-out production evaluation.",
            "Claims and passages are supplied; extraction and document parsing are not evaluated.",
            "Recall uses one labeled relevant source per claim, including passages explaining non-support.",
        ] + ([
            "Every case fits within the first-stage retrieval limit; candidate pruning is not tested."
        ] if all(len(case["passages"]) <= RETRIEVAL_TOP_K for case in cases) else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, help="Save a new JSON report; uses live Qwen APIs.")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {
        HISTORICAL_RESULTS_PATH.resolve(), args.benchmark.resolve(),
    }:
        parser.error("Choose a new output path; do not overwrite historical results or benchmark inputs.")
    report = evaluate(args.benchmark)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
