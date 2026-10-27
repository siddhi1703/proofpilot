"""Checkpoint open-weight evaluation to Drive after each small group of cases."""
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path

import benchmark
from app.evaluation.metrics import score

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_fingerprint() -> str:
    paths = sorted((ROOT / "backend/app").rglob("*.py")) + [ROOT / "backend/benchmark.py", Path(__file__)]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_resumable(dataset_path: Path, models, output_root: Path, *, limit: int | None = 10,
                  chunk_size: int = 5, verdicts: bool = True, mode: str = "compare") -> dict:
    if (limit is not None and limit < 1) or chunk_size < 1:
        raise ValueError("Limit and chunk size must be positive.")
    if mode not in {"compare", "cosine", "rerank"}:
        raise ValueError("Choose compare, cosine, or rerank.")
    strategy_names = ["cosine", "rerank"] if mode == "compare" else [mode]
    dataset = benchmark.read_dataset(dataset_path)
    cases = dataset["cases"][:limit] if limit else dataset["cases"]
    configuration = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "code_sha256": code_fingerprint(), "models": models.metadata,
        "mode": mode, "verdicts": verdicts, "verdict_batch_size": 1,
    }
    # The limit is excluded so the pilot's completed chunks can be reused later.
    run_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    directory = output_root / dataset["split"] / run_id
    directory.mkdir(parents=True, exist_ok=True)
    packages = {}
    for package in ["torch", "transformers", "sentence-transformers", "bitsandbytes", "accelerate", "numpy"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    configuration["packages"] = packages
    # Package versions are part of resume identity: never mix different stacks.
    existing = directory / "configuration.json"
    if existing.exists() and json.loads(existing.read_text()) != configuration:
        raise ValueError("Installed package versions changed. Use a new output folder for this experiment.")
    atomic_json(existing, configuration)
    chunks = []
    with models.activate():
        for start in range(0, len(cases), chunk_size):
            group = cases[start:start + chunk_size]
            group_id = hashlib.sha256(json.dumps([c["id"] for c in group]).encode()).hexdigest()[:16]
            checkpoint = directory / f"chunk-{group_id}.json"
            if checkpoint.exists():
                report = json.loads(checkpoint.read_text())
                print(f"Resuming: reused {start + 1}-{start + len(group)}", flush=True)
            else:
                part = {**dataset, "cases": group}
                part_path = directory / f"input-{group_id}.json"
                atomic_json(part_path, part)
                report = benchmark.run(part_path, mode, verdicts, 1, None,
                                       output_root / "embedding-cache", verdict_batch_size=1)
                atomic_json(checkpoint, report)
                print(f"Saved checkpoint for cases {start + 1}-{start + len(group)}", flush=True)
            chunks.append(report)
            strategies = {}
            for name in strategy_names:
                rows = [row for chunk in chunks for row in chunk["strategies"][name]["claim_results"]]
                strategies[name] = {
                    "metrics": score(rows, verdicts=verdicts),
                    "errors": [error for chunk in chunks for error in chunk["strategies"][name]["errors"]],
                    "reranker_fallback_claims": sum(chunk["strategies"][name]["reranker_fallback_claims"] for chunk in chunks),
                    "claim_results": rows,
                }
            report = {
                "schema_version": 1, "profile": "free-colab-open-weights",
                "updated_at": datetime.now(UTC).isoformat(), "configuration": configuration,
                "dataset_id": dataset["dataset_id"], "split": dataset["split"],
                "dataset_total_cases": len(dataset["cases"]), "requested_cases": len(cases),
                "completed_cases": sum(chunk["evaluated_cases"] for chunk in chunks),
                "partial_run": sum(chunk["evaluated_cases"] for chunk in chunks) < len(dataset["cases"]),
                "strategies": strategies, "limitations": dataset.get("limitations", []) + [
                    "Open-weight models and windowing differ from the hosted Qwen pipeline.",
                    "Cross-encoder scores rank relevance, not entailment or calibrated truth probabilities.",
                    "Inference errors are preserved as missing predictions; cached failed chunks are not silently retried.",
                ],
            }
            report_path = directory / f"report-{len(cases)}-cases.json"
            atomic_json(report_path, report)
    report["saved_report"] = str(report_path)
    return report
