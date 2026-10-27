import json
from pathlib import Path

import pytest

import benchmark
from app.evaluation.metrics import score, wilson

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_benchmark_sizes_and_document_separation():
    root = ROOT / "evaluation/datasets/scifact-v2"
    dev = benchmark.read_dataset(root / "development.json")
    test = benchmark.read_dataset(root / "test.json")
    assert len(dev["cases"]) == 100
    assert len(test["cases"]) == 300
    documents = lambda data: {p["id"] for c in data["cases"] for p in c["passages"]}
    assert not documents(dev) & documents(test)
    assert all(len(c["passages"]) == 16 for c in dev["cases"] + test["cases"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source_document_overlap"] == 0


def test_no_evidence_claims_are_excluded_from_retrieval_denominator():
    rows = [
        {"expected_source_ids": ["A", "B"], "ranked_source_ids": ["A"], "expected_verdict": "supported", "actual_verdict": "supported"},
        {"expected_source_ids": [], "ranked_source_ids": ["X"], "expected_verdict": "unsupported", "actual_verdict": None},
    ]
    metrics = score(rows, verdicts=True)
    assert metrics["retrieval_recall_at_4"] == 0.5
    assert metrics["retrieval_hit_rate_at_4"] == 1.0
    assert metrics["retrieval_applicable_claims"] == 1
    assert metrics["verdict_accuracy"] == 0.5
    assert metrics["missing_verdicts"] == 1
    assert metrics["macro_f1_observed_labels"] == 0.5


def test_wilson_interval_does_not_treat_twelve_successes_as_certainty():
    lower, upper = wilson(12, 12)
    assert 0.75 < lower < 0.8
    assert upper == 1.0


def test_offline_baseline_can_run_without_provider(monkeypatch):
    monkeypatch.setattr(benchmark, "embed_texts", lambda _: pytest.fail("Unexpected embedding call"))
    monkeypatch.setattr(benchmark, "evaluate_claims", lambda *args: pytest.fail("Unexpected verdict call"))
    report = benchmark.run(ROOT / "evaluation/datasets/scifact-v2/development.json", "lexical", False, 1, 2, Path("unused"))
    assert report["partial_run"]
    assert report["evaluated_cases"] == 2
    assert report["models"]["embedding"] is None
    assert "verdict_accuracy" not in report["strategies"]["lexical"]["metrics"]


def test_comparison_uses_same_vectors_and_records_real_pipeline_outputs(monkeypatch, tmp_path):
    import numpy as np
    from app.schemas.audit import ClaimAudit
    from app.services import retrieval
    seen_batches = []
    def embeddings(texts):
        seen_batches.extend(texts)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))
    monkeypatch.setattr(benchmark, "embed_texts", embeddings)
    monkeypatch.setattr(retrieval, "_rerank", lambda query, candidates: list(reversed(candidates))[:4])
    def verdicts(claims, evidence):
        return [ClaimAudit(claim=c, verdict="unsupported", explanation="Mock prediction only.", needs_human_review=True) for c in claims]
    monkeypatch.setattr(benchmark, "evaluate_claims", verdicts)
    report = benchmark.run(ROOT / "evaluation/datasets/scifact-v2/development.json", "compare", True, 1, 2, tmp_path)
    assert set(report["strategies"]) == {"cosine", "rerank"}
    assert report["verdicts_evaluated"]
    assert len(seen_batches) == len(set(seen_batches))
    for result in report["strategies"].values():
        assert result["metrics"]["verdict_coverage"] == 1.0
        assert len(result["claim_results"]) == 2
        assert all(row["actual_verdict"] == "unsupported" for row in result["claim_results"])
        assert not result["errors"]


def test_policy_examples_cover_all_five_labels_without_claiming_expert_review():
    data = benchmark.read_dataset(ROOT / "evaluation/datasets/policy-v2.json")
    assert len(data["cases"]) == 10
    assert {c["expected_verdict"] for case in data["cases"] for c in case["claims"]} == {
        "supported", "unsupported", "contradicted", "weakly_supported", "human_review_required"
    }
    assert "not independently" in data["annotation_status"]
