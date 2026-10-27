"""Evaluation integrity checks; these never call a model provider."""

import hashlib
import json

import pytest

import evaluate as runner
from app.schemas.audit import ClaimAudit, EvidencePassage
from app.services.retrieval import RetrievalResult


@pytest.fixture
def benchmark(tmp_path):
    case = {
        "id": "pilot",
        "passages": [{"id": "S1", "page": 1, "text": "The pilot enrolled 120 editors."}],
        "claims": [
            {"id": "C1", "text": "The pilot enrolled 120 editors.",
             "type": "statistic", "importance": "high",
             "expected_verdict": "supported", "expected_source_id": "S1"},
            {"id": "C2", "text": "The pilot enrolled 200 editors.",
             "type": "statistic", "importance": "high",
             "expected_verdict": "contradicted", "expected_source_id": "S1"},
        ],
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([case]))
    return path


def install_fakes(monkeypatch, mode="missing"):
    def retrieve(claims, chunks):
        evidence = EvidencePassage(
            source_id="S1", filename="pilot.txt", page_number=1,
            text=chunks[0].text, retrieval_score=0.8,
        )
        return RetrievalResult({claim.id: [evidence] for claim in claims}, False)

    def verdicts(claims, evidence):
        def audit(claim):
            return ClaimAudit(
                claim=claim, verdict="supported", explanation="Matches source.",
                evidence_quote="The pilot enrolled 120 editors.",
                citations=evidence.get(claim.id, []), needs_human_review=False,
            )
        if mode == "empty":
            return []
        if mode == "duplicate":
            return [audit(claims[0]), audit(claims[0])]
        if mode == "unknown":
            return [audit(claims[0].model_copy(update={"id": "C99"}))]
        return [audit(claims[0])]

    monkeypatch.setattr(runner, "retrieve_evidence", retrieve)
    monkeypatch.setattr(runner, "evaluate_claims", verdicts)


def test_missing_verdict_is_in_denominator_and_report(monkeypatch, benchmark):
    install_fakes(monkeypatch)
    report = runner.evaluate(benchmark)
    assert report["claims"] == 2
    assert report["verdict_accuracy"] == 0.5
    assert report["verdict_coverage"] == 0.5
    assert report["missing_verdicts"] == 1
    assert report["confusion_matrix"]["contradicted"] == {"missing": 1}
    assert report["per_verdict"]["contradicted"]["recall"] == 0.0
    assert report["per_verdict"]["weakly_supported"]["f1"] is None
    # Retrieval is scored independently of whether verdict generation succeeded.
    assert report["retrieval_recall_at_4"] == 1.0
    assert report["claim_results"][1]["actual_verdict"] is None
    assert report["case_runs"][0]["reranker_used"] is False
    assert report["benchmark_sha256"] == hashlib.sha256(benchmark.read_bytes()).hexdigest()
    assert any("candidate pruning" in item for item in report["limitations"])
    json.dumps(report)


def test_empty_verdict_output_scores_zero_instead_of_dividing_by_zero(monkeypatch, benchmark):
    install_fakes(monkeypatch, "empty")
    report = runner.evaluate(benchmark)
    assert report["claims"] == 2
    assert report["verdict_accuracy"] == 0.0
    assert report["missing_verdicts"] == 2


@pytest.mark.parametrize("mode", ["duplicate", "unknown"])
def test_invalid_verdict_ids_do_not_generate_a_score(monkeypatch, benchmark, mode):
    install_fakes(monkeypatch, mode)
    with pytest.raises(ValueError, match=mode):
        runner.evaluate(benchmark)


def test_invalid_gold_source_is_rejected_before_provider_call(monkeypatch, benchmark):
    cases = json.loads(benchmark.read_text())
    cases[0]["claims"][0]["expected_source_id"] = "S99"
    benchmark.write_text(json.dumps(cases))
    def forbidden(*args):
        pytest.fail("Invalid benchmark must not call the provider")
    monkeypatch.setattr(runner, "retrieve_evidence", forbidden)
    with pytest.raises(ValueError, match="expected source"):
        runner.evaluate(benchmark)


def test_wrong_verdict_populates_confusion_matrix(monkeypatch, benchmark):
    install_fakes(monkeypatch)
    original = runner.evaluate_claims
    def wrong_verdicts(claims, evidence):
        return original(claims, evidence) + [ClaimAudit(
            claim=claims[1], verdict="supported", explanation="Incorrect prediction.",
            needs_human_review=False,
        )]
    monkeypatch.setattr(runner, "evaluate_claims", wrong_verdicts)
    report = runner.evaluate(benchmark)
    assert report["verdict_coverage"] == 1.0
    assert report["verdict_accuracy"] == 0.5
    assert report["per_verdict"]["supported"]["precision"] == 0.5
    assert report["confusion_matrix"]["contradicted"]["supported"] == 1


@pytest.mark.parametrize("target", ["historical", "benchmark"])
def test_cli_protects_original_results_and_inputs(monkeypatch, benchmark, target):
    output = runner.HISTORICAL_RESULTS_PATH if target == "historical" else benchmark
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--benchmark", str(benchmark),
                                     "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_cli_writes_machine_readable_json(monkeypatch, benchmark, tmp_path, capsys):
    install_fakes(monkeypatch)
    output = tmp_path / "runs" / "result.json"
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--benchmark", str(benchmark),
                                     "--output", str(output)])
    runner.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == json.loads(output.read_text())
    assert "pilot/C1" in captured.err
