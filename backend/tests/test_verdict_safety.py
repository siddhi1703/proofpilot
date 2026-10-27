import json
from types import SimpleNamespace

import pytest

from app.schemas.audit import EvidencePassage, ExtractedClaim
from app.services import qwen_client as service


@pytest.fixture
def claim():
    return ExtractedClaim(id="C1", text="The pilot enrolled 200 editors.", type="statistic", importance="high")


@pytest.fixture
def evidence():
    return {"C1": [EvidencePassage(source_id="S1", filename="source.txt", page_number=1,
                                  text="The pilot enrolled 120 editors.", retrieval_score=0.8)]}


def mock_response(monkeypatch, verdicts):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"verdicts": verdicts})))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion)))
    monkeypatch.setattr(service, "get_client", lambda: client)


def verdict(**updates):
    return {"claim_id": "C1", "verdict": "supported", "explanation": "A source was supplied.",
            "source_ids": ["S1"], "evidence_quote": "The pilot enrolled 120 editors.", **updates}


def test_exact_quote_does_not_approve_a_different_numeric_claim(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [verdict()])
    audit = service.evaluate_claims([claim], evidence)[0]
    assert audit.verdict == "human_review_required"
    assert audit.needs_human_review


def test_paraphrased_quote_is_not_repaired_into_supported_verdict(monkeypatch, claim, evidence):
    claim = claim.model_copy(update={"text": "The pilot enrolled 120 editors."})
    mock_response(monkeypatch, [verdict(evidence_quote="120 editors took part in the pilot.")])
    assert service.evaluate_claims([claim], evidence)[0].verdict == "human_review_required"


@pytest.mark.parametrize("label", ["supported", "weakly_supported", "contradicted"])
def test_quote_required_verdicts_reject_invented_quotes(monkeypatch, claim, evidence, label):
    mock_response(monkeypatch, [verdict(verdict=label, evidence_quote="Invented evidence.")])
    assert service.evaluate_claims([claim], evidence)[0].verdict == "human_review_required"


def test_unknown_citation_cannot_be_silently_dropped(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [verdict(source_ids=["S1", "S99"])])
    assert service.evaluate_claims([claim], evidence)[0].verdict == "human_review_required"


def test_duplicate_verdicts_require_review(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [verdict(), verdict()])
    assert service.evaluate_claims([claim], evidence)[0].verdict == "human_review_required"


def test_unknown_claim_id_rejects_model_response(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [verdict(claim_id="C99")])
    with pytest.raises(service.VerdictError, match="unknown"):
        service.evaluate_claims([claim], evidence)


def test_missing_verdict_requires_review(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [])
    assert service.evaluate_claims([claim], evidence)[0].verdict == "human_review_required"


def test_valid_supported_verdict_retains_quote(monkeypatch, claim, evidence):
    claim = claim.model_copy(update={"text": "The pilot enrolled 120 editors."})
    mock_response(monkeypatch, [verdict()])
    audit = service.evaluate_claims([claim], evidence)[0]
    assert audit.verdict == "supported"
    assert not audit.needs_human_review
    assert audit.evidence_quote == evidence["C1"][0].text


def test_unsupported_verdict_has_no_synthesized_quote(monkeypatch, claim, evidence):
    mock_response(monkeypatch, [verdict(verdict="unsupported", evidence_quote="")])
    audit = service.evaluate_claims([claim], evidence)[0]
    assert audit.verdict == "unsupported"
    assert audit.evidence_quote == ""


def test_claim_extraction_rejects_possible_truncation(monkeypatch):
    claims = [{"id": f"C{i}", "text": f"Claim number {i} is factual.", "type": "general_fact", "importance": "high"} for i in range(12)]
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"claims": claims})))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion)))
    monkeypatch.setattr(service, "get_client", lambda: client)
    with pytest.raises(service.ClaimExtractionError, match="12-claim"):
        service.extract_claims("A draft with many factual statements.")
