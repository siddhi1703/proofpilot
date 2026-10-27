from app.schemas.audit import ClaimAudit, ExtractedClaim
from app.services import audit_engine as engine
from app.services.document_parser import parse_document
from app.services.retrieval import RetrievalResult


def run(monkeypatch, label=None, draft="The pilot enrolled participants and completed its work."):
    claims = [] if label is None else [ExtractedClaim(id="C1", text="The pilot enrolled participants.", type="general_fact", importance="low")]
    audits = [] if label is None else [ClaimAudit(claim=claims[0], verdict=label, explanation="Test evidence.", needs_human_review=label != "supported")]
    monkeypatch.setattr(engine, "extract_claims", lambda _: claims)
    monkeypatch.setattr(engine, "retrieve_evidence", lambda *args: RetrievalResult({}, False))
    monkeypatch.setattr(engine, "evaluate_claims", lambda *args: audits)
    return engine.run_audit(draft, parse_document("source.txt", b"The pilot enrolled participants."))


def test_no_claims_is_not_a_verified_draft(monkeypatch):
    report = run(monkeypatch)
    assert report.summary.publication_status == "blocked"
    assert report.warnings


def test_low_importance_weak_support_still_requires_review(monkeypatch):
    assert run(monkeypatch, "weakly_supported").summary.publication_status == "blocked"


def test_supported_claims_still_require_final_editor_review(monkeypatch):
    assert run(monkeypatch, "supported").summary.publication_status == "ready_with_review"


def test_high_privacy_risk_blocks_even_when_claim_is_supported(monkeypatch):
    report = run(monkeypatch, "supported", "The source includes this identifier 123-45-6789.")
    assert report.summary.publication_status == "blocked"
    assert report.privacy_findings[0].risk == "high"
