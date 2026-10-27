from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_audit_rejects_short_draft_before_model_call():
    response = client.post(
        "/api/audits/run",
        data={"draft": "Too short"},
        files={"source": ("source.txt", b"Some source text.", "text/plain")},
    )

    assert response.status_code == 422
    assert "at least 40" in response.json()["detail"]


def test_audit_rejects_unsupported_file_type_before_model_call():
    response = client.post(
        "/api/audits/run",
        data={"draft": "This draft is comfortably longer than forty characters."},
        files={"source": ("source.csv", b"a,b,c", "text/csv")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Only PDF and TXT source files are supported."


def test_complete_audit_route_parses_retrieves_and_returns_citations(monkeypatch):
    from types import SimpleNamespace
    from app.schemas.audit import ExtractedClaim
    from app.services import qwen_client, retrieval
    from app.services import audit_engine
    import json
    import numpy as np

    claim = ExtractedClaim(id="C1", text="The pilot enrolled 120 editors.", type="statistic", importance="high")
    monkeypatch.setattr(audit_engine, "extract_claims", lambda draft: [claim])
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: np.array([[1., 0.]] * len(texts)))
    result = {"verdicts": [{"claim_id": "C1", "verdict": "supported", "explanation": "The source explicitly states the count.",
                            "evidence_quote": claim.text, "source_ids": ["S1"]}]}
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
    monkeypatch.setattr(qwen_client, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))))
    response = client.post("/api/audits/run", data={"draft": "The pilot enrolled 120 editors, according to the report."},
                           files={"source": ("pilot.txt", claim.text.encode(), "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["publication_status"] == "ready_with_review"
    assert data["claim_audits"][0]["citations"][0]["filename"] == "pilot.txt"
    assert data["claim_audits"][0]["evidence_quote"] == claim.text
    assert data["pipeline"]["reranker_used"] is False


def test_quota_failure_is_actionable_without_echoing_provider_body(monkeypatch):
    from app.services import audit_engine
    from app.services.qwen_client import ClaimExtractionError
    class QuotaError(RuntimeError):
        body = {"code": "AllocationQuota.FreeTierOnly", "message": "private provider payload"}
        status_code = 400
    def fail(draft):
        raise ClaimExtractionError("Request failed") from QuotaError("private payload")
    monkeypatch.setattr(audit_engine, "extract_claims", fail)
    response = client.post("/api/audits/run", data={"draft": "A draft that is long enough to contain factual statements."},
                           files={"source": ("source.txt", b"A real source.", "text/plain")})
    assert response.status_code == 503
    assert "quota" in response.json()["detail"]
    assert "private" not in response.text
