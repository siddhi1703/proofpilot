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
