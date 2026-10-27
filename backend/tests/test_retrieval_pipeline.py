from types import SimpleNamespace

import numpy as np
import pytest

from app.schemas.audit import ExtractedClaim
from app.services import retrieval as service
from app.services.document_parser import SourceChunk


def inputs():
    claims = [ExtractedClaim(id="C1", text="Pilot participation count.", type="general_fact", importance="high")]
    chunks = [SourceChunk(f"S{i}", f"Passage number {i}.", i, "source.txt") for i in range(1, 17)]
    # Deliberately put the best match outside the first eight input passages.
    vectors = np.array([[0.0, 1.0]] * 15 + [[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    return claims, chunks, vectors


def test_cosine_baseline_searches_whole_pool_without_reranker(monkeypatch):
    claims, chunks, vectors = inputs()
    monkeypatch.setattr(service, "_rerank", lambda *args: pytest.fail("Baseline called reranker"))
    result = service.retrieve_evidence(claims, chunks, use_reranker=False, embedding_fn=lambda _: vectors)
    assert result.evidence_by_claim["C1"][0].source_id == "S16"
    assert len(result.candidate_source_ids["C1"]) == 8
    assert not result.reranker_used
    assert not result.fallback_claim_ids


def test_reranker_failure_is_reported_per_claim(monkeypatch):
    claims, chunks, vectors = inputs()
    def fail(*args):
        raise RuntimeError("Provider outage")
    monkeypatch.setattr(service, "_rerank", fail)
    result = service.retrieve_evidence(claims, chunks, embedding_fn=lambda _: vectors)
    assert not result.reranker_used
    assert result.fallback_claim_ids == ["C1"]
    assert result.evidence_by_claim["C1"][0].source_id == "S16"


def test_bad_embedding_shape_raises_controlled_error():
    claims, chunks, _ = inputs()
    with pytest.raises(service.RetrievalError, match="inputs"):
        service.retrieve_evidence(claims, chunks, embedding_fn=lambda _: np.ones((1, 2)))


@pytest.mark.parametrize("indexes", [[0, 0], [0], [True, 1]])
def test_incomplete_or_duplicate_embedding_indexes_fail(monkeypatch, indexes):
    response = SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0]) for i in indexes])
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(service, "_embedding_client", lambda: client)
    with pytest.raises(service.RetrievalError):
        service.embed_texts(["one", "two"])
