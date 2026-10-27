from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from typing import Callable

import numpy as np
from openai import OpenAI

from app.schemas.audit import EvidencePassage, ExtractedClaim
from app.services.document_parser import SourceChunk
from app.services.qwen_client import BASE_URL, get_api_key

EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "qwen3-rerank")
RERANK_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-api/v1"
EMBEDDING_BATCH_SIZE = 10
RETRIEVAL_TOP_K = 8
RERANK_TOP_K = 4
LEXICAL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
}


class RetrievalError(RuntimeError):
    """Raised when evidence retrieval cannot be completed."""


@dataclass(frozen=True)
class RetrievalResult:
    evidence_by_claim: dict[str, list[EvidencePassage]]
    reranker_used: bool
    candidate_source_ids: dict[str, list[str]] = field(default_factory=dict)
    fallback_claim_ids: list[str] = field(default_factory=list)


def _embedding_client() -> OpenAI:
    return OpenAI(api_key=get_api_key(), base_url=BASE_URL, timeout=60, max_retries=2)


def _rerank_client() -> OpenAI:
    return OpenAI(
        api_key=get_api_key(),
        base_url=RERANK_BASE_URL,
        timeout=60,
        max_retries=2,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    client = _embedding_client()

    try:
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise RetrievalError("Embedding response has missing or duplicate indexes.")
            vectors.extend(item.embedding for item in ordered)
    except Exception as exc:
        raise RetrievalError("Qwen could not create evidence embeddings.") from exc

    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise RetrievalError("Embedding response has inconsistent dimensions.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise RetrievalError("Embedding response contains invalid vectors.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RetrievalError("Embedding response contains a zero vector.")
    return matrix / np.clip(norms, 1e-12, None)


def _coerce_payload(response: object) -> dict:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "json"):
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    raise RetrievalError("Qwen returned an unreadable rerank response.")


def _lexical_overlap(query: str, passage: str) -> float:
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", query.casefold())
        if token not in LEXICAL_STOP_WORDS
    }
    passage_tokens = set(re.findall(r"[a-z0-9]+", passage.casefold()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


def _rerank(
    query: str,
    candidates: list[EvidencePassage],
) -> list[EvidencePassage]:
    response = _rerank_client().post(
        "/reranks",
        body={
            "model": RERANK_MODEL,
            "query": query,
            "documents": [candidate.text for candidate in candidates],
            "top_n": min(RERANK_TOP_K, len(candidates)),
            "return_documents": False,
            "instruct": (
                "Rank passages by how directly they support, weaken, or contradict "
                "the factual claim. Prefer exact numbers, names, dates, and qualifiers."
            ),
        },
        cast_to=object,
    )
    payload = _coerce_payload(response)
    raw_results = payload.get("results") or payload.get("output", {}).get("results") or []

    reranked: list[EvidencePassage] = []
    seen_indexes: set[int] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if type(index) is not int or not 0 <= index < len(candidates) or index in seen_indexes:
            continue
        seen_indexes.add(index)
        score = item.get("relevance_score", item.get("score"))
        candidate = candidates[index]
        reranked.append(
            candidate.model_copy(
                update={
                    "rerank_score": (
                        max(0.0, min(1.0, float(score)))
                        if isinstance(score, (int, float)) and np.isfinite(score)
                        else None
                    )
                }
            )
        )

    if len(reranked) < min(RERANK_TOP_K, len(candidates)):
        raise RetrievalError("Qwen reranker returned an incomplete ranking.")

    # Fact checking is unusually sensitive to exact metrics and entities. Keep
    # Qwen's ordering, but reserve one of four slots for the strongest lexical
    # match when the neural reranker omitted it (often because it contains a
    # negation such as "did not measure").
    lexical_best = max(
        candidates,
        key=lambda candidate: _lexical_overlap(query, candidate.text),
    )
    if _lexical_overlap(query, lexical_best.text) > 0 and all(item.source_id != lexical_best.source_id for item in reranked):
        reranked = reranked[: max(0, RERANK_TOP_K - 1)] + [lexical_best]

    return reranked[:RERANK_TOP_K]


def retrieve_evidence(
    claims: list[ExtractedClaim],
    chunks: list[SourceChunk],
    *,
    use_reranker: bool = True,
    embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> RetrievalResult:
    if not claims:
        return RetrievalResult(evidence_by_claim={}, reranker_used=False)
    if not chunks:
        raise RetrievalError("No source chunks are available for retrieval.")

    # One combined embedding request avoids a second network round trip when
    # the source and claims fit in the same API batch.
    if len({claim.id for claim in claims}) != len(claims) or len({chunk.id for chunk in chunks}) != len(chunks):
        raise RetrievalError("Claim and source IDs must be unique within an audit.")
    combined_vectors = (embedding_fn or embed_texts)(
        [chunk.text for chunk in chunks] + [claim.text for claim in claims]
    )
    chunk_vectors = combined_vectors[: len(chunks)]
    claim_vectors = combined_vectors[len(chunks) :]
    if combined_vectors.ndim != 2 or len(combined_vectors) != len(chunks) + len(claims) or not np.isfinite(combined_vectors).all():
        raise RetrievalError("Embedding vectors do not match retrieval inputs.")
    similarities = np.clip(claim_vectors @ chunk_vectors.T, -1.0, 1.0)

    candidates_by_claim: dict[str, list[EvidencePassage]] = {}
    for claim_index, claim in enumerate(claims):
        top_count = min(RETRIEVAL_TOP_K, len(chunks))
        top_indexes = np.argsort(-similarities[claim_index], kind="stable")[:top_count]
        candidates_by_claim[claim.id] = [
            EvidencePassage(
                source_id=chunks[int(index)].id,
                filename=chunks[int(index)].filename,
                page_number=chunks[int(index)].page_number,
                text=chunks[int(index)].text,
                retrieval_score=float(similarities[claim_index, int(index)]),
            )
            for index in top_indexes
        ]

    evidence_by_claim: dict[str, list[EvidencePassage]] = {}
    reranker_used = True
    fallback_claim_ids: list[str] = []
    candidate_ids = {key: [p.source_id for p in value] for key, value in candidates_by_claim.items()}

    if len(chunks) == 1 or not use_reranker:
        return RetrievalResult(
            evidence_by_claim={
                claim.id: candidates_by_claim[claim.id][:RERANK_TOP_K]
                for claim in claims
            },
            reranker_used=False,
            candidate_source_ids=candidate_ids,
        )

    # Each claim is independent, so reranking in parallel keeps the interactive
    # demo responsive without changing the evidence selected for any claim.
    with ThreadPoolExecutor(max_workers=min(4, len(claims))) as executor:
        future_to_claim = {
            executor.submit(_rerank, claim.text, candidates_by_claim[claim.id]): claim
            for claim in claims
        }
        for future in as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                evidence_by_claim[claim.id] = future.result()
            except Exception:
                # A cosine fallback keeps the safety audit available during temporary
                # reranker outages, while PipelineDetails exposes the degraded mode.
                reranker_used = False
                fallback_claim_ids.append(claim.id)
                evidence_by_claim[claim.id] = candidates_by_claim[claim.id][:RERANK_TOP_K]

    return RetrievalResult(
        evidence_by_claim={
            claim.id: evidence_by_claim[claim.id]
            for claim in claims
        },
        reranker_used=reranker_used,
        candidate_source_ids=candidate_ids,
        fallback_claim_ids=sorted(fallback_claim_ids),
    )
