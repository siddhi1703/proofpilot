from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re

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
            vectors.extend(item.embedding for item in ordered)
    except Exception as exc:
        raise RetrievalError("Qwen could not create evidence embeddings.") from exc

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
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
    for item in raw_results:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            continue
        score = item.get("relevance_score", item.get("score"))
        candidate = candidates[index]
        reranked.append(
            candidate.model_copy(
                update={
                    "rerank_score": (
                        max(0.0, min(1.0, float(score)))
                        if isinstance(score, (int, float))
                        else None
                    )
                }
            )
        )

    if not reranked:
        raise RetrievalError("Qwen reranker returned no usable passages.")

    # Fact checking is unusually sensitive to exact metrics and entities. Keep
    # Qwen's ordering, but reserve one of four slots for the strongest lexical
    # match when the neural reranker omitted it (often because it contains a
    # negation such as "did not measure").
    lexical_best = max(
        candidates,
        key=lambda candidate: _lexical_overlap(query, candidate.text),
    )
    if all(item.source_id != lexical_best.source_id for item in reranked):
        reranked = reranked[: max(0, RERANK_TOP_K - 1)] + [lexical_best]

    return reranked[:RERANK_TOP_K]


def retrieve_evidence(
    claims: list[ExtractedClaim],
    chunks: list[SourceChunk],
) -> RetrievalResult:
    if not claims:
        return RetrievalResult(evidence_by_claim={}, reranker_used=True)
    if not chunks:
        raise RetrievalError("No source chunks are available for retrieval.")

    # One combined embedding request avoids a second network round trip when
    # the source and claims fit in the same API batch.
    combined_vectors = embed_texts(
        [chunk.text for chunk in chunks] + [claim.text for claim in claims]
    )
    chunk_vectors = combined_vectors[: len(chunks)]
    claim_vectors = combined_vectors[len(chunks) :]
    similarities = claim_vectors @ chunk_vectors.T

    candidates_by_claim: dict[str, list[EvidencePassage]] = {}
    for claim_index, claim in enumerate(claims):
        top_count = min(RETRIEVAL_TOP_K, len(chunks))
        top_indexes = np.argsort(similarities[claim_index])[::-1][:top_count]
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

    if len(chunks) == 1:
        return RetrievalResult(
            evidence_by_claim={
                claim.id: candidates_by_claim[claim.id]
                for claim in claims
            },
            reranker_used=False,
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
                evidence_by_claim[claim.id] = candidates_by_claim[claim.id][:RERANK_TOP_K]

    return RetrievalResult(
        evidence_by_claim={
            claim.id: evidence_by_claim[claim.id]
            for claim in claims
        },
        reranker_used=reranker_used,
    )
