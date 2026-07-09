import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.schemas.audit import (
    ClaimAudit,
    ClaimBatch,
    EvidencePassage,
    ExtractedClaim,
    ModelVerdict,
    VerdictBatch,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_NAME = os.getenv("QWEN_TEXT_MODEL", "qwen3.6-flash")
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class ClaimExtractionError(RuntimeError):
    """Raised when claim extraction fails or produces invalid output."""


class VerdictError(RuntimeError):
    """Raised when evidence verdict generation fails."""


def get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ClaimExtractionError("DASHSCOPE_API_KEY is not configured.")
    return api_key


def get_client() -> OpenAI:
    return OpenAI(
        api_key=get_api_key(),
        base_url=BASE_URL,
        timeout=90,
        max_retries=2,
    )


def extract_claims(draft: str) -> list[ExtractedClaim]:
    system_prompt = """
You are the claim-extraction component of ProofPilot, a responsible publishing
audit system. Extract only externally verifiable factual claims from the draft.

Include statistics, dates, named people or organizations, causal statements,
comparisons, and concrete general facts. Exclude opinions, recommendations,
headings, rhetorical questions, and purely subjective language.

Return a JSON object with this exact structure:
{
  "claims": [
    {
      "id": "C1",
      "text": "A complete standalone factual claim.",
      "type": "statistic | date | person_or_organization | causal | comparison | general_fact",
      "importance": "high | medium | low",
      "requires_source": true
    }
  ]
}

Return no more than 12 claims. Preserve numbers, names, dates, and qualifiers
exactly. Output JSON only.
""".strip()

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": draft},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if not content:
            raise ClaimExtractionError("Qwen returned an empty response.")

        batch = ClaimBatch.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError, IndexError, AttributeError) as exc:
        raise ClaimExtractionError("Qwen returned an invalid claim structure.") from exc
    except ClaimExtractionError:
        raise
    except Exception as exc:
        raise ClaimExtractionError("Qwen claim extraction request failed.") from exc

    normalized_claims: list[ExtractedClaim] = []
    for index, claim in enumerate(batch.claims, start=1):
        normalized_claims.append(claim.model_copy(update={"id": f"C{index}"}))

    return normalized_claims


def _evidence_prompt(
    claims: list[ExtractedClaim],
    evidence_by_claim: dict[str, list[EvidencePassage]],
) -> str:
    sections: list[str] = []
    for claim in claims:
        passages = evidence_by_claim.get(claim.id, [])
        rendered_passages = "\n".join(
            (
                f"[{passage.source_id} | {passage.filename} | page "
                f"{passage.page_number}]\n{passage.text}"
            )
            for passage in passages
        )
        sections.append(
            f"CLAIM {claim.id}: {claim.text}\n"
            f"IMPORTANCE: {claim.importance}\n"
            f"CANDIDATE EVIDENCE:\n{rendered_passages or '(none)'}"
        )
    return "\n\n---\n\n".join(sections)


def _quote_is_grounded(quote: str, passages: list[EvidencePassage]) -> bool:
    normalized_quote = " ".join(quote.split()).casefold()
    if not normalized_quote:
        return False
    return any(
        normalized_quote in " ".join(passage.text.split()).casefold()
        for passage in passages
    )


QUOTE_STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in QUOTE_STOP_WORDS and len(token) > 1
    }


def _find_exact_source_sentence(
    claim_text: str,
    passages: list[EvidencePassage],
) -> str:
    """Return a high-overlap exact source sentence, or fail closed."""
    claim_tokens = _content_tokens(claim_text)
    if len(claim_tokens) < 2:
        return ""

    best_quote = ""
    best_score = 0.0

    for passage in passages:
        raw_candidates = re.split(r"(?<=[.!?])\s+|\n+", passage.text)
        candidates = [" ".join(candidate.split()) for candidate in raw_candidates]

        for candidate in candidates:
            if len(candidate) < 12 or len(candidate) > 1_500:
                continue

            candidate_tokens = _content_tokens(candidate)
            coverage = len(claim_tokens & candidate_tokens) / len(claim_tokens)
            if coverage > best_score or (
                coverage == best_score and best_quote and len(candidate) < len(best_quote)
            ):
                best_score = coverage
                best_quote = candidate

    # Requiring at least 75% content-token coverage prevents a loosely related
    # passage from being presented as evidence merely to avoid human review.
    if best_score >= 0.75 and _quote_is_grounded(best_quote, passages):
        return best_quote
    return ""


def evaluate_claims(
    claims: list[ExtractedClaim],
    evidence_by_claim: dict[str, list[EvidencePassage]],
) -> list[ClaimAudit]:
    if not claims:
        return []

    system_prompt = """
You are ProofPilot's evidence verdict engine. Judge each factual claim using
ONLY its candidate evidence. Never use outside knowledge and never assume a
missing fact is true.

Use exactly one verdict:
- supported: the evidence directly establishes the complete claim.
- weakly_supported: related evidence exists but a qualifier, scope, number, or
  causal link is not fully established.
- unsupported: the evidence does not establish the claim.
- contradicted: the evidence provides a mutually exclusive opposite fact,
  countervalue, or explicit denial of the claim.
- human_review_required: the evidence is ambiguous, internally inconsistent,
  or too risky to decide safely.

If a source merely says that a metric was not measured, evaluated, or assessed,
the claim is unsupported rather than contradicted. Use contradicted only when
the source establishes the opposite.

For supported, weakly_supported, and contradicted verdicts, evidence_quote must
be a short EXACT copy from one provided passage. Never invent or paraphrase a
quote. For unsupported, evidence_quote must be an empty string. source_ids may
only contain IDs shown with that claim.

Return JSON only:
{
  "verdicts": [
    {
      "claim_id": "C1",
      "verdict": "supported | weakly_supported | unsupported | contradicted | human_review_required",
      "explanation": "One concise explanation grounded only in the evidence.",
      "evidence_quote": "Exact source text or empty string.",
      "source_ids": ["S1"]
    }
  ]
}
Return exactly one verdict for every supplied claim.
""".strip()

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _evidence_prompt(claims, evidence_by_claim)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise VerdictError("Qwen returned an empty verdict response.")
        batch = VerdictBatch.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError, IndexError, AttributeError) as exc:
        raise VerdictError("Qwen returned an invalid verdict structure.") from exc
    except VerdictError:
        raise
    except Exception as exc:
        raise VerdictError("Qwen verdict request failed.") from exc

    verdict_by_claim: dict[str, ModelVerdict] = {
        verdict.claim_id: verdict for verdict in batch.verdicts
    }
    audits: list[ClaimAudit] = []

    for claim in claims:
        passages = evidence_by_claim.get(claim.id, [])
        allowed_by_id = {passage.source_id: passage for passage in passages}
        model_verdict = verdict_by_claim.get(claim.id)

        if model_verdict is None:
            audits.append(
                ClaimAudit(
                    claim=claim,
                    verdict="human_review_required",
                    explanation="The model did not return a verdict for this claim.",
                    needs_human_review=True,
                )
            )
            continue

        citations = [
            allowed_by_id[source_id]
            for source_id in model_verdict.source_ids
            if source_id in allowed_by_id
        ]
        quote_required = model_verdict.verdict in {
            "supported",
            "weakly_supported",
            "contradicted",
        }
        quote_is_valid = _quote_is_grounded(model_verdict.evidence_quote, citations)
        grounded_quote = (
            model_verdict.evidence_quote
            if quote_is_valid
            else _find_exact_source_sentence(claim.text, citations)
        )

        if quote_required and (not citations or not grounded_quote):
            audits.append(
                ClaimAudit(
                    claim=claim,
                    verdict="human_review_required",
                    explanation=(
                        "The model verdict could not be tied to a verifiable exact "
                        "quote, so ProofPilot paused it for human review."
                    ),
                    citations=citations,
                    needs_human_review=True,
                )
            )
            continue

        needs_review = model_verdict.verdict != "supported"
        audits.append(
            ClaimAudit(
                claim=claim,
                verdict=model_verdict.verdict,
                explanation=model_verdict.explanation,
                evidence_quote=grounded_quote,
                citations=citations,
                needs_human_review=needs_review,
            )
        )

    return audits
