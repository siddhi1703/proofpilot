from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "statistic",
    "date",
    "person_or_organization",
    "causal",
    "comparison",
    "general_fact",
]
Importance = Literal["high", "medium", "low"]
Verdict = Literal[
    "supported",
    "weakly_supported",
    "unsupported",
    "contradicted",
    "human_review_required",
]
PublicationStatus = Literal["ready_with_review", "blocked"]
PrivacyRisk = Literal["low", "medium", "high"]


class ExtractedClaim(BaseModel):
    id: str = Field(description="Stable claim identifier, such as C1.")
    text: str = Field(min_length=3, max_length=1_000)
    type: ClaimType
    importance: Importance
    requires_source: bool = True


class ClaimBatch(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=12)


class SourceSummary(BaseModel):
    filename: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    character_count: int = Field(ge=1)


class PrepareAuditResponse(BaseModel):
    draft_characters: int = Field(ge=1)
    source: SourceSummary
    claims: list[ExtractedClaim]


class EvidencePassage(BaseModel):
    source_id: str
    filename: str
    page_number: int = Field(ge=1)
    text: str
    retrieval_score: float = Field(ge=-1, le=1)
    rerank_score: float | None = Field(default=None, ge=0, le=1)


class ModelVerdict(BaseModel):
    claim_id: str
    verdict: Verdict
    explanation: str = Field(min_length=3, max_length=1_500)
    evidence_quote: str = Field(default="", max_length=1_500)
    source_ids: list[str] = Field(default_factory=list, max_length=4)


class VerdictBatch(BaseModel):
    verdicts: list[ModelVerdict] = Field(default_factory=list, max_length=12)


class ClaimAudit(BaseModel):
    claim: ExtractedClaim
    verdict: Verdict
    explanation: str
    evidence_quote: str = ""
    citations: list[EvidencePassage] = Field(default_factory=list, max_length=4)
    needs_human_review: bool


class PrivacyFinding(BaseModel):
    kind: str
    label: str
    masked_value: str
    risk: PrivacyRisk
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    recommendation: str


class AuditSummary(BaseModel):
    total_claims: int = Field(ge=0)
    supported: int = Field(ge=0)
    weakly_supported: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    contradicted: int = Field(ge=0)
    human_review_required: int = Field(ge=0)
    privacy_findings: int = Field(ge=0)
    publication_status: PublicationStatus


class PipelineDetails(BaseModel):
    embedding_model: str
    rerank_model: str
    verdict_model: str
    retrieved_per_claim: int = Field(ge=1)
    reranked_per_claim: int = Field(ge=1)
    reranker_used: bool
    fallback_claim_ids: list[str] = Field(default_factory=list)


class AuditResponse(BaseModel):
    audit_id: str
    created_at: str
    draft_characters: int = Field(ge=1)
    source: SourceSummary
    summary: AuditSummary
    claim_audits: list[ClaimAudit]
    privacy_findings: list[PrivacyFinding]
    pipeline: PipelineDetails
    warnings: list[str] = Field(default_factory=list)
