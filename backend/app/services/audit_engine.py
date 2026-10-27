from collections import Counter
from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.audit import (
    AuditResponse,
    AuditSummary,
    PipelineDetails,
    SourceSummary,
)
from app.services.document_parser import ParsedDocument
from app.services.privacy import scan_privacy
from app.services.qwen_client import MODEL_NAME, evaluate_claims, extract_claims
from app.services.retrieval import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    retrieve_evidence,
)


def run_audit(draft: str, document: ParsedDocument) -> AuditResponse:
    claims = extract_claims(draft)
    retrieval = retrieve_evidence(claims, document.chunks)
    claim_audits = evaluate_claims(claims, retrieval.evidence_by_claim)
    privacy_findings = scan_privacy(draft)

    verdict_counts = Counter(audit.verdict for audit in claim_audits)
    has_blocker = not claim_audits or any(
        audit.needs_human_review
        or
        audit.verdict in {"unsupported", "contradicted", "human_review_required"}
        or (audit.claim.importance == "high" and audit.verdict != "supported")
        for audit in claim_audits
    ) or any(finding.risk == "high" for finding in privacy_findings)

    return AuditResponse(
        audit_id=f"audit_{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        draft_characters=len(draft),
        source=SourceSummary(
            filename=document.filename,
            page_count=document.page_count,
            chunk_count=len(document.chunks),
            character_count=document.character_count,
        ),
        summary=AuditSummary(
            total_claims=len(claim_audits),
            supported=verdict_counts["supported"],
            weakly_supported=verdict_counts["weakly_supported"],
            unsupported=verdict_counts["unsupported"],
            contradicted=verdict_counts["contradicted"],
            human_review_required=verdict_counts["human_review_required"],
            privacy_findings=len(privacy_findings),
            publication_status="blocked" if has_blocker else "ready_with_review",
        ),
        claim_audits=claim_audits,
        privacy_findings=privacy_findings,
        pipeline=PipelineDetails(
            embedding_model=EMBEDDING_MODEL,
            rerank_model=RERANK_MODEL,
            verdict_model=MODEL_NAME,
            retrieved_per_claim=RETRIEVAL_TOP_K,
            reranked_per_claim=RERANK_TOP_K,
            reranker_used=retrieval.reranker_used,
            fallback_claim_ids=retrieval.fallback_claim_ids,
        ),
        warnings=(
            (["No factual claims were extracted; this draft has not been verified."] if not claims else [])
            + (["Reranking failed for some claims; cosine retrieval was used."] if retrieval.fallback_claim_ids else [])
        ),
    )
