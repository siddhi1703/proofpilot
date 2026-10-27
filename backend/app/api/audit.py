from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.schemas.audit import AuditResponse, PrepareAuditResponse, SourceSummary
from app.services.audit_engine import run_audit
from app.services.document_parser import (
    DocumentParseError,
    MAX_FILE_BYTES,
    ParsedDocument,
    parse_document,
)
from app.services.qwen_client import (
    ClaimExtractionError,
    VerdictError,
    extract_claims,
)
from app.services.retrieval import RetrievalError
from app.services.provider_status import public_provider_problem

router = APIRouter(prefix="/api/audits", tags=["audits"])

MIN_DRAFT_CHARACTERS = 40
MAX_DRAFT_CHARACTERS = 20_000


def _check_provider_failure(exc: Exception) -> None:
    problem = public_provider_problem(exc)
    if problem:
        raise HTTPException(status_code=503, detail=problem) from exc
    if isinstance(exc, ClaimExtractionError) and "12-claim audit limit" in str(exc):
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validate_draft(draft: str) -> str:
    clean_draft = draft.strip()

    if len(clean_draft) < MIN_DRAFT_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Draft must contain at least {MIN_DRAFT_CHARACTERS} characters.",
        )

    if len(clean_draft) > MAX_DRAFT_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Draft cannot exceed {MAX_DRAFT_CHARACTERS} characters.",
        )

    return clean_draft


async def _read_document(source: UploadFile) -> ParsedDocument:
    file_bytes = await source.read(MAX_FILE_BYTES + 1)

    if len(file_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Source file cannot exceed 10 MB.",
        )

    try:
        document = await run_in_threadpool(
            parse_document,
            source.filename or "source",
            file_bytes,
        )
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return document


@router.post("/prepare", response_model=PrepareAuditResponse)
async def prepare_audit(
    draft: str = Form(...),
    source: UploadFile = File(...),
) -> PrepareAuditResponse:
    clean_draft = _validate_draft(draft)
    document = await _read_document(source)

    try:
        claims = await run_in_threadpool(extract_claims, clean_draft)
    except ClaimExtractionError as exc:
        _check_provider_failure(exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Qwen could not extract claims from this draft.",
        ) from exc

    return PrepareAuditResponse(
        draft_characters=len(clean_draft),
        source=SourceSummary(
            filename=document.filename,
            page_count=document.page_count,
            chunk_count=len(document.chunks),
            character_count=document.character_count,
        ),
        claims=claims,
    )


@router.post("/run", response_model=AuditResponse)
async def create_audit(
    draft: str = Form(...),
    source: UploadFile = File(...),
) -> AuditResponse:
    clean_draft = _validate_draft(draft)
    document = await _read_document(source)

    try:
        return await run_in_threadpool(run_audit, clean_draft, document)
    except ClaimExtractionError as exc:
        _check_provider_failure(exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Qwen could not extract factual claims from this draft.",
        ) from exc
    except RetrievalError as exc:
        _check_provider_failure(exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Qwen could not retrieve evidence from this source.",
        ) from exc
    except VerdictError as exc:
        _check_provider_failure(exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Qwen could not produce grounded evidence verdicts.",
        ) from exc
