from dataclasses import dataclass
from pathlib import Path
import re

import fitz

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 25
CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 200


class DocumentParseError(ValueError):
    """Raised when an uploaded source cannot be safely parsed."""


@dataclass(frozen=True)
class SourceChunk:
    id: str
    text: str
    page_number: int
    filename: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    page_count: int
    character_count: int
    chunks: list[SourceChunk]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(
    text: str,
    *,
    filename: str,
    page_number: int,
    first_chunk_index: int,
) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    start = 0
    chunk_index = first_chunk_index

    while start < len(text):
        proposed_end = min(start + CHUNK_SIZE, len(text))
        end = proposed_end

        if proposed_end < len(text):
            minimum_boundary = start + CHUNK_SIZE // 2
            boundary_candidates = [
                text.rfind("\n\n", minimum_boundary, proposed_end),
                text.rfind(". ", minimum_boundary, proposed_end),
                text.rfind(" ", minimum_boundary, proposed_end),
            ]
            best_boundary = max(boundary_candidates)
            if best_boundary > minimum_boundary:
                end = best_boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                SourceChunk(
                    id=f"S{chunk_index}",
                    text=chunk_text,
                    page_number=page_number,
                    filename=filename,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break

        next_start = max(end - CHUNK_OVERLAP, start + 1)
        start = next_start

    return chunks


def parse_document(filename: str, file_bytes: bytes) -> ParsedDocument:
    if not file_bytes:
        raise DocumentParseError("The uploaded source file is empty.")

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        pages = _extract_pdf_pages(file_bytes)
    elif suffix == ".txt":
        pages = [_decode_text(file_bytes)]
    else:
        raise DocumentParseError("Only PDF and TXT source files are supported.")

    chunks: list[SourceChunk] = []
    character_count = 0

    for page_number, raw_text in enumerate(pages, start=1):
        page_text = normalize_text(raw_text)
        if not page_text:
            continue

        character_count += len(page_text)
        chunks.extend(
            chunk_page(
                page_text,
                filename=filename,
                page_number=page_number,
                first_chunk_index=len(chunks) + 1,
            )
        )

    if not chunks:
        raise DocumentParseError(
            "No readable text was found. Scanned PDFs require OCR and are not supported yet."
        )

    return ParsedDocument(
        filename=filename,
        page_count=len(pages),
        character_count=character_count,
        chunks=chunks,
    )


def _extract_pdf_pages(file_bytes: bytes) -> list[str]:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentParseError("The uploaded PDF is invalid or damaged.") from exc

    try:
        if document.page_count == 0:
            raise DocumentParseError("The uploaded PDF does not contain pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise DocumentParseError(
                f"PDFs are currently limited to {MAX_PDF_PAGES} pages."
            )

        return [page.get_text("text") for page in document]
    finally:
        document.close()


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise DocumentParseError("The text file encoding is not supported.")
