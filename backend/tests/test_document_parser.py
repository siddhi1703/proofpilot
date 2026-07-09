from app.services.document_parser import normalize_text, parse_document


def test_parse_text_document_preserves_source_metadata():
    document = parse_document(
        "study.txt",
        (
            b"The newsroom pilot included 120 participants. "
            b"Editing time improved by 16 percent during the study."
        ),
    )

    assert document.filename == "study.txt"
    assert document.page_count == 1
    assert document.character_count > 0
    assert len(document.chunks) == 1
    assert document.chunks[0].filename == "study.txt"
    assert document.chunks[0].page_number == 1


def test_normalize_text_repairs_pdf_line_break_hyphenation():
    assert normalize_text("A feasible so-\nlution satisfies constraints.") == (
        "A feasible solution satisfies constraints."
    )
