from app.schemas.audit import EvidencePassage
from app.services.qwen_client import _quote_is_grounded


def test_quote_must_be_exact_source_text():
    passages = [
        EvidencePassage(
            source_id="S1",
            filename="study.txt",
            page_number=1,
            text="The pilot included 120 editors across three newsrooms.",
            retrieval_score=0.9,
        )
    ]

    assert _quote_is_grounded("included 120 editors", passages)
    assert not _quote_is_grounded("included about 120 journalists", passages)
    assert not _quote_is_grounded("", passages)
