from app.schemas.audit import EvidencePassage
from app.services.qwen_client import _find_exact_source_sentence, _quote_is_grounded


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


def test_exact_sentence_fallback_recovers_a_paraphrased_model_quote():
    passages = [
        EvidencePassage(
            source_id="S6",
            filename="optimization.pdf",
            page_number=4,
            text=(
                "A feasible solution is a candidate solution that satisfies all "
                "constraints. Other candidates are infeasible."
            ),
            retrieval_score=0.8,
        )
    ]

    quote = _find_exact_source_sentence(
        "A feasible solution satisfies all constraints.",
        passages,
    )

    assert quote == (
        "A feasible solution is a candidate solution that satisfies all constraints."
    )
    assert _quote_is_grounded(quote, passages)


def test_exact_sentence_fallback_rejects_loosely_related_evidence():
    passages = [
        EvidencePassage(
            source_id="S1",
            filename="optimization.pdf",
            page_number=1,
            text="An objective function is minimized or maximized.",
            retrieval_score=0.5,
        )
    ]

    assert (
        _find_exact_source_sentence(
            "A feasible solution satisfies all constraints.",
            passages,
        )
        == ""
    )
