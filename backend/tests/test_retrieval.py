from app.services.retrieval import _lexical_overlap


def test_lexical_overlap_preserves_exact_fact_check_terms():
    query = "The pilot reduced staffing costs by 60 percent."
    relevant = "Staffing costs were unchanged; there was no 60 percent reduction."
    distractor = "The pilot reduced average editing time."

    assert _lexical_overlap(query, relevant) > _lexical_overlap(query, distractor)
