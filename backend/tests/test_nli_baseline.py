import importlib.util
from pathlib import Path
import re

path = Path(__file__).resolve().parents[2] / 'colab/nli_baseline.py'
spec = importlib.util.spec_from_file_location('nli_diagnostic_for_test', path)
nli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nli)


def test_nli_windows_preserve_exact_source_spans_including_final_negation():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}
    text = ' '.join('long_scientific_word_' + str(i) for i in range(700)) + '\nIt did not improve.'
    windows = list(nli.raw_windows(Tokenizer(), text))
    assert all(window in text and len(window) <= 1400 for window in windows)
    assert set(' '.join(windows).split()) == set(text.split())
    assert windows[-1].endswith('It did not improve.')


def test_nli_top_one_configuration_does_not_use_lower_ranked_distractors():
    row = {'nli_windows': [
        {'source_id': 'first', 'retrieval_rank': 1, 'text': 'Exact source.',
         'entailment': 0.91, 'contradiction': 0.04},
        {'source_id': 'distractor', 'retrieval_rank': 2, 'text': 'Another statement.',
         'entailment': 0.01, 'contradiction': 0.99},
    ]}
    result = nli.classify(row, top_k=1, threshold=0.5)
    assert result['actual_verdict'] == 'supported'
    assert result['cited_source_ids'] == ['first']
    assert result['evidence_quote'] == 'Exact source.'


def test_nli_missing_evidence_is_missing_prediction_and_low_scores_do_not_invent_quotes():
    missing = nli.classify({'nli_windows': []}, top_k=1, threshold=0.5)
    assert missing['actual_verdict'] is None
    row = {'nli_windows': [{'source_id': 'S1', 'retrieval_rank': 1, 'text': 'A source.',
                           'entailment': 0.2, 'contradiction': 0.1}]}
    result = nli.classify(row, top_k=1, threshold=0.5)
    assert result['actual_verdict'] == 'unsupported'
    assert result['evidence_quote'] == '' and result['cited_source_ids'] == []
