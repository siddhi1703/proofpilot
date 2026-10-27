"""GPU-only three-class NLI diagnostic; separate from application verdict policy.

Uses saved retrieval rankings, never gold evidence to select model inputs.
Calibration is restricted to development; test requires its frozen configuration.
"""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

MODEL_ID = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import benchmark
from app.evaluation.metrics import score


def raw_windows(tokenizer, text, size=256, overlap=48, max_chars=1400):
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        verbose=False)['offset_mapping']
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        while end > start + 1 and offsets[end - 1][1] - offsets[start][0] > max_chars:
            end -= 1
        yield text[offsets[start][0]:offsets[end - 1][1]]
        if end == len(offsets):
            break
        start = max(end - overlap, start + 1)


def classify(row, top_k, threshold):
    pairs = [p for p in row['nli_windows'] if p['retrieval_rank'] <= top_k]
    if not pairs:
        return {**row, 'actual_verdict': None, 'evidence_quote': '', 'cited_source_ids': []}
    best = max(((p[label], label, p) for p in pairs for label in ['entailment', 'contradiction']),
               key=lambda item: item[0])
    value, label, window = best
    verdict = ({'entailment': 'supported', 'contradiction': 'contradicted'}[label]
               if value >= threshold else 'unsupported')
    return {**row, 'actual_verdict': verdict,
            'evidence_quote': window['text'] if verdict != 'unsupported' else '',
            'cited_source_ids': [window['source_id']] if verdict != 'unsupported' else []}


class NLIDiagnostic:
    def __init__(self):
        import torch
        from huggingface_hub import model_info
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not torch.cuda.is_available():
            raise RuntimeError('Run this diagnostic on a Colab GPU; no laptop fallback.')
        self.torch = torch
        self.revision = model_info(MODEL_ID).sha
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=self.revision, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError('Exact source-offset windowing requires a fast tokenizer.')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=self.revision, torch_dtype=torch.float16,
        ).to('cuda').eval()
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(self.labels) != {0, 1, 2} or set(self.labels.values()) != {'entailment', 'neutral', 'contradiction'}:
            raise ValueError('Unexpected NLI model label mapping.')

    def run(self, dataset_path, retrieval_path, output_path, *, development_report=None):
        dataset_path, retrieval_path, output_path = map(Path, [dataset_path, retrieval_path, output_path])
        if output_path.exists():
            raise ValueError('Preserve prior results: choose a new output filename.')
        data = benchmark.read_dataset(dataset_path)
        if development_report is None and data['split'] != 'development':
            raise ValueError('Test configuration must be frozen from a development report.')
        if {c['expected_verdict'] for case in data['cases'] for c in case['claims']} - {'supported', 'unsupported', 'contradicted'}:
            raise ValueError('This diagnostic supports only the three expert benchmark labels.')
        retrieval = json.loads(retrieval_path.read_text())
        digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if retrieval['dataset_sha256'] != digest or retrieval['partial_run']:
            raise ValueError('Retrieval report must cover exactly this complete dataset.')
        ranked_rows = retrieval['strategies']['rerank']['claim_results']
        rankings = {r['claim_id']: r for r in ranked_rows}
        expected_ids = {c['id'] for case in data['cases'] for c in case['claims']}
        if set(rankings) != expected_ids or len(rankings) != len(ranked_rows):
            raise ValueError('Retrieval claim IDs do not match the dataset.')
        rows, pairs, owners = [], [], []
        for case in data['cases']:
            passages = {p['id']: p for p in case['passages']}
            for claim in case['claims']:
                row = {**rankings[claim['id']], 'nli_windows': []}
                rows.append(row)
                for rank, sid in enumerate(row['ranked_source_ids'][:4], 1):
                    if sid not in passages:
                        raise ValueError('Unknown source in saved retrieval report.')
                    for text in raw_windows(self.tokenizer, passages[sid]['text']):
                        pairs.append((text, claim['text']))
                        owners.append((len(rows) - 1, sid, rank, text))
        torch = self.torch
        for start in range(0, len(pairs), 8):
            group = pairs[start:start + 8]
            encoded = self.tokenizer([p for p, h in group], [h for p, h in group],
                                     padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > 512:
                raise ValueError('NLI input exceeds 512 tokens; nothing was silently truncated.')
            with torch.inference_mode():
                distribution = self.model(**encoded.to('cuda')).logits.float().softmax(-1)
                if not torch.isfinite(distribution).all():
                    raise ValueError('NLI returned non-finite probabilities.')
                probabilities = distribution.cpu().tolist()
            for owner, values in zip(owners[start:start + 8], probabilities, strict=True):
                row_index, sid, rank, text = owner
                rows[row_index]['nli_windows'].append({
                    'source_id': sid, 'retrieval_rank': rank, 'text': text,
                    **{self.labels[i]: float(value) for i, value in enumerate(values)},
                })
            if start % 400 == 0:
                print(f'NLI windows: {min(start + 8, len(pairs))}/{len(pairs)}', flush=True)
        if development_report is None:
            settings = [{'top_k': k, 'threshold': t} for k in [1, 4] for t in [0.5, 0.65, 0.8, 0.9]]
        else:
            frozen = json.loads(Path(development_report).read_text())
            if frozen['split'] != 'development' or frozen['model_revision'] != self.revision:
                raise ValueError('Development profile does not match this model revision.')
            if frozen['code_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise ValueError('NLI implementation changed after development.')
            settings = [frozen['selected_configuration']]
        candidates = []
        for setting in settings:
            predictions = [classify(row, **setting) for row in rows]
            candidates.append({'configuration': setting, 'metrics': score(predictions, verdicts=True)})
        chosen = max(candidates, key=lambda c: (c['metrics']['macro_f1_observed_labels'],
                                               c['configuration']['threshold']))
        report = {
            'profile': 'three-class-nli-diagnostic', 'run_at': datetime.now(UTC).isoformat(),
            'dataset_id': data['dataset_id'], 'split': data['split'], 'dataset_sha256': digest,
            'claims': len(rows), 'model': MODEL_ID, 'model_revision': self.revision,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'retrieval_report_sha256': hashlib.sha256(retrieval_path.read_bytes()).hexdigest(),
            'selected_configuration': chosen['configuration'], 'metrics': chosen['metrics'],
            'development_candidates': candidates if development_report is None else [],
            'claim_results': [classify(row, **chosen['configuration']) for row in rows],
            'limitations': [
                'Separate three-class NLI baseline; does not implement all five application verdicts or production safety checks.',
                'Softmax values and tuned thresholds are not calibrated truth probabilities.',
                'Top-k and threshold selected on 100 development claims; test uses the frozen selection.',
                'Maximum entailment/contradiction over windows can amplify errors on irrelevant evidence.',
                'Controlled 16-abstract pools and public benchmark limitations still apply.',
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'selected': chosen['configuration'], 'metrics': chosen['metrics']}, indent=2))
        return report
