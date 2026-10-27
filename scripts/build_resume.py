"""Recover checkpoint data from notebook outputs without executing notebook code."""
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class BackupTextareas(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = None
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag == 'textarea':
            self.active = [dict(attrs).get('aria-label', ''), '']

    def handle_data(self, data):
        if self.active is not None:
            self.active[1] += data

    def handle_endtag(self, tag):
        if tag == 'textarea' and self.active is not None:
            self.values.append(self.active)
            self.active = None


def output_text(output):
    value = output.get('text', '')
    return ''.join(value) if isinstance(value, list) else value


def extract_backup(notebook):
    candidates = []
    for cell in notebook.get('cells', []):
        metadata, backups = [], []
        for output in cell.get('outputs', []):
            for line in output_text(output).splitlines():
                try:
                    item = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(item, dict) and type(item.get('completed')) is int:
                    metadata.append(item)
            html = output.get('data', {}).get('text/html', '')
            html = ''.join(html) if isinstance(html, list) else html
            parser = BackupTextareas()
            parser.feed(html)
            for label, encoded in parser.values:
                match = re.fullmatch(r'ProofPilot test backup (\d+)', label)
                if match:
                    backups.append((int(match[1]), encoded))
        for completed, encoded in backups:
            candidates.append((completed, encoded, metadata, cell))
    if not candidates:
        raise ValueError('No portable ProofPilot test checkpoint found in notebook outputs.')
    completed, encoded, metadata, cell = max(candidates, key=lambda item: item[0])
    content = base64.b64decode(encoded.strip(), validate=True)
    digest = hashlib.sha256(content).hexdigest()
    if not any(m.get('completed') == completed and m.get('bytes') == len(content)
               and m.get('sha256') == digest for m in metadata):
        raise ValueError('Notebook backup does not match its recorded size and checksum.')
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        members = bundle.infolist()
        if len(members) > 20_000 or sum(item.file_size for item in members) > 128 * 1024 * 1024:
            raise ValueError('Checkpoint archive exceeds its size or file-count limit.')
        if bundle.testzip() is not None:
            raise ValueError('Corrupted checkpoint archive.')
    progress = [int(match.group(1)) for output in cell.get('outputs', [])
                for match in re.finditer(r'Saved checkpoint for cases \d+-(\d+)', output_text(output))]
    return content, {'portable_cases': completed, 'logged_cases': max([completed, *progress]),
                     'archive_sha256': digest}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fingerprint():
    paths = sorted((ROOT / 'backend/app').rglob('*.py')) + [ROOT / 'backend/benchmark.py', ROOT / 'backend/colab_runner.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build(notebook_path, destination):
    notebook_path, destination = Path(notebook_path), Path(destination)
    if notebook_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Unexpectedly large notebook.')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Choose a new or empty destination to preserve previous recovery files.')
    notebook = json.loads(notebook_path.read_text())
    content, status = extract_backup(notebook)
    destination.mkdir(parents=True, exist_ok=True)
    legacy = destination / 'original-test-backup.zip'
    legacy.write_bytes(content)
    storage = load_module('proofpilot_checkpoint_storage', ROOT / 'colab/checkpoints.py')
    outputs = destination / 'recovered-outputs'
    storage.restore_checkpoint(legacy, outputs, status['archive_sha256'])
    reports = [(path, json.loads(path.read_text())) for path in outputs.glob('test/*/report-*-cases.json')]
    if not reports:
        raise ValueError('Archive contains no test report.')
    report_path, report = max(reports, key=lambda pair: pair[1]['completed_cases'])
    config = report['configuration']
    dataset_path = ROOT / 'evaluation/datasets/scifact-v2/test.json'
    if (report['completed_cases'] != status['portable_cases'] or report['dataset_total_cases'] != 300
            or config['dataset_sha256'] != hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            or config['code_sha256'] != source_fingerprint()
            or config['mode'] != 'rerank' or config['verdict_batch_size'] != 1 or config.get('verdicts') is not True):
        raise ValueError('Checkpoint does not match the current frozen test implementation.')
    if json.loads((report_path.parent / 'configuration.json').read_text()) != config:
        raise ValueError('Checkpoint configuration and report disagree.')
    identity = {key: config[key] for key in ['dataset_sha256', 'code_sha256', 'models', 'mode', 'verdicts', 'verdict_batch_size']}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] != report_path.parent.name:
        raise ValueError('Checkpoint run identity is invalid.')
    dataset = json.loads(dataset_path.read_text())
    rows = []
    for start in range(0, report['completed_cases'], 5):
        group = dataset['cases'][start:min(start + 5, report['completed_cases'])]
        group_id = hashlib.sha256(json.dumps([case['id'] for case in group]).encode()).hexdigest()[:16]
        chunk = json.loads((report_path.parent / f'chunk-{group_id}.json').read_text())
        if chunk['evaluated_cases'] != len(group):
            raise ValueError('Checkpoint chunk size does not match the frozen cases.')
        rows.extend(chunk['strategies']['rerank']['claim_results'])
    if rows != report['strategies']['rerank']['claim_results']:
        raise ValueError('Combined report does not match its completed chunks.')
    gold = {claim['id']: claim for case in dataset['cases'][:report['completed_cases']] for claim in case['claims']}
    if len(rows) != len(gold) or {row['claim_id'] for row in rows} != set(gold):
        raise ValueError('Completed claims do not match the frozen dataset.')
    for row in rows:
        if row['expected_verdict'] != gold[row['claim_id']]['expected_verdict'] or row['expected_source_ids'] != gold[row['claim_id']]['expected_source_ids']:
            raise ValueError('Checkpoint gold labels differ from the pinned dataset.')
    model_ids = {'sentence-transformers/all-MiniLM-L6-v2', 'cross-encoder/ms-marco-MiniLM-L6-v2', 'Qwen/Qwen3-4B-Instruct-2507'}
    revisions = config['models']['revisions']
    if set(revisions) != model_ids or not all(re.fullmatch('[0-9a-f]{40}', r) for r in revisions.values()):
        raise ValueError('Unexpected model profile or revision.')
    status.update({'source_notebook_sha256': hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                   'code_sha256': config['code_sha256'], 'run_id': report_path.parent.name,
                   'remaining_cases': 300 - report['completed_cases'], 'partial_run': report['partial_run']})
    (destination / 'recovery-status.json').write_text(json.dumps(status, indent=2) + '\n')
    portable = storage.create_checkpoint(outputs, destination / 'proofpilot-resume-checkpoints.zip')
    builder = load_module('proofpilot_notebook_builder', ROOT / 'scripts/build_colab.py')
    with tempfile.TemporaryDirectory(prefix='proofpilot-source-') as temporary:
        builder.DEST = Path(temporary)
        builder.build()
        template = json.loads((builder.DEST / 'ProofPilot_Free_Colab.ipynb').read_text())
        shutil.copy2(builder.DEST / 'proofpilot-colab-bundle.zip', destination / 'proofpilot-resume-source.zip')
    def original_code(marker):
        return next(''.join(c['source']) for c in template['cells']
                    if c['cell_type'] == 'code' and marker in ''.join(c['source']))
    storage_code = original_code('USE_DRIVE = True').replace(
        "BUNDLE = STORAGE / 'proofpilot-colab-bundle.zip'", "BUNDLE = Path('/content/proofpilot-resume-source.zip')").replace(
        "BUNDLE = Path('/content/proofpilot-colab-bundle.zip')", "BUNDLE = Path('/content/proofpilot-resume-source.zip')")
    restore_code = original_code('checkpoint_spec =').replace(
        "RESTORE_CHECKPOINT = ''", "RESTORE_CHECKPOINT = '/content/proofpilot-resume-checkpoints.zip'").replace(
        "RESTORE_SHA256 = ''", f"RESTORE_SHA256 = {hashlib.sha256(portable.read_bytes()).hexdigest()!r}")
    environment = json.loads((outputs / 'environment-colab.json').read_text())
    pins = [name + '==' + value for name, value in environment['packages'].items()
            if name != 'torch' and value != 'not-installed']
    dependencies = f'''import subprocess, importlib.metadata
assert importlib.metadata.version('torch') == {config['packages']['torch']!r}, 'Torch differs from the saved run. Use the matching Colab runtime; do not mix experiment stacks.'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT / 'colab/requirements-colab.txt')])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{pins!r}])
for distribution, expected in {environment['packages']!r}.items():
    if expected != 'not-installed':
        assert importlib.metadata.version(distribution) == expected
for module_name in ['torch', 'numpy', 'pydantic', 'transformers', 'tokenizers']:
    loaded = sys.modules.get(module_name)
    if loaded is not None and getattr(loaded, '__version__', None) != importlib.metadata.version(module_name):
        raise RuntimeError('Packages were updated. Restart the runtime and run all cells again before inference.')
os.environ['HF_HOME'] = '/content/proofpilot-hf-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.pop('DASHSCOPE_API_KEY', None)
BACKEND = PROJECT / 'backend'
os.chdir(BACKEND)
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))'''
    setup = f'''from unittest.mock import patch
from huggingface_hub import model_info as resolve_model_info
from app.evaluation.colab_models import ColabModels
import colab_runner
from colab_runner import run_resumable
frozen_configuration = json.loads({json.dumps(config)!r})
assert colab_runner.code_fingerprint() == frozen_configuration['code_sha256']
def pinned_model_info(repo_id, *args, **kwargs):
    kwargs['revision'] = frozen_configuration['models']['revisions'][repo_id]
    return resolve_model_info(repo_id, *args, **kwargs)
with patch('huggingface_hub.model_info', pinned_model_info):
    models = ColabModels()
assert models.metadata == frozen_configuration['models'], 'Select the same T4 GPU and frozen model profile.'
atomic_json = getattr(colab_runner.atomic_json, '_proofpilot_original', colab_runner.atomic_json)
def save_checkpoint(path, value):
    atomic_json(path, value)
    if path.name.startswith('report-') and not USE_DRIVE:
        checkpoint_storage.create_checkpoint(OUTPUTS, '/content/proofpilot-latest-checkpoints.zip')
save_checkpoint._proofpilot_original = atomic_json
colab_runner.atomic_json = save_checkpoint
print('Frozen model revisions and inference code verified.')'''
    resume = f'''test_dataset = PROJECT / 'evaluation/datasets/scifact-v2/test.json'
directory = OUTPUTS / 'test' / {status['run_id']!r}
assert json.loads((directory / 'configuration.json').read_text()) == frozen_configuration
saved_reports = [json.loads(p.read_text()) for p in directory.glob('report-*-cases.json')]
report = max(saved_reports, key=lambda item: item['completed_cases'])
print('Restored completed cases:', report['completed_cases'])
for limit in [100, 150, 200, 250, 300]:
    if limit <= report['completed_cases']:
        continue
    report = run_resumable(test_dataset, models, OUTPUTS, limit=limit, mode='rerank')
    backup_folder = STORAGE if USE_DRIVE else Path('/content')
    backup = checkpoint_storage.create_checkpoint(OUTPUTS, backup_folder / ('proofpilot-test-' + str(report['completed_cases']) + '.zip'))
    print('BACKUP:', backup, 'SHA256:', hashlib.sha256(backup.read_bytes()).hexdigest(), flush=True)
    print('Completed:', report['completed_cases'], '/', report['dataset_total_cases'], flush=True)
assert report['completed_cases'] == 300 and not report['partial_run']
print('FULL QWEN TEST COMPLETE')
print(json.dumps(report['strategies']['rerank']['metrics'], indent=2))'''
    cells = [builder.markdown(f'''# ProofPilot: resume the frozen Qwen test

Recovered **{status['portable_cases']}/300** portable cases. The previous log reached {status['logged_cases']}, but only archived results are reused. This notebook runs the remaining {status['remaining_cases']} cases on a free Colab T4; it does not run models on your laptop.

Select **Runtime > Change runtime type > T4 GPU**. Upload `proofpilot-resume-source.zip` and `proofpilot-resume-checkpoints.zip` through Colab's **Files > Upload to session storage**, then run the cells in order. Drive storage is the default so progress survives a recycled runtime. Use the same Google account containing your ProofPilot Colab folder. No paid API fallback is enabled.'''),
        builder.code(original_code('import os, sys')),
        builder.markdown('## Storage\nApprove Drive mounting when prompted. If it fails, change `USE_DRIVE` to `False`; session-mode backups must be downloaded before disconnecting.'),
        builder.code(storage_code), builder.markdown('## Verify the source bundle'),
        builder.code(original_code('EXPECTED_SHA256 =')), builder.markdown('## Restore the verified checkpoint'),
        builder.code(restore_code), builder.markdown('## Restore the recorded software versions'),
        builder.code(dependencies), builder.markdown('## Check the implementation'),
        builder.code(original_code("'-m', 'pytest'")), builder.markdown('## Restore the exact model revisions'),
        builder.code(setup), builder.markdown('## Resume the remaining test claims\nCompleted chunks are reused. Errors remain in the score; no prompts, thresholds or labels are tuned on test outcomes.'),
        builder.code(resume), builder.markdown('## Save the final archive\nWith Drive storage, the final backup is in **My Drive / ProofPilot Colab / proofpilot-test-300.zip**. With session storage, run this download before disconnecting. Keep the archive and its printed SHA-256.'),
        builder.code(original_code('from google.colab import files'))]
    for index, cell in enumerate(cells):
        cell['id'] = f'proofpilot-resume-{index:02d}'
    template['cells'] = cells
    template['metadata']['colab']['name'] = 'ProofPilot_Resume_Qwen.ipynb'
    (destination / 'ProofPilot_Resume_Qwen.ipynb').write_text(json.dumps(template, indent=2) + '\n')
    (destination / 'START_HERE.txt').write_text(
        '1. Open ProofPilot_Resume_Qwen.ipynb in Google Colab using your project account.\n'
        '2. Select a free T4 GPU and upload the two proofpilot-resume ZIPs through the Files panel.\n'
        f"3. Run the cells in order. The first {status['portable_cases']} test cases are reused; the other {status['remaining_cases']} are evaluated.\n"
        '4. Keep Drive storage enabled for durable results. Save proofpilot-test-300.zip when complete.\n')
    package = destination / 'ProofPilot_Resume_Package.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ['ProofPilot_Resume_Qwen.ipynb', 'proofpilot-resume-source.zip',
                     'proofpilot-resume-checkpoints.zip', 'recovery-status.json', 'START_HERE.txt']:
            bundle.write(destination / name, name)
    print(json.dumps({'package': str(package), **status}, indent=2))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.notebook, arguments.output)
