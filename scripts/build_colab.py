"""Build a secret-excluding backend/data archive and a standard Jupyter notebook."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "colab"


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    text = text.strip() + "\n"
    compile(text, "notebook-cell", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build():
    DEST.mkdir(exist_ok=True)
    fixed = ["backend/benchmark.py", "backend/colab_runner.py", "backend/main.py", "backend/pytest.ini",
             "colab/requirements-colab.txt", "colab/nli_baseline.py", "colab/checkpoints.py", "colab/README.md", "evaluation/README.md", "README.md", "LICENSE"]
    paths = [ROOT / p for p in fixed]
    paths += list((ROOT / "backend/app").rglob("*.py"))
    paths += list((ROOT / "backend/tests").rglob("*.py"))
    paths += [p for p in (ROOT / "evaluation/datasets").rglob("*") if p.suffix in {".json", ".md"}]
    # The legacy evaluator and its tests remain runnable in the isolated bundle.
    paths += [ROOT / "backend/evaluate.py", ROOT / "evaluation/benchmark.json", ROOT / "evaluation/results.json"]
    paths = sorted(set(paths))
    files = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.is_symlink() or any(part.startswith(".") for part in Path(relative).parts):
            raise ValueError(f"Unexpected bundle source: {relative}")
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"format_version": 1, "source_git_base_revision": revision,
                "includes_uncommitted_source": True, "files": files}
    archive = DEST / "proofpilot-colab-bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cells = [markdown('''# ProofPilot: free cloud evaluation

Heavy processing runs on **Google Colab's GPU**, not your laptop. No paid model API or AWS account is used.

Use **Runtime > Change runtime type > T4 GPU** before running the cells. Keep the free plan. Choose Drive storage or upload the bundle into the remote session.

The default experiment is a **10-case development pilot**. Full 100-case development and 300-case test runs are separate optional cells. Do not tune on test results. Checkpoints save every five cases.

Free GPUs and session length are limited. If none is available, stop and try later. This is an interactive notebook, not permanent web hosting.'''),
    code('''import os, sys
from pathlib import Path
import torch
if not Path('/content').is_dir() or not torch.cuda.is_available():
    raise RuntimeError('Use a Colab-hosted GPU runtime: Runtime > Change runtime type > T4 GPU. No laptop or paid-API fallback is enabled.')
print('Colab GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))'''),
    markdown('''## 1. Choose cloud storage

With `USE_DRIVE = True`, use the account containing **My Drive / ProofPilot Colab** and approve Google's Drive prompt. The notebook reads the bundle and saves checkpoints there.

If Drive reports a credential-propagation error, set `USE_DRIVE = False`. Open Colab's **Files** panel and use **Upload to session storage** to upload `proofpilot-colab-bundle.zip`. Results then live in temporary cloud storage: export the checkpoints before the runtime ends. To resume later, use the **Restore a saved run** cell below.'''),
    code('''USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/ProofPilot Colab')
    BUNDLE = STORAGE / 'proofpilot-colab-bundle.zip'
else:
    STORAGE = Path('/content/proofpilot-session')
    BUNDLE = Path('/content/proofpilot-colab-bundle.zip')
    STORAGE.mkdir(exist_ok=True)
assert BUNDLE.is_file(), 'Upload the matching project ZIP to the selected location, then rerun this cell.'
OUTPUTS = STORAGE / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
print('Results will be saved in:', OUTPUTS)
if not USE_DRIVE:
    print('Temporary session storage: export checkpoint backups before disconnecting.')'''),
    markdown('''## 2. Verify and unpack the current implementation

This is the locally reviewed source, including changes not yet pushed to GitHub. The bundle excludes API keys and private files. A checksum prevents silently loading another archive.'''),
    code(f'''import hashlib, json, zipfile
EXPECTED_SHA256 = {digest!r}
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == EXPECTED_SHA256, 'Bundle checksum mismatch. Use the ZIP delivered with this notebook.'
PROJECT = Path('/content/proofpilot-colab')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as bundle:
    manifest = json.loads(bundle.read('bundle-manifest.json'))
    assert sum(i.file_size for i in bundle.infolist()) < 30_000_000, 'Unexpectedly large bundle.'
    assert set(bundle.namelist()) == set(manifest['files']) | {{'bundle-manifest.json'}}
    for name, expected in manifest['files'].items():
        target = (PROJECT / name).resolve()
        assert target.is_relative_to(PROJECT.resolve()), 'Unsafe archive path.'
        content = bundle.read(name)
        assert hashlib.sha256(content).hexdigest() == expected, f'Invalid bundled file: {{name}}'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
print('Verified', len(manifest['files']), 'source/data files.')'''),
    markdown('''## 2a. Restore a saved run (optional)

For a new run, leave `RESTORE_CHECKPOINT` empty. To continue a previous session, upload its checkpoint ZIP through **Files > Upload to session storage**, then enter its `/content/...zip` path below. If you saved the printed archive SHA-256, enter it too.

Restoration checks paths, JSON, archive integrity and manifest hashes when present. It refuses to overwrite a different existing result. Older ProofPilot archives are supported; vector caches are rebuilt as needed. Completed inference chunks are reused only when code, models, data and package versions match.'''),
    code('''import importlib.util
checkpoint_spec = importlib.util.spec_from_file_location('proofpilot_checkpoints', PROJECT / 'colab/checkpoints.py')
checkpoint_storage = importlib.util.module_from_spec(checkpoint_spec)
checkpoint_spec.loader.exec_module(checkpoint_storage)
RESTORE_CHECKPOINT = ''
RESTORE_SHA256 = ''
if RESTORE_CHECKPOINT:
    restored = checkpoint_storage.restore_checkpoint(RESTORE_CHECKPOINT, OUTPUTS, RESTORE_SHA256 or None)
    print('Checkpoint restore:', restored)
else:
    print('No checkpoint archive selected; existing output files are preserved.')'''),
    markdown('''## 3. Install notebook dependencies

This keeps Colab's CUDA-enabled PyTorch and installs the evaluation libraries in the remote runtime. Public model weights download later, inside Colab. No inference API key is needed. If Colab requests a session restart after installation, restart, then rerun the preceding cells.'''),
    code('''import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT / 'colab/requirements-colab.txt')])
os.environ['HF_HOME'] = '/content/proofpilot-hf-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.pop('DASHSCOPE_API_KEY', None)
BACKEND = PROJECT / 'backend'
os.chdir(BACKEND)
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
print('Dependencies ready. No paid inference key is configured.')'''),
    markdown('''## 4. Check the implementation

These regression tests use mocked model responses. They validate code behavior, not model accuracy.'''),
    code('''subprocess.check_call([sys.executable, '-m', 'pytest', '-q'], cwd=BACKEND)'''),
    markdown('''## 5. Record the models and revisions

The GPU profile uses lightweight MiniLM retrieval/reranking and a 4-bit Qwen3 4B verifier. This differs from the original hosted Qwen models. Long abstracts are processed in overlapping windows rather than silently truncated. Exact model revisions and software versions go into each experiment's record.'''),
    code('''import benchmark
from app.evaluation.colab_models import ColabModels
import colab_runner
from colab_runner import run_resumable
atomic_json = getattr(colab_runner.atomic_json, '_proofpilot_original', colab_runner.atomic_json)
models = ColabModels()
def save_with_snapshot(path, value):
    atomic_json(path, value)
    if path.name.startswith('report-') and not USE_DRIVE:
        backup = checkpoint_storage.create_checkpoint(OUTPUTS, '/content/proofpilot-checkpoints.zip')
        print('Session checkpoint backup:', backup, flush=True)
save_with_snapshot._proofpilot_original = atomic_json
colab_runner.atomic_json = save_with_snapshot
def saved_retrieval(dataset_path, report_path):
    # Preserve the exact report bytes referenced by existing NLI results.
    if report_path.exists():
        report = json.loads(report_path.read_text())
        dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        expected_embedding = models.metadata['embedding'] + ':window-mean-v1:' + models.revisions[models.metadata['embedding']]
        if report['dataset_sha256'] != dataset_hash or report['partial_run'] or report['verdicts_evaluated']:
            raise ValueError('Saved retrieval report does not match this complete retrieval experiment.')
        if report['models']['embedding'] != expected_embedding or report['models']['rerank'] != models.metadata['rerank']:
            raise ValueError('Saved retrieval models differ. Use a new output folder for a new experiment.')
        for relative_path in ['backend/benchmark.py', 'backend/app/services/retrieval.py', 'backend/app/services/qwen_client.py']:
            if report['code_sha256'].get(relative_path) != hashlib.sha256((PROJECT / relative_path).read_bytes()).hexdigest():
                raise ValueError('Saved retrieval code differs. Use a new output folder.')
        if report.get('free_profile') != models.metadata:
            compatible = False
            protocol_path = OUTPUTS / 'frozen-test-protocol.json'
            if protocol_path.exists():
                frozen = json.loads(protocol_path.read_text()).get('configuration', {})
                compatible = (frozen.get('test_dataset_sha256') == dataset_hash and frozen.get('models') == models.metadata
                              and frozen.get('code_sha256') == colab_runner.code_fingerprint())
            for config_path in OUTPUTS.glob('*/*/configuration.json'):
                config = json.loads(config_path.read_text())
                compatible |= (config.get('dataset_sha256') == dataset_hash and config.get('models') == models.metadata
                               and config.get('code_sha256') == colab_runner.code_fingerprint())
            if not compatible:
                raise ValueError('Legacy retrieval report needs its matching configuration checkpoint to verify model revisions.')
        print('Reusing saved retrieval report:', report_path)
        return report
    with models.activate():
        report = benchmark.run(dataset_path, 'compare', False, 1, None, OUTPUTS / 'embedding-cache')
    report['free_profile'] = models.metadata
    atomic_json(report_path, report)
    return report
print(json.dumps(models.metadata, indent=2))'''),
    markdown('''## 6. Reproduce the offline BM25 baseline

This evaluates all 300 fixed test cases without model inference. Retrieval recall excludes the 112 claims with no labeled evidence. It does not measure verdict accuracy.'''),
    code('''import benchmark
baseline = benchmark.run(PROJECT / 'evaluation/datasets/scifact-v2/test.json', 'lexical', False, 1, None, OUTPUTS / 'unused-cache')
atomic_json(OUTPUTS / 'bm25-test-baseline.json', baseline)
print(json.dumps(baseline['strategies']['lexical']['metrics'], indent=2))'''),
    markdown('''## 7. Run the ten-case development pilot

The first run downloads public model files into the remote runtime. Keep this notebook open while it runs. Results save after every five cases; rerunning the cell reuses completed checkpoints. Invalid model JSON, missing verdicts, and other failures remain visible and count against accuracy.'''),
    code('''development = PROJECT / 'evaluation/datasets/scifact-v2/development.json'
pilot = run_resumable(development, models, OUTPUTS, limit=10)
for strategy, result in pilot['strategies'].items():
    print(strategy, json.dumps(result['metrics'], indent=2))
print('Pilot is partial:', pilot['partial_run'])
print('Saved report:', pilot['saved_report'])'''),
    markdown('''## 8. Optional: full development evaluation

After inspecting the pilot, set `RUN_FULL_DEVELOPMENT = True`. Both retrieval strategies are compared on all 100 cases; the larger verdict evaluation uses the reranked evidence once per claim to limit GPU time. The ten-case pilot compares verdicts for both strategies. Development results may inform changes; keep a record of each version.'''),
    code('''RUN_FULL_DEVELOPMENT = False
if RUN_FULL_DEVELOPMENT:
    retrieval_development = saved_retrieval(development, OUTPUTS / 'retrieval-development-100.json')
    development_report = run_resumable(development, models, OUTPUTS, limit=None, mode='rerank')
    for strategy, result in development_report['strategies'].items():
        print(strategy, json.dumps(result['metrics'], indent=2))
    print(development_report['saved_report'])
else:
    print('Full development run not started.')'''),
    markdown('''## 9. Optional: the fixed 300-case test

Freeze prompts, model configuration, and code before changing the flag below. Do not improve the implementation based on this test and then describe the same set as unseen. A complete run may require more than one free Colab session; checkpoints allow continuation. Public SciFact data may be in model pretraining.'''),
    code('''RUN_FROZEN_TEST = False
if RUN_FROZEN_TEST:
    if 'development_report' not in globals() or development_report['partial_run']:
        raise RuntimeError('Complete and review the full development run before starting the fixed test.')
    if colab_runner.code_fingerprint() != development_report['configuration']['code_sha256'] or models.metadata != development_report['configuration']['models']:
        raise RuntimeError('Inference code or models changed since development. Record a new development experiment first.')
    test = PROJECT / 'evaluation/datasets/scifact-v2/test.json'
    retrieval_test = saved_retrieval(test, OUTPUTS / 'retrieval-test-300.json')
    test_report = run_resumable(test, models, OUTPUTS, limit=None, mode='rerank')
    for strategy, result in test_report['strategies'].items():
        print(strategy, json.dumps(result['metrics'], indent=2))
    print(test_report['saved_report'])
else:
    print('Fixed test run not started.')'''),
    markdown('''## 10. Optional: all five verdict categories

SciFact has only three expert-annotated categories. These ten separate synthetic policy examples exercise weak support and conflicting evidence too. Do not merge their scores with the expert-labeled benchmark.'''),
    code('''RUN_POLICY_EXAMPLES = False
if RUN_POLICY_EXAMPLES:
    policy = run_resumable(PROJECT / 'evaluation/datasets/policy-v3.json', models, OUTPUTS, limit=None, mode='rerank')
    print(json.dumps({k: v['metrics'] for k, v in policy['strategies'].items()}, indent=2))
else:
    print('Policy examples not started.')'''),
    markdown('''## 11. Optional: an in-notebook application demo

This uses the production parser, claim extraction, evidence validation, privacy scanner, and human-review gate. It does not start a public server or tunnel.'''),
    code('''RUN_APPLICATION_DEMO = False
if RUN_APPLICATION_DEMO:
    from IPython.display import JSON, display
    with models.activate():
        # Import here so model names in the report identify this profile.
        from app.services.audit_engine import run_audit
        from app.services.document_parser import parse_document
        source = parse_document('pilot.txt', b'The pilot enrolled 120 editors. Editing time fell by 16 percent. Staffing costs were unchanged. Advertising revenue was not assessed.')
        draft = 'The pilot enrolled 120 editors. Editing time fell by 16 percent, and advertising revenue increased.'
        demo = run_audit(draft, source)
    display(JSON(demo.model_dump()))
else:
    print('Application demo not started.')'''),
    markdown('''## Where your results are saved

Each model experiment has its own fingerprinted folder, per-chunk checkpoints, configuration, and combined reports. With Drive storage, open **Google Drive > ProofPilot Colab > outputs**. With session storage, use the export cell below while the runtime is connected, then save the ZIP in your Drive folder. Temporary session files disappear when Colab recycles the runtime.

To resume, reconnect a free GPU, restore the checkpoint ZIP into `OUTPUTS` if necessary, and rerun setup and experiment cells. Model weights may need downloading again. Preserve the same code, data, model revisions and software versions to reuse checkpoints.

No results have been pre-filled. Only scores actually generated by your run should enter the README or resume. Free Colab GPU access is limited; no keep-alive scripts, paid upgrades, or paid model fallbacks are used.'''),
    code('''from google.colab import files
backup = checkpoint_storage.create_checkpoint(OUTPUTS, '/content/proofpilot-checkpoints.zip')
print('Checkpoint archive:', backup)
print('SHA-256:', hashlib.sha256(Path(backup).read_bytes()).hexdigest())
files.download(backup)
# If the browser blocks the download, open Files, find proofpilot-checkpoints.zip,
# and use its Download menu. Save the archive before ending a temporary session.''')]
    for i, cell in enumerate(cells):
        cell["id"] = f"proofpilot-{i:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"name": "ProofPilot_Free_Colab.ipynb", "provenance": []},
                             "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}, "cells": cells}
    (DEST / "ProofPilot_Free_Colab.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    (DEST / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"notebook": str(DEST / "ProofPilot_Free_Colab.ipynb"), "bundle": str(archive),
                      "bundle_bytes": archive.stat().st_size, "bundle_sha256": digest, "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    build()
