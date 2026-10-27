import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/build_resume.py'
spec = importlib.util.spec_from_file_location('notebook_recovery_for_test', path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def backup_outputs(completed):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        bundle.writestr('report.json', json.dumps({'mock_completed': completed}))
    content = buffer.getvalue()
    label = 'ProofPilot test backup ' + str(completed)
    return [{'data': {'text/html': '<textarea aria-label="' + label + '">' + base64.b64encode(content).decode() + '</textarea>'}},
            {'text': json.dumps({'completed': completed, 'bytes': len(content),
                                 'sha256': hashlib.sha256(content).hexdigest()}) + '\n'}]


def test_highest_portable_backup_is_distinct_from_stream_progress():
    notebook = {'cells': [
        {'source': ['raise RuntimeError("Notebook code must never execute")'], 'outputs': backup_outputs(5)},
        {'outputs': backup_outputs(10) + [{'text': 'Saved checkpoint for cases 11-15\n'}]},
    ]}
    content, status = recovery.extract_backup(notebook)
    assert status['portable_cases'] == 10 and status['logged_cases'] == 15
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        assert json.loads(bundle.read('report.json'))['mock_completed'] == 10


def test_corrupt_highest_backup_is_not_silently_replaced_by_an_older_run():
    outputs = backup_outputs(10)
    outputs[-1]['text'] = json.dumps({'completed': 10, 'bytes': 0, 'sha256': 'wrong'})
    with pytest.raises(ValueError, match='recorded size and checksum'):
        recovery.extract_backup({'cells': [{'outputs': backup_outputs(5)}, {'outputs': outputs}]})


def test_unexecuted_notebook_has_no_recoverable_checkpoint():
    with pytest.raises(ValueError, match='No portable'):
        recovery.extract_backup({'cells': [{'cell_type': 'code', 'source': ['print("test")'], 'outputs': []}]})
