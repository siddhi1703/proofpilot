import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

path = Path(__file__).resolve().parents[2] / 'colab/checkpoints.py'
spec = importlib.util.spec_from_file_location('checkpoint_storage_for_test', path)
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)


def test_checkpoint_roundtrip_excludes_caches_and_is_idempotent(tmp_path):
    source = tmp_path / 'outputs'
    source.mkdir()
    (source / 'report.json').write_text('{"completed_cases": 5}\n')
    (source / 'model.bin').write_bytes(b'not a checkpoint')
    archive = checkpoints.create_checkpoint(source, tmp_path / 'backup.zip')
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {'report.json', checkpoints.MANIFEST}
    restored = tmp_path / 'restored'
    assert checkpoints.restore_checkpoint(archive, restored)['restored_files'] == 1
    assert checkpoints.restore_checkpoint(archive, restored)['unchanged_files'] == 1


def test_checkpoint_conflicts_do_not_write_other_files(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    destination.mkdir()
    (source / 'first.json').write_text('{}')
    (source / 'last.json').write_text('{"completed_cases": 5}')
    (destination / 'last.json').write_text('{"completed_cases": 10}')
    archive = checkpoints.create_checkpoint(source, tmp_path / 'backup.zip')
    with pytest.raises(ValueError, match='Existing result differs'):
        checkpoints.restore_checkpoint(archive, destination)
    assert not (destination / 'first.json').exists()
    assert json.loads((destination / 'last.json').read_text())['completed_cases'] == 10


def test_legacy_checkpoint_restores_json_and_skips_vector_cache(tmp_path):
    archive = tmp_path / 'old.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('development/run/configuration.json', '{}')
        bundle.writestr('embedding-cache/model/vector.npy', b'not read as code or pickle')
    result = checkpoints.restore_checkpoint(archive, tmp_path / 'outputs')
    assert result['legacy_archive'] and result['restored_files'] == 1
    assert not (tmp_path / 'outputs/embedding-cache').exists()


def test_unsafe_or_tampered_archives_are_rejected_before_any_write(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('valid.json', '{}')
        bundle.writestr('../outside.json', '{}')
    with pytest.raises(ValueError, match='Unsafe checkpoint path'):
        checkpoints.restore_checkpoint(archive, tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('valid.json', '{}')
        bundle.writestr(checkpoints.MANIFEST, json.dumps({'format_version': 1, 'files': {
            'valid.json': {'bytes': 2, 'sha256': 'wrong'}}}))
    with pytest.raises(ValueError, match='Invalid checkpoint file'):
        checkpoints.restore_checkpoint(archive, tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()
