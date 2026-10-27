"""Portable JSON checkpoints; never overwrite a different saved result."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MANIFEST = 'checkpoint-manifest.json'
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 20_000


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or any(part.startswith('.') for part in path.parts)
            or path.as_posix() != name):
        raise ValueError(f'Unsafe checkpoint path: {name}')
    return path


def create_checkpoint(output_root, archive_path):
    """Back up benchmark JSON and hashes, excluding rebuildable model/vector caches."""
    root, archive = Path(output_root).resolve(), Path(archive_path)
    files = {}
    total = 0
    for path in sorted(root.rglob('*.json')):
        if path.name == MANIFEST:
            continue
        name = path.relative_to(root).as_posix()
        _safe_name(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Checkpoint sources cannot be symlinks.')
        content = path.read_bytes()
        json.loads(content)
        total += len(content)
        if total > MAX_BYTES or len(files) >= MAX_FILES - 1:
            raise ValueError('Checkpoint exceeds its size or file-count limit.')
        files[name] = content
    if not files:
        raise ValueError('No checkpoint JSON files found.')
    manifest = {
        'format_version': 1, 'created_at': datetime.now(UTC).isoformat(),
        'files': {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                  for name, content in files.items()},
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
    if total + len(manifest_bytes) > MAX_BYTES:
        raise ValueError('Checkpoint exceeds its size limit including the manifest.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
        bundle.writestr(MANIFEST, manifest_bytes)
    temporary.replace(archive)
    return archive


def restore_checkpoint(archive_path, output_root, expected_sha256=None):
    """Validate every member before writing; accept the earlier JSON/NumPy archives.

    Legacy vector caches and source snapshots are skipped. They are not required
    for resuming completed inference chunks. Differing existing files cause a
    conflict rather than rewinding progress or silently mixing experiments.
    """
    archive, root = Path(archive_path), Path(output_root).resolve()
    if expected_sha256 and hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Checkpoint archive checksum mismatch.')
    contents, names = {}, set()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > MAX_FILES or sum(item.file_size for item in members) > MAX_BYTES:
            raise ValueError('Checkpoint exceeds its size or file-count limit.')
        for item in members:
            name = item.filename.rstrip('/') if item.is_dir() else item.filename
            _safe_name(name)
            if name.casefold() in names or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Duplicate or symlink checkpoint member.')
            names.add(name.casefold())
            if item.is_dir():
                continue
            if name == MANIFEST or name.endswith('.json'):
                content = bundle.read(item)  # zipfile also validates the CRC.
                json.loads(content)
                contents[name] = content
            elif name.startswith('embedding-cache/') and name.endswith('.npy'):
                continue
            elif name.startswith('reproducibility/') and name.endswith('.py'):
                continue
            else:
                raise ValueError(f'Unexpected checkpoint member: {name}')
    manifest_bytes = contents.pop(MANIFEST, None)
    if manifest_bytes:
        manifest = json.loads(manifest_bytes)
        if (not isinstance(manifest, dict) or manifest.get('format_version') != 1
                or not isinstance(manifest.get('files'), dict)
                or set(manifest['files']) != set(contents)):
            raise ValueError('Invalid checkpoint manifest.')
        for name, content in contents.items():
            expected = manifest['files'][name]
            if expected != {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}:
                raise ValueError(f'Invalid checkpoint file: {name}')
    if not contents:
        raise ValueError('No checkpoint JSON files found.')
    pending = []
    for name, content in contents.items():
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError('Checkpoint target escapes its output folder.')
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f'Existing result differs: {name}. Restore into an empty output folder.')
        else:
            pending.append((target, content))
    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.restore-tmp')
        temporary.write_bytes(content)
        temporary.replace(target)
    return {'restored_files': len(pending), 'unchanged_files': len(contents) - len(pending),
            'legacy_archive': manifest_bytes is None}
