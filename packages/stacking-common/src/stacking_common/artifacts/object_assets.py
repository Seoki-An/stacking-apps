"""Content-addressed object snapshots for replay, independent of live assets.

Only object models are embedded here. Execution/resume must still validate their
physical inputs; extracting a recorded model is not permission to execute a run.
"""

import hashlib
from pathlib import Path

from .run_document import StackRunStore


def object_asset_paths(context, repository):
    directory = Path(repository) / context.profiles['inventory_profile']['model_directory']
    paths = {}
    for role in context.asset_hashes:
        kind, _, identifier = role.partition('/')
        if kind in {'object', 'object_surface'}:
            suffix = '.obj' if kind == 'object' else '_mesh.ply'
            paths[role] = directory / f'model_{int(identifier)}{suffix}'
    return paths


def _name(digest):
    return f'object-models/{digest}'


def _verify(data, digest, role):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Recorded object asset differs: {role}')


def embed_object_assets(writer, context, paths):
    """Validate all inputs before adding compressed, deduplicated model blobs.

    Explicit paths also allow verified historical backups to repair old runs.
    Contexts, physics, results, and run status are never rewritten.
    """
    existing = {item.name: item for item in StackRunStore.attachments(writer.path)}
    pending = {}
    for role, digest in context.asset_hashes.items():
        if role.partition('/')[0] not in {'object', 'object_surface'}:
            continue
        name = _name(digest)
        if name in existing:
            _verify(StackRunStore.read_attachment(writer.path, name), digest, role)
            continue
        data = Path(paths[role]).read_bytes()
        _verify(data, digest, role)
        pending[name] = data
    for name, data in pending.items():
        writer.add_attachment(name, data, role='object_model',
                              media_type='application/octet-stream')
    return len(pending)


def copy_object_assets(source, writer):
    """Retain model snapshots when branching a recorded run."""
    for item in StackRunStore.attachments(source):
        if item.role == 'object_model':
            writer.add_attachment(item.name, StackRunStore.read_attachment(source, item.name),
                                  role=item.role, media_type=item.media_type,
                                  metadata=item.metadata)


def materialize_object_assets(source, context, repository, destination):
    """Prefer recorded bytes; strictly verify local assets for older documents.

    OBJ and PLY keep their conventional sibling filenames for rendering mode
    switches. Every context has its own directory, allowing multiple versions
    of the same object ID in a single document. Callers own temporary storage.
    """
    paths = object_asset_paths(context, repository)
    attachments = {item.name for item in StackRunStore.attachments(source)}
    output = Path(destination) / context.sha256
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, path in paths.items():
        digest = context.asset_hashes[role]
        name = _name(digest)
        data = (StackRunStore.read_attachment(source, name) if name in attachments
                else path.read_bytes())
        _verify(data, digest, role)
        target = output / path.name
        target.write_bytes(data)
        result[role] = target
    return result
