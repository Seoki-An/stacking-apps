"""Recorded model identity survives mutable repository assets."""

from dataclasses import replace
import hashlib
import sqlite3

import pytest

from stacking_common.artifacts import PlanningRunContext, StackRunStore
from stacking_common.artifacts.object_assets import (
    copy_object_assets, embed_object_assets, materialize_object_assets, object_asset_paths,
)
from stacking_common.visualization import MCTSDebugArtifact


def example(tmp_path):
    directory = tmp_path / 'models'
    directory.mkdir()
    files = {'object/1': directory / 'model_1.obj',
             'object_surface/1': directory / 'model_1_mesh.ply'}
    files['object/1'].write_bytes(b'# mu: 0.5\nv 0 0 0\n')
    files['object_surface/1'].write_bytes(b'ply\n' + b'surface\n' * 100)
    context = PlanningRunContext('stack', ('joint',),
        {'inventory_profile': {'model_directory': 'models'}},
        asset_hashes={role: hashlib.sha256(path.read_bytes()).hexdigest()
                      for role, path in files.items()})
    return context, files


def test_embed_deduplicate_extract_and_branch(tmp_path):
    context, files = example(tmp_path)
    path = tmp_path / 'original.stack'
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('stack', ())) as writer:
        writer.append_run_context(context)
        assert embed_object_assets(writer, context, object_asset_paths(context, tmp_path)) == 2
        assert embed_object_assets(writer, context, files) == 0
    with StackRunStore.open_append(path) as writer:
        assert embed_object_assets(writer, context, files) == 0
    assert StackRunStore.info(path).status == 'complete'
    files['object/1'].write_bytes(b'# mu: 0.4\nv 0 0 0\n')
    files['object_surface/1'].unlink()
    extracted = materialize_object_assets(path, context, tmp_path, tmp_path / 'cache')
    assert b'mu: 0.5' in extracted['object/1'].read_bytes()
    assert extracted['object_surface/1'].is_file()
    assert extracted['object/1'].parent == extracted['object_surface/1'].parent
    branch = tmp_path / 'branch.stack'
    with StackRunStore.create_mcts(branch, MCTSDebugArtifact('stack', ())) as writer:
        copy_object_assets(path, writer)
        writer.append_run_context(context)
    assert StackRunStore.attachments(path) == StackRunStore.attachments(branch)
    with sqlite3.connect(path) as conn:
        assert conn.execute('select count(*) from attachments').fetchone()[0] == 2
        assert conn.execute("select codec from blobs where sha256=?",
            (context.asset_hashes['object_surface/1'],)).fetchone()[0] == 'zlib'


def test_legacy_fallback_and_invalid_backup_do_not_rewrite_context(tmp_path):
    context, files = example(tmp_path)
    path = tmp_path / 'old.stack'
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('stack', ())) as writer:
        writer.append_run_context(context)
    materialize_object_assets(path, context, tmp_path, tmp_path / 'cache')
    files['object_surface/1'].write_bytes(b'changed')
    with StackRunStore.open_append(path) as writer:
        with pytest.raises(ValueError, match='Recorded object asset differs'):
            embed_object_assets(writer, context, files)
    assert StackRunStore.attachments(path) == ()
    assert StackRunStore.load_run_contexts(path)[0].sha256 == context.sha256
    with pytest.raises(ValueError, match='Recorded object asset differs'):
        materialize_object_assets(path, context, tmp_path, tmp_path / 'cache')


def test_corrupted_embedded_model_does_not_fall_back_to_live_asset(tmp_path):
    context, files = example(tmp_path)
    path = tmp_path / 'bad.stack'
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('stack', ())) as writer:
        writer.append_run_context(context)
        embed_object_assets(writer, context, files)
    with sqlite3.connect(path) as conn:
        conn.execute("update blobs set codec='none', data=?, raw_size=? where sha256=?",
                     (b'corrupt', 7, context.asset_hashes['object/1']))
    with pytest.raises(ValueError):
        materialize_object_assets(path, context, tmp_path, tmp_path / 'cache')


def test_multiple_versions_of_one_object_are_isolated(tmp_path):
    context, files = example(tmp_path)
    path = tmp_path / 'versions.stack'
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('stack', ())) as writer:
        writer.append_run_context(context)
        embed_object_assets(writer, context, files)
        files['object/1'].write_bytes(b'# mu: 0.4\n')
        newer = replace(context, asset_hashes={**context.asset_hashes,
            'object/1': hashlib.sha256(files['object/1'].read_bytes()).hexdigest()})
        writer.append_run_context(newer)
        embed_object_assets(writer, newer, files)
    assert len(StackRunStore.attachments(path)) == 3  # unchanged PLY shared
    a = materialize_object_assets(path, context, tmp_path, tmp_path / 'cache')
    b = materialize_object_assets(path, newer, tmp_path, tmp_path / 'cache')
    assert a['object/1'] != b['object/1']
    assert b'mu: 0.5' in a['object/1'].read_bytes()
    assert b'mu: 0.4' in b['object/1'].read_bytes()


def test_tabletop_view_uses_snapshot(tmp_path):
    from stacking_tabletop.gui.operations.controller import OperationsController
    from stacking_common.visualization import MCTSDebugStep, SceneSnapshot
    context, files = example(tmp_path)
    path = tmp_path / 'tabletop.stack'
    step = MCTSDebugStep(1, SceneSnapshot('stack', ()), (), None, False)
    with StackRunStore.create_mcts(path, MCTSDebugArtifact('stack', ())) as writer:
        writer.append_run_context(context)
        embed_object_assets(writer, context, files)
        writer.append_mcts_step(step)
    for value in files.values():
        value.unlink()
    controller = OperationsController(repository=tmp_path)
    controller.load(path)
    assert b'mu: 0.5' in controller.frame().object_mesh_paths[1].read_bytes()
