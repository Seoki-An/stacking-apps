"""Interrupted search-only runs retain exact checkpoints and their search mode."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
from dataclasses import replace
import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore
from stacking_common.domain import Action
from stacking_common.search import MonteCarloTreeSearch, MCTSNode, MCTSSearchResult
from stacking_excavator.planning.cli import main
from stacking_excavator.planning.resume import load_resume_checkpoint, restore_pick_scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('change', [None, 'position', 'orientation', 'source', 'metadata', 'object_ids'])
def test_restore_pick_scene_only_accepts_quaternion_roundoff(change):
    from stacking_common.planning import PickSceneSnapshot
    pose = np.array([1., 2., 3., .5392192987778156, -.5540334524286524,
                     -.29579898436322427, .5610636704200641])
    original = PickSceneSnapshot('world', {12: pose}, 'recorded scan', {'format': 'legacy'})
    context = SimpleNamespace(profiles={'pick_scene': original.artifact_metadata()})
    drifted = pose.copy()
    drifted[3:] /= np.linalg.norm(drifted[3:])
    assert not np.array_equal(drifted, pose)
    if change == 'position':
        drifted[0] += 1e-8
    if change == 'orientation':
        drifted[3] += 1e-8
    scene = PickSceneSnapshot('world', {13 if change == 'object_ids' else 12: drifted},
        'other scan' if change == 'source' else original.source,
        {'format': 'other'} if change == 'metadata' else original.metadata)
    checkpoint = SimpleNamespace(pick_scene=scene)
    if change is not None:
        with pytest.raises(ValueError, match='checkpoint pickup'):
            restore_pick_scene(checkpoint, context)
    else:
        restored = restore_pick_scene(checkpoint, context)
        assert restored.artifact_metadata() == context.profiles['pick_scene']
        np.testing.assert_array_equal(restored.pose(12), pose)


def controlled_search(seen, *, interrupt_at=None):
    def search(self, root, request):
        seen.append(root.state.copy())
        if len(seen) == interrupt_at:
            raise KeyboardInterrupt('interrupted next search')
        state = root.state.copy()
        index = next(i for i in range(len(state.object_ids)) if i not in state.placed_object_indices)
        object_id = state.object_ids[index]
        pose = np.array([index * .5, 0., .4, 0., 0., 0., 1.])
        action = Action(index, pose, pose, object_id=object_id)
        state.placed_object_indices.append(index)
        state.object_poses[object_id] = pose
        state.action_history.append(action)
        child = MCTSNode(state=state, action=action, reward=1., info={'target_IoU': .1})
        child.parent = root
        root.children.append(child)
        return MCTSSearchResult(root=root, selected=child, ranked_candidates=(child,), completed_iterations=1,
            expansion_calls=1, generated_candidates=1, evaluated_candidates=1,
            failed_evaluations=0, effective_max_depth=1, timed_out=False)
    return search


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(MonteCarloTreeSearch, 'search', controlled_search(seen, interrupt_at=2))
    source = tmp_path / 'source.stack'
    with pytest.raises(KeyboardInterrupt):
        main(['--worker', '--repository', str(ROOT), '--search-only', '--object-ids', '55', '50', '10',
              '--steps', '3', '--iterations', '1', '--save-run', str(source)])
    return source, seen


def test_interrupted_search_resumes_exact_state_in_search_mode(interrupted_run, tmp_path, monkeypatch):
    source, seen = interrupted_run
    checkpoint = load_resume_checkpoint(source)
    assert checkpoint.step == 1
    assert checkpoint.pick_scene.frame_id == 'world'
    assert len(checkpoint.state.action_history) == 1
    assert [s.step for s in StackRunStore.load_mcts(source).steps] == [1]
    np.testing.assert_array_equal(checkpoint.state.object_poses[55], seen[1].object_poses[55])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    resumed = []
    monkeypatch.setattr(MonteCarloTreeSearch, 'search', controlled_search(resumed))
    output = tmp_path / 'resumed.stack'
    # No --search-only: mode is restored from the source context automatically.
    assert main(['--worker', '--repository', str(ROOT), '--resume-run', str(source),
                 '--steps', '1', '--save-run', str(output)]) == 0
    assert resumed[0].placed_object_indices == checkpoint.state.placed_object_indices
    assert len(resumed[0].action_history) == 1
    np.testing.assert_array_equal(resumed[0].object_poses[55], checkpoint.state.object_poses[55])
    restored = load_resume_checkpoint(output)
    assert restored.step == 2 and restored.state.placed_object_indices == [0, 1]
    assert [s.step for s in StackRunStore.load_mcts(output).steps] == [1, 2]
    assert StackRunStore.load_placement_plans(output) == ()
    assert StackRunStore.info(output).metadata['mode'] == 'search'
    assert StackRunStore.load_run_contexts(output)[0].sha256 == checkpoint.context_sha256
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    # A branch can itself be resumed, preserving the accumulated prefix.
    final = tmp_path / 'third.stack'
    assert main(['--worker', '--repository', str(ROOT), '--resume-run', str(output),
                 '--search-only', '--steps', '1', '--save-run', str(final)]) == 0
    assert load_resume_checkpoint(final).step == 3
    assert [s.step for s in StackRunStore.load_mcts(final).steps] == [1, 2, 3]


def test_search_step_and_checkpoint_commit_together(tmp_path, monkeypatch):
    from stacking_common.artifacts.run_document import StackRunWriter
    monkeypatch.setattr(MonteCarloTreeSearch, 'search', controlled_search([]))
    def fail(self, checkpoint):
        raise RuntimeError('checkpoint write failed')
    monkeypatch.setattr(StackRunWriter, '_append_planning_checkpoint', fail)
    path = tmp_path / 'failed.stack'
    assert main(['--worker', '--repository', str(ROOT), '--search-only', '--object-ids', '55', '50',
                 '--save-run', str(path)]) == 2
    assert not StackRunStore.load_mcts(path).steps
    assert not StackRunStore.load_mcts_statistics(path)


def test_old_search_run_has_actionable_error_without_modification(interrupted_run, tmp_path):
    source, _ = interrupted_run
    old = tmp_path / 'old.stack'
    artifact = StackRunStore.load_mcts(source)
    with StackRunStore.create_mcts(old, replace(artifact, steps=()),
                                   metadata={'platform': 'excavator', 'mode': 'search'}) as writer:
        writer.append_run_context(StackRunStore.load_run_contexts(source)[0])
        writer.append_mcts_step(artifact.steps[0])  # Legacy search-only format.
    before = old.read_bytes()
    with pytest.raises(ValueError, match='saved without resume checkpoints'):
        load_resume_checkpoint(old)
    assert old.read_bytes() == before


def test_search_resume_dialog_preserves_mode_and_additional_steps(interrupted_run):
    from PySide6.QtWidgets import QApplication
    from stacking_excavator.gui.operations.planning_run import PlanningRunDialog
    source, _ = interrupted_run
    app = QApplication.instance() or QApplication([])
    dialog = PlanningRunDialog(ROOT, source)
    try:
        assert dialog.search_only.isChecked() and not dialog.search_only.isEnabled()
        checkpoint = load_resume_checkpoint(source)
        context = StackRunStore.load_run_context(source, checkpoint.context_sha256)
        assert not dialog.pose_from_scan.isEnabled()
        assert dialog.pose_from_scan.currentData() is context.profiles["pose_from_scan"]
        dialog.steps.setValue(21)
        dialog.wall_seconds.setValue(3600)
        args = dialog.arguments()
        assert args[args.index('--steps') + 1] == '21'
        assert args[args.index('--max-run-time') + 1] == '3600'
        assert '--resume-run' in args and '--pick-pose-data' not in args
        assert '--pose-from-scan' not in args and '--no-pose-from-scan' not in args
    finally:
        dialog.close()
