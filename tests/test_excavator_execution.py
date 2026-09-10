"""Simulation owner failures, measured completion, and restart invariants."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest
from stacking_common.planning import MotionPlanSegment, GraspEvent
from stacking_excavator.acquisition.capture import digest
from stacking_excavator.execution.cli import SimulationClock, demonstration_command
from stacking_excavator.execution.runtime import SimulationRuntime
from stacking_excavator.execution.simulation import SimulatedDevice

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def owner(tmp_path):
    clock = SimulationClock()
    device = SimulatedDevice(ROOT, clock=clock)
    runtime = SimulationRuntime(ROOT, tmp_path / 'session', device=device, clock=clock)
    yield runtime, device, clock
    runtime.close()


def advance(runtime, clock, count=1, heartbeat=True):
    for _ in range(count):
        clock.advance()
        runtime.tick()
        if heartbeat:
            runtime.heartbeat()


def finish(runtime, clock):
    for _ in range(20000):
        advance(runtime, clock)
        if runtime.state['status'] not in ('running', 'held'):
            return
    pytest.fail('simulation did not terminate')


@pytest.mark.parametrize('regrasp', [False, True])
def test_direct_and_regrasp_order_and_measured_completion(owner, regrasp):
    runtime, device, clock = owner
    command = demonstration_command(device, regrasp=regrasp)
    runtime.start(command, expected_revision=0)
    advance(runtime, clock)
    assert device.acknowledgements == [1]
    assert runtime.state['sample_index'] == 0
    assert not any(e.get('diagnostics', {}).get('transition') == 'sample_completed' for e in runtime.history())
    finish(runtime, clock)
    assert runtime.state['status'] == 'succeeded'
    assert device.stopped and not runtime.state['attachment']
    history = runtime.history()
    assert [e['sequence'] for e in history] == list(range(1, len(history)+1))
    completed = [e for e in history if e['diagnostics'].get('transition') == 'sample_completed']
    assert [(e['segment_index'], e['sample_index']) for e in completed] == [
        (s, i) for s, segment in enumerate(command.segments) for i in range(len(segment.positions))]
    assert [e['diagnostics']['grasp'] for e in completed if e['diagnostics']['grasp']] == (
        ['acquire', 'release'] * (2 if regrasp else 1))
    with pytest.raises(ValueError, match='already used'):
        runtime.start(command, expected_revision=0)


def test_hold_resume_cancel_and_explicit_reconciliation(owner):
    runtime, device, clock = owner
    runtime.start(demonstration_command(device), expected_revision=0)
    advance(runtime, clock, 4)
    runtime.hold()
    q = device.positions.copy()
    advance(runtime, clock, 10)
    np.testing.assert_array_equal(device.positions, q)
    assert runtime.state['status'] == 'held'
    runtime.resume()
    advance(runtime, clock, 4)
    runtime.cancel()
    assert device.stopped and runtime.state['needs_replan']
    with pytest.raises(ValueError, match='healthy held'):
        runtime.resume()
    with pytest.raises(ValueError, match='confirmation'):
        runtime.reconcile()
    runtime.reconcile(empty_gripper=True)
    assert not runtime.state['needs_replan']
    runtime.start(demonstration_command(device), expected_revision=0)
    finish(runtime, clock)
    assert runtime.state['status'] == 'succeeded'


@pytest.mark.parametrize('fault', ['lease', 'feedback', 'jaw', 'stall'])
def test_faults_stop_and_require_replanning(owner, fault):
    runtime, device, clock = owner
    runtime.start(demonstration_command(device), expected_revision=0)
    advance(runtime, clock, 4)
    if fault == 'lease':
        q = device.positions.copy()
        clock.advance(2.)
        device.advance()  # Watchdog works even without an owner tick.
        assert device.stopped
        np.testing.assert_array_equal(q, device.positions)
        runtime.heartbeat()  # Late heartbeat cannot revive the attempt.
    elif fault == 'feedback':
        device.feedback_enabled = False
        advance(runtime, clock, 12)
    elif fault == 'jaw':
        device.gripper_known = False
        advance(runtime, clock)
    else:
        device.stalled = True
        finish(runtime, clock)
    assert runtime.state['status'] == 'failed'
    assert runtime.state['needs_replan'] and device.stopped


def proposal(revision=0):
    value = dict(schema_version=1, kind='scene', frame='world', scene_revision=revision,
        valid=True, opening=[], objects=[dict(object_id=10, pose=np.eye(4).tolist(), valid=True, reasons=[])])
    return {**value, 'sha256': digest(value)}


@pytest.mark.parametrize('external', [False, True])
def test_accepted_perception_invalidates_remaining_motion(owner, external):
    runtime, device, clock = owner
    runtime.start(demonstration_command(device), expected_revision=0)
    token = runtime.review.stage(proposal())
    reviewer = runtime.review if external else runtime
    reviewer.decide(token, accept=True, expected_revision=0)
    advance(runtime, clock)
    assert runtime.state['status'] == 'failed'
    assert runtime.state['needs_replan'] and device.stopped
    assert runtime.history()[-1].get('failure') == 'perception_failed' or not external
    assert runtime.review.snapshot()['revision'] == 1


def test_single_owner_and_crash_recovery(owner):
    runtime, device, clock = owner
    with pytest.raises(ValueError, match='already has an owner'):
        SimulationRuntime(ROOT, runtime.directory)
    runtime.start(demonstration_command(device), expected_revision=0)
    advance(runtime, clock, 4)
    # Model abrupt process exit: no graceful terminal event was persisted.
    runtime._lock.close()
    runtime.closed = True
    restarted = SimulationRuntime(ROOT, runtime.directory, device=device, clock=clock)
    try:
        assert restarted.state['status'] == 'failed'
        assert restarted.state['needs_replan'] and device.stopped
        assert restarted.history()[-1]['failure'] == 'interrupted'
        with pytest.raises(ValueError, match='reconcile'):
            restarted.start(demonstration_command(device), expected_revision=0)
    finally:
        restarted.close()


def test_invalid_command_is_rejected_without_device_effect(owner):
    runtime, device, clock = owner
    command = demonstration_command(device)
    for bad in (replace(command, joint_names=command.joint_names[::-1]),
                replace(command, grasp_events=()),
                replace(command, segments=(MotionPlanSegment('bad', np.full((2, 6), 500.)),),
                        grasp_events=())):
        with pytest.raises(ValueError):
            runtime.start(bad, expected_revision=0)
    assert not device.acknowledgements
    device.grant_lease()
    with pytest.raises(ValueError, match='out-of-order'):
        device.send(2, device.positions, 0.)


@pytest.mark.parametrize('case_index', [0, 1])
def test_recorded_legacy_direct_and_regrasp_joint_paths(owner, case_index):
    runtime, device, clock = owner
    fixture = ROOT / 'tests/fixtures/legacy_planner'
    case = json.loads((fixture / 'reference.json').read_text())['cases'][case_index]
    template = demonstration_command(device, regrasp=case_index == 1)
    with np.load(fixture / 'motion.npz', allow_pickle=False) as arrays:
        segments = tuple(MotionPlanSegment(s.name, arrays[f"step_{case['step']}_joints_{i}"])
                         for i, s in enumerate(template.segments))
    events = tuple(replace(e, sample_index=len(segments[e.segment_index].positions)-1)
                   for e in template.grasp_events)
    # Logged joints exercise path ordering. Jaw events are explicit simulation
    # fixtures, not claimed to be measured legacy gripper feedback.
    device.positions = segments[0].positions[0].copy()
    device.stop()
    command = replace(template, segments=segments, grasp_events=events)
    runtime.start(command, expected_revision=0)
    finish(runtime, clock)
    assert runtime.state['status'] == 'succeeded'
    np.testing.assert_allclose(device.positions, segments[-1].positions[-1], atol=.002)


def test_saved_command_requires_commit_and_verified_excavator_assets(tmp_path):
    from test_excavator_operations import replay_run
    from stacking_common.artifacts import StackRunStore, decode_legacy_mcts_debug
    from stacking_excavator.execution.saved_plan import load_command
    path = tmp_path / 'recorded.stack'
    plan, context = replay_run(path, 0)
    with pytest.raises(ValueError, match='no committed'):
        load_command(ROOT, path)
    # Commit a selected search step to the independently stored attempt.
    fixture = ROOT / 'tests/fixtures/legacy_visualization/mcts_candidate_viewer.json'
    step = decode_legacy_mcts_debug(json.loads(fixture.read_text())).steps[0]
    step = replace(step, step=plan.step, selected_candidate_id=plan.candidate_id,
        candidates=tuple(replace(c, candidate_id=plan.candidate_id) if c.selected else c for c in step.candidates))
    with StackRunStore.open_append(path) as writer:
        writer.append_mcts_step(step)
    command = load_command(ROOT, path, step=plan.step)
    assert command.step == plan.step and command.context_sha256 == context.sha256
    assert len(command.segments) == 4
    np.testing.assert_array_equal(command.segments[0].positions, plan.result.segments[0].positions)
    # UI-characterization fixture has no grasp events, so the runtime rejects
    # execution instead of guessing acquire/release from segment names.
    runtime = SimulationRuntime(ROOT, tmp_path / 'session')
    try:
        with pytest.raises(ValueError, match='explicit acquire'):
            runtime.start(command, expected_revision=0)
    finally:
        runtime.close()
