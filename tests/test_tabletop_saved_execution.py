"""Full saved-plan sequencing and durable recovery against fake devices."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from stacking_common.artifacts import (
    FixedCollisionGeometry, PlacementPlanRecord, PlanningRunContext, StackRunStore,
    build_execution_command, decode_legacy_mcts_debug, decode_placement_plan, encode_placement_plan,
)
from stacking_common.artifacts.execution import encode_execution_command, decode_execution_command
from stacking_common.domain import Action
from stacking_common.planning import GraspEvent, MotionPlanSegment, PlacementPlanningProblem, PlacementPlanningResult
from stacking_tabletop.config import load_planning_profile
from stacking_tabletop.execution import DynamixelAdapter, UR5eAdapter, UR_JOINT_NAMES
from stacking_tabletop.execution.motion import WRIST3_SOFT_LIMIT
from stacking_tabletop.execution.saved_plan import STAGES, MODES, load_execution_run, validate_command
from stacking_tabletop.execution.sequence import TabletopMotionExecutor, execute_saved_run, _pose
from stacking_tabletop.execution.recovery import (
    accept_released_placement,
    pending_released_stack_capture,
    reconcile_saved_run,
    released_placement_recovery,
    retry_released_stack_capture,
)
from stacking_tabletop.planning.persistence import _values
from test_tabletop_hardware_adapters import FakeTransport, Kinematics

ROOT = Path(__file__).resolve().parents[1]


def test_planning_urdf_uses_execution_wrist3_cable_limit():
    planning = load_planning_profile(ROOT / 'apps/tabletop/config/planning.yml')
    root = ET.parse(ROOT / planning.robot.urdf).getroot()
    joint = next(value for value in root.findall('joint')
                 if value.attrib['name'] == 'wrist_3_joint')
    limit = joint.find('limit')
    assert float(limit.attrib['lower']) == pytest.approx(-WRIST3_SOFT_LIMIT)
    assert float(limit.attrib['upper']) == pytest.approx(WRIST3_SOFT_LIMIT)


@pytest.fixture
def saved(tmp_path):
    import stacking_core._native as native
    planning = load_planning_profile(ROOT / 'apps/tabletop/config/planning.yml')
    planning = replace(planning, robot=replace(planning.robot, home_positions=np.zeros(6)))
    fixture = ROOT / 'tests/fixtures/legacy_visualization/mcts_candidate_viewer.json'
    artifact = decode_legacy_mcts_debug(json.loads(fixture.read_text()))
    step = artifact.steps[0]
    step = replace(step, candidates=tuple(replace(value, object_id=1) if value.selected else value for value in step.candidates))
    selected = next(value for value in step.candidates if value.selected)
    assets = {'stacking_core_native': Path(native.__file__)}
    for kind in ('robot', 'gripper'):
        urdf = ROOT / getattr(planning, kind).urdf
        assets[f'{kind}/urdf'] = urdf
        for element in ET.parse(urdf).iter():
            filename = element.get('filename')
            if filename:
                path = ROOT / filename
                assets[f'{kind}/{filename}'] = path if path.is_file() else urdf.parent / filename
    assets[f'object/{selected.object_id}'] = ROOT / f'apps/tabletop/assets/objects/model_{selected.object_id}.obj'
    context = PlanningRunContext('world', UR_JOINT_NAMES,
        {'planning_profile': _values(planning),
         'native_solver': {'allow_regrasp': False, 'motion': {'direct': {'grasp_steps': 2}}},
         'inventory_profile': {'model_directory': 'apps/tabletop/assets/objects'}},
        tuple(FixedCollisionGeometry(box.obstacle_id, 'box', np.r_[box.center, 0, 0, 0, 1],
              planning.obstacle_friction, tuple(box.size)) for box in planning.obstacles),
        {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in assets.items()})
    points = [np.array([x, 0, 0, 0, 0, 0.]) for x in (0, .015, .025, .04, .05, 0)]
    # Match native segmentation: placement is only the final grasp_steps,
    # whereas pickup also includes a travel prefix.
    counts = (4, 4, 4, 2, 4)
    segments = tuple(MotionPlanSegment(stage, np.linspace(points[i], points[i + 1], counts[i]),
                     openings=np.full(counts[i], .45), mode=mode) for i, (stage, mode) in enumerate(zip(STAGES, MODES)))
    events = tuple(GraspEvent(event, index, counts[index] - 1, _pose(Kinematics().pose(points[index + 1])), .45)
                   for event, index in (('acquire', 0), ('release', 3)))
    pick, place = events[0].frame_from_grasp, events[1].frame_from_grasp
    action = Action(selected.object_index, place, pick, object_id=selected.object_id)
    problem = PlacementPlanningProblem('world', action, pick, place, ())
    result = PlacementPlanningResult('success', segments, grasp_events=events)
    plan = PlacementPlanRecord(1, 1, problem, result, step.selected_candidate_id, context.sha256)
    path = tmp_path / 'execution.stack'
    with StackRunStore.create_mcts(path, replace(artifact, steps=())) as writer:
        writer.append_run_context(context)
        writer.append_placement_plan(plan)
        writer.append_mcts_step(step)
        writer.append_placement_plan(replace(plan, step=2))
        writer.append_mcts_step(replace(step, step=2))
    return load_execution_run(path, ROOT)


def executor_for(saved, gate=None):
    transport = FakeTransport()
    kin = Kinematics()
    kin.grasp_pose = kin.pose
    gripper = DynamixelAdapter(transport, runtime=transport.runtime)
    arm = UR5eAdapter(transport, kin, gripper=gripper)
    executor = TabletopMotionExecutor(saved, arm, gripper)
    executor.gate_changed = gate or (lambda value: executor.approve(value.token) if value else None)
    return transport, executor


def test_saved_roundtrip_keeps_grasp_semantics_and_legacy_encoding(saved):
    plan = saved.plans[0]
    encoded = encode_placement_plan(plan)
    assert encode_placement_plan(decode_placement_plan(encoded)) == encoded
    command = build_execution_command(plan, saved.context)
    assert encode_execution_command(decode_execution_command(encode_execution_command(command))) == encode_execution_command(command)
    assert command.grasp_events[0].event == 'acquire'
    assert command.segments[2].mode == 'attached'
    old = replace(plan, result=replace(plan.result, grasp_events=(),
          segments=tuple(replace(segment, mode=None) for segment in plan.result.segments)))
    legacy = encode_placement_plan(old)
    assert 'grasp_events' not in legacy['result']
    assert all('mode' not in item for item in legacy['result']['segments'])
    assert encode_placement_plan(decode_placement_plan(legacy)) == legacy
    with pytest.raises(ValueError, match='regenerate'):
        validate_command(build_execution_command(old, saved.context), saved.planning, saved.grasp_steps)


def test_signed_model_opening_is_not_a_common_hardware_limit(saved):
    plan = saved.plans[0]
    segments = tuple(replace(item, openings=np.full(len(item.positions), -.05)) for item in plan.result.segments)
    result = replace(plan.result, segments=segments, grasp_events=tuple(replace(event, opening=-.05) for event in plan.result.grasp_events))
    command = build_execution_command(replace(plan, result=result), saved.context)
    validate_command(command, saved.planning, saved.grasp_steps)
    with pytest.raises(ValueError, match='outside'):
        replace(result, grasp_events=(replace(result.grasp_events[0], sample_index=99),))


@pytest.mark.parametrize('pickup_count,place_count,grasp_steps,error', [
    (4, 2, 2, None),
    (4, 3, 2, None),
    (2, 2, 2, 'pickup approach'),
    (4, 2, 3, 'placement approach'),
])
def test_approach_validation_distinguishes_travel_from_contact(saved, pickup_count, place_count, grasp_steps, error):
    command = build_execution_command(saved.plans[0], saved.context)
    segments = list(command.segments)
    for index, count in ((0, pickup_count), (3, place_count)):
        segment = segments[index]
        segments[index] = replace(segment,
            positions=np.linspace(segment.positions[0], segment.positions[-1], count),
            openings=np.full(count, .45))
    command = replace(command, segments=tuple(segments), grasp_events=tuple(
        replace(event, sample_index=len(segments[event.segment_index].positions) - 1)
        for event in command.grasp_events))
    if error is None:
        validate_command(command, saved.planning, grasp_steps)
    else:
        with pytest.raises(ValueError, match=error):
            validate_command(command, saved.planning, grasp_steps)


def test_native_length_placement_runs_fully_compliant_and_retreats_rigid(saved, monkeypatch):
    transport, executor = executor_for(saved)
    execute_path = executor.arm.execute_path
    moves = []

    def record_path(path, **kwargs):
        use_wrench = next(call[2] for call in reversed(transport.calls) if call[0] == 'wrench')
        moves.append((executor._segment, use_wrench, kwargs['attached'], np.array(path)))
        return execute_path(path, **kwargs)

    monkeypatch.setattr(executor.arm, 'execute_path', record_path)
    results = execute_saved_run(saved, executor)
    assert [result.status.value for result in results] == ['succeeded', 'succeeded']
    assert [(index, contact, attached) for index, contact, attached, _ in moves] == [
        (0, False, False), (0, True, False), (1, False, True),
        (2, False, True), (3, True, True), (4, False, False),
    ] * 2
    segments = saved.plans[0].result.segments
    np.testing.assert_array_equal(moves[0][3], segments[0].positions[:3])
    np.testing.assert_array_equal(moves[1][3], segments[0].positions[2:])
    np.testing.assert_array_equal(moves[4][3], segments[3].positions)
    assert len(moves[4][3]) == saved.grasp_steps


def test_saved_executor_gates_hold_before_lift_and_durable_success(saved):
    transport, executor = executor_for(saved)
    gates = []
    def gate(value):
        if value is None:
            return
        gates.append(value.name)
        assert StackRunStore.load_execution_commands(saved.path)[-1].command_id == value.command_id
        if value.name == 'review':
            assert not transport.feedback.gripper_state().holding
        elif value.name in ('place', 'release'):
            assert transport.feedback.gripper_state().holding
        executor.approve(value.token)
    executor.gate_changed = gate
    results = execute_saved_run(saved, executor)
    assert [result.status.value for result in results] == ['succeeded', 'succeeded']
    assert gates == ['review', 'close', 'place', 'release'] * 2
    assert StackRunStore.load_execution_checkpoint(saved.path).completed_steps == (1, 2)
    for command in StackRunStore.load_execution_commands(saved.path):
        events = [event for event in StackRunStore.load_execution_events(saved.path) if event.command_id == command.command_id]
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert events[-1].diagnostics['release_pose'] == pytest.approx(saved.plans[0].problem.place_pose)
    count = len(transport.calls)
    assert execute_saved_run(saved, executor) == ()
    assert len(transport.calls) == count


def test_small_start_mismatch_is_automatically_aligned(saved):
    transport, executor = executor_for(saved)
    transport.joints[0] = np.deg2rad(3.0)
    transport.pose = executor.arm.kinematics.pose(transport.joints)
    transport.refresh()
    gates = []

    def gate(value):
        if value is not None:
            gates.append(value.name)
            executor.approve(value.token)

    executor.gate_changed = gate
    results = execute_saved_run(saved, executor)
    assert all(result.status.value == "succeeded" for result in results)
    assert gates[0] == "review"
    assert "start_alignment" not in gates
    np.testing.assert_allclose(transport.joints, saved.planning.robot.home_positions)


def test_large_start_mismatch_requires_approval_before_alignment(saved):
    transport, executor = executor_for(saved)
    transport.joints[0] = np.deg2rad(6.0)
    transport.pose = executor.arm.kinematics.pose(transport.joints)
    transport.refresh()
    gates = []

    def gate(value):
        if value is not None:
            gates.append(value.name)
            executor.approve(value.token)

    executor.gate_changed = gate
    results = execute_saved_run(saved, executor)
    assert all(result.status.value == "succeeded" for result in results)
    assert gates[0] == "start_alignment"
    assert gates.count("start_alignment") == 1
    np.testing.assert_allclose(transport.joints, saved.planning.robot.home_positions)


@pytest.mark.parametrize('mode', ['cancel', 'hold_failure', 'slip'])
def test_fault_never_releases_or_advances_and_requires_reconciliation(saved, mode):
    transport, executor = executor_for(saved)
    def gate(value):
        if value is None:
            return
        if value.name == 'close' and mode == 'hold_failure':
            transport.ack_hold = False
        if value.name == 'place':
            if mode == 'cancel':
                executor.cancel(value.command_id)
            elif mode == 'slip':
                for _ in range(5):
                    transport.feedback.receive_gripper(transport.position + 100)
        executor.approve(value.token)
    executor.gate_changed = gate
    results = execute_saved_run(saved, executor)
    assert len(results) == 1 and results[0].status.value in ('failed', 'cancelled')
    assert [call[2] for call in transport.calls if call[0] == 'flag'] == [2]  # Only initial OPEN.
    events = StackRunStore.load_execution_events(saved.path)
    if mode == 'hold_failure':
        assert not any(event.segment_index == 1 for event in events)
    assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    with pytest.raises(ValueError, match='reconciliation'):
        execute_saved_run(saved, executor)
    with pytest.raises(ValueError, match='checks'):
        reconcile_saved_run(saved.path, 'unchecked', robot_stopped=True, scene_restored=False, gripper_checked=True)
    reconcile_saved_run(saved.path, 'operator restored pickup and verified controller stopped',
        robot_stopped=True, scene_restored=True, gripper_checked=True)
    _, retry = executor_for(saved)
    assert all(result.status.value == 'succeeded' for result in execute_saved_run(saved, retry))
    assert StackRunStore.load_execution_commands(saved.path)[1].execution_attempt == 2


def test_pause_excludes_operator_time_from_motion_deadline(saved):
    transport, executor = executor_for(saved)
    paused_at = None
    did_pause = False
    def events(event):
        nonlocal paused_at, did_pause
        if not did_pause and event.segment_index == 2 and event.diagnostics.get('reference_index') is not None:
            did_pause = True
            executor.hold(event.command_id)
        if event.status.value == 'held' and event.diagnostics.get('gate') is None:
            paused_at = transport.now
    def sleep():
        if paused_at is not None and transport.now - paused_at > 130:
            executor.resume(executor.active_command_id)
    transport.on_sleep = sleep
    results = execute_saved_run(saved, executor, events=events)
    assert did_pause and executor.runtime.paused_seconds > 130
    assert all(result.status.value == 'succeeded' for result in results)


def test_manual_correction_is_token_scoped_and_tapers_to_saved_path(saved):
    transport, executor = executor_for(saved)
    gate_token = None
    offset_seen = False
    def gate(value):
        nonlocal gate_token
        if value is None:
            return
        if value.name == 'close':
            gate_token = value.token
            assert not executor.approve('stale-token')
            assert executor.adjust(value.token, 0, 1)
        else:
            executor.approve(value.token)
    def events(event):
        nonlocal offset_seen
        if event.diagnostics.get('manual_offset_degrees') and event.status.value == 'held':
            offset_seen = True
            assert event.diagnostics['manual_offset_degrees'][0] == pytest.approx(.5)
            executor.approve(gate_token)
    executor.gate_changed = gate
    results = execute_saved_run(saved, executor, events=events)
    assert offset_seen
    assert all(result.status.value == 'succeeded' for result in results)
    np.testing.assert_allclose(transport.joints, np.zeros(6), atol=1e-9)


def test_queued_crash_can_only_retry_after_operator_reconciliation(saved):
    with StackRunStore.open_append(saved.path) as writer:
        writer.append_execution_command(build_execution_command(saved.plans[0], saved.context))
    _, executor = executor_for(saved)
    with pytest.raises(ValueError, match='reconciliation'):
        execute_saved_run(saved, executor)
    reconcile_saved_run(saved.path, 'operator checked no motion and restored pickup',
        robot_stopped=True, scene_restored=True, gripper_checked=True)
    assert StackRunStore.load_execution_results(saved.path)[0].failure.value == 'interrupted'
    assert all(result.status.value == 'succeeded' for result in execute_saved_run(saved, executor))


def test_post_release_failure_can_accept_placement_and_continue_at_next_step(saved):
    _, executor = executor_for(saved)

    def fail_capture(_executor):
        raise RuntimeError('camera failed after durable release')

    executor.stack_scene_handler = fail_capture
    result = execute_saved_run(saved, executor)[0]
    assert result.status.value == 'failed'
    recovery = released_placement_recovery(saved.path)
    assert recovery.step == 1
    assert len(recovery.release_pose) == 7

    accept_released_placement(
        saved.path, 'operator verified the released stone in its saved placement',
        robot_stopped=True, placed_scene_confirmed=True, gripper_empty=True,
    )
    checkpoint = StackRunStore.load_execution_checkpoint(saved.path)
    assert checkpoint.completed_steps == (1,)
    assert checkpoint.pending_steps == (2,)
    assert not checkpoint.requires_reconciliation

    _, next_executor = executor_for(saved)
    results = execute_saved_run(saved, next_executor)
    assert len(results) == 1 and results[0].status.value == 'succeeded'
    assert StackRunStore.load_execution_commands(saved.path)[-1].step == 2


def test_post_release_failure_can_retry_only_stack_capture_then_continue(saved):
    _, executor = executor_for(saved)

    def fail_capture(_executor):
        raise RuntimeError('camera failed after durable release')

    executor.stack_scene_handler = fail_capture
    assert execute_saved_run(saved, executor)[0].status.value == 'failed'
    recovery = retry_released_stack_capture(
        saved.path, 'operator verified the released stone and empty gripper',
        robot_stopped=True, placed_scene_confirmed=True, gripper_empty=True,
    )
    assert pending_released_stack_capture(saved.path, 1) == recovery

    transport, retry = executor_for(saved)
    captured = []

    def capture_only(active):
        captured.append((active._command.step, active._attached,
                         active._release_pose.copy(), transport.joints.copy()))
        # Keep the following ordinary step focused on sequencing in this test.
        active.stack_scene_handler = None

    retry.stack_scene_handler = capture_only
    results = execute_saved_run(saved, retry)
    assert [result.status.value for result in results] == ['succeeded', 'succeeded']
    assert [result.diagnostics.get('stack_scene_recovery') for result in results] == [True, None]
    assert len(captured) == 1
    step, attached, release_pose, joints = captured[0]
    assert step == 1 and attached is False
    np.testing.assert_allclose(release_pose, recovery.release_pose)
    np.testing.assert_allclose(joints, np.zeros(6))
    commands = StackRunStore.load_execution_commands(saved.path)
    recovery_events = [event for event in StackRunStore.load_execution_events(saved.path)
                       if event.command_id == commands[1].command_id]
    assert not any(event.segment_index == 0 for event in recovery_events)
    assert StackRunStore.load_execution_checkpoint(saved.path).completed_steps == (1, 2)


def test_failed_stack_capture_retry_remains_recoverable(saved):
    _, executor = executor_for(saved)
    executor.stack_scene_handler = lambda _executor: (_ for _ in ()).throw(
        RuntimeError('first camera failure')
    )
    assert execute_saved_run(saved, executor)[0].status.value == 'failed'
    retry_released_stack_capture(
        saved.path, 'operator verified the released scene', robot_stopped=True,
        placed_scene_confirmed=True, gripper_empty=True,
    )

    _, retry = executor_for(saved)
    retry.stack_scene_handler = lambda _executor: (_ for _ in ()).throw(
        RuntimeError('second camera failure')
    )
    assert execute_saved_run(saved, retry)[0].status.value == 'failed'
    assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    assert released_placement_recovery(saved.path).step == 1


def test_released_placement_recovery_is_unavailable_before_release(saved):
    _, executor = executor_for(saved)

    def gate(value):
        if value is None:
            return
        if value.name == 'place':
            executor.cancel(value.command_id)
        executor.approve(value.token)

    executor.gate_changed = gate
    result = execute_saved_run(saved, executor)[0]
    assert result.status.value == 'cancelled'
    assert released_placement_recovery(saved.path) is None
    with pytest.raises(ValueError, match='no safely recoverable'):
        accept_released_placement(
            saved.path, 'not actually released', robot_stopped=True,
            placed_scene_confirmed=True, gripper_empty=True,
        )


def test_missing_asset_hash_or_changed_bytes_fail_before_any_device(saved, tmp_path):
    role = next(role for role in saved.context.asset_hashes if role.startswith('robot/') and role != 'robot/urdf')
    hashes = dict(saved.context.asset_hashes)
    del hashes[role]
    without_hash = replace(saved, context=replace(saved.context, asset_hashes=hashes))
    with pytest.raises(ValueError):
        without_hash.verify_assets()
    changed = tmp_path / 'changed.obj'
    changed.write_text('different mesh')
    with pytest.raises(ValueError, match='hash'):
        replace(saved, assets={**saved.assets, role: changed}).verify_assets()


def test_reconnected_cached_hold_is_rejected_before_open(saved):
    transport, executor = executor_for(saved)
    transport.feedback.receive_hold(700)
    result = execute_saved_run(saved, executor)[0]
    assert result.failure.value == 'invalid_plan'
    assert 'already holds' in result.reason
    assert not any(call[0] in ('flag', 'position') for call in transport.calls)


def test_journal_callback_failure_stops_without_guessing_durable_state(saved):
    transport, executor = executor_for(saved)
    def events(event):
        if event.segment_index == 1:
            raise OSError('journal observer disconnected')
    with pytest.raises(OSError, match='observer'):
        execute_saved_run(saved, executor, events=events)
    assert StackRunStore.load_execution_checkpoint(saved.path).requires_reconciliation
    assert len(StackRunStore.load_execution_commands(saved.path)) == 1
    assert transport.feedback.gripper_state().holding
    assert [call[2] for call in transport.calls if call[0] == 'flag'] == [2]
