"""Composite .stack replay and checkpoint branching without enabling MCTS."""

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from stacking_common.artifacts import (
    PlanningCheckpoint, PlanningRunContext, PlacementPlanRecord, StackRunStore,
    decode_mcts_debug, encode_mcts_debug,
)
from stacking_common.artifacts.planning_context import (
    _values, branch_planning_run, build_planning_context, validate_planning_resume,
)
from stacking_common.artifacts.target import encode_target, decode_target, target_from_context
from stacking_common.domain import Action, State
from stacking_common.domain.geometry import transform_points
from stacking_common.planning import (
    FrameTransform, PickSceneSnapshot, PlacementPlanningProblem, PlacementPlanningResult, MotionPlanSegment,
)
from stacking_common.profiles.inventory import (
    CompositeTargetProfile, TargetComponentProfile, target_profile_from_record,
)
from stacking_common.visualization import CompositeTargetVolume, MCTSDebugArtifact, target_display
from stacking_common.visualization.operations import OperationsController
from test_composite_target_sampling import box, pose
from test_tabletop_operations_console import _artifact

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return CompositeTargetProfile((
        TargetComponentProfile("west", nodes=box(2., 1., 2.).points, pose=pose(-3., yaw=.3)),
        TargetComponentProfile("east", nodes=box(1., 2., 1.).points, pose=pose(3., yaw=-.6)),
    ))


def display():
    return target_display(profile().geometry(), structure_offset=[.1, -.2, .3])


def context():
    return PlanningRunContext("world", ("joint_a",), {
        "inventory_profile": {"target": _values(profile())},
        "target_geometry": encode_target(target_display(profile().geometry())),
        "runtime": {"search_only": False},
    })


def test_composite_header_meshes_ids_and_poses_roundtrip_json_and_stack(tmp_path):
    target = display()
    artifact = replace(_artifact(), target=target)
    assert artifact.schema_version == 2
    encoded = encode_mcts_debug(artifact)
    restored = decode_mcts_debug(json.loads(json.dumps(encoded)))
    path = StackRunStore.save_mcts(restored, tmp_path / "composite.stack")
    before = path.read_bytes()
    loaded = StackRunStore.load_mcts(path)
    assert path.read_bytes() == before
    assert encode_mcts_debug(loaded) == encoded
    for actual, expected in zip(loaded.target.components, target.components):
        assert actual.component_id == expected.component_id
        np.testing.assert_array_equal(actual.geometry.points, expected.geometry.points)
        np.testing.assert_array_equal(actual.geometry.faces, expected.geometry.faces)
        np.testing.assert_allclose(actual.pose, expected.pose, atol=1e-15)
    encoded["schema_version"] = 1
    with pytest.raises(ValueError, match="require MCTS artifact schema version 2"):
        decode_mcts_debug(encoded)


def test_old_single_target_headers_keep_their_schema_and_payload(tmp_path):
    artifact = _artifact()
    encoded = encode_mcts_debug(artifact)
    assert encoded["schema_version"] == 1 and "kind" not in encoded["target"]
    path = StackRunStore.save_mcts(artifact, tmp_path / "old.stack")
    assert encode_mcts_debug(StackRunStore.load_mcts(path)) == encoded


def test_legacy_recorded_target_without_overlap_or_taper_still_opens():
    recorded = SimpleNamespace(profiles={"inventory_profile": {"target": {
        "origin": [0., 0., 0.], "width": 2., "length": 3., "height": 1.,
    }}})
    target = target_from_context(recorded)
    assert target.width == 2. and target.length == 3.
    np.testing.assert_allclose(target.points[4:] - target.points[:4], [[0., 0., 1.]] * 4)


@pytest.mark.parametrize("change", ["version", "duplicate_id", "kind", "pose"])
def test_malformed_target_records_fail_clearly(change):
    values = encode_target(display())
    if change == "version":
        values["schema_version"] = 50
    elif change == "duplicate_id":
        values["components"][1]["id"] = "west"
    elif change == "kind":
        values["kind"] = "unknown"
    else:
        values["components"][0]["pose"] = [0.] * 7
    with pytest.raises(ValueError):
        decode_target(values)


def test_resolved_profile_roundtrip_and_recorded_mesh_authority():
    recorded = context()
    restored = target_profile_from_record(recorded.profiles["inventory_profile"]["target"])
    np.testing.assert_allclose(restored.geometry().points, profile().geometry().points)
    # Explicit recorded meshes are authoritative, even if a profile is unavailable.
    mesh_only = replace(recorded, profiles={"target_geometry": recorded.profiles["target_geometry"]})
    assert encode_target(target_from_context(mesh_only, display())) == encode_target(display())
    profile_only = replace(recorded, profiles={"inventory_profile": recorded.profiles["inventory_profile"]})
    np.testing.assert_allclose(target_from_context(profile_only).points, profile().geometry().points)


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_both_viewers_receive_disconnected_meshes_without_bridge_triangles(platform):
    pytest.importorskip("pyvistaqt")
    module = importlib.import_module(f"stacking_{platform}.gui.operations.scene_view")
    view = Mock()
    target = display()
    module.OperationsSceneView._add_target(view, target)
    mesh = view.add_mesh.call_args.args[0]
    np.testing.assert_allclose(mesh.points, target.points)
    assert list(mesh.field_data["target_component_ids"]) == ["west", "east"]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # No face can connect a vertex in the western component to the eastern one.
    assert np.all((faces < 8).all(axis=1) | (faces >= 8).all(axis=1))
    assert set(mesh.cell_data["target_component_index"]) == {0, 1}
    assert mesh.n_open_edges == 8  # Two independent open bottom boundaries.
    assert view.add_mesh.call_args.kwargs["name"] == "target"


def test_tabletop_reopen_replay_and_planning_transform_preserve_target(tmp_path):
    from stacking_tabletop.gui.operations import OperationsController as TabletopController
    target = display()
    artifact = replace(_artifact(), target=target)
    path = StackRunStore.save_mcts(artifact, tmp_path / "replay.stack")
    controller = TabletopController()
    controller.load(path)
    controller.set_timeline_frame(1)
    np.testing.assert_allclose(controller.frame().target.points, target.points)
    transform = pose(-6., 3., 2., yaw=.7)
    record = SimpleNamespace(problem=SimpleNamespace(metadata={"planning_T_search": transform}))
    projected = controller._planning_target(record)
    np.testing.assert_allclose(projected.points, transform_points(target.points, transform))
    assert [item.component_id for item in projected.components] == ["west", "east"]


def write_checkpoint_run(path, *, search_only=False):
    original = _artifact()
    artifact = replace(original, frame_id="stack", target=target_display(profile().geometry()),
        steps=tuple(replace(step, scene=replace(step.scene, frame_id="stack")) for step in original.steps))
    step = artifact.steps[0]
    candidate = next(item for item in step.candidates if item.selected)
    recorded = context()
    recorded = replace(recorded, profiles={**recorded.profiles, "runtime": {"search_only": search_only}})
    value = pose(-3., z=.2)
    ids = [11, 22]
    action = Action(candidate.object_index, value, value, object_id=candidate.object_id)
    state = State(ids, [candidate.object_index], {candidate.object_id: value}, [], [action])
    checkpoint = PlanningCheckpoint(1, recorded.sha256, state,
        PickSceneSnapshot("world", {oid: pose() for oid in ids}, "test"),
        FrameTransform("world", artifact.frame_id, pose()))
    problem = PlacementPlanningProblem("world", action, pose(), value, ())
    plan = PlacementPlanRecord(1, 1, problem, PlacementPlanningResult("success", (
        MotionPlanSegment("place", [[0.], [1.]], [0., 1.]),)),
        step.selected_candidate_id, recorded.sha256)
    with StackRunStore.create_mcts(path, replace(artifact, steps=()), metadata={
            "platform": "excavator", "mode": "search" if search_only else "planning"}) as writer:
        writer.append_run_context(recorded)
        if not search_only:
            writer.append_placement_plan(plan)
        writer.append_mcts_step(step, checkpoint=checkpoint)
    return recorded, checkpoint, artifact


@pytest.mark.parametrize("search_only", [False, True])
def test_checkpoint_branch_preserves_composite_and_source_bytes(tmp_path, search_only):
    from stacking_excavator.planning.resume import branch_excavator_run
    source, destination = tmp_path / "source.stack", tmp_path / "branch.stack"
    recorded, checkpoint, artifact = write_checkpoint_run(source, search_only=search_only)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    branch = branch_excavator_run if search_only else branch_planning_run
    writer, resumed = branch(source, destination, recorded)
    writer.finish()
    assert resumed.context_sha256 == checkpoint.context_sha256
    assert resumed.state.placed_object_indices == checkpoint.state.placed_object_indices
    for oid, value in checkpoint.state.object_poses.items():
        np.testing.assert_array_equal(resumed.state.object_poses[oid], value)
    assert encode_target(StackRunStore.load_mcts(destination).target) == encode_target(artifact.target)
    loaded_context = StackRunStore.load_run_context(destination, resumed.context_sha256)
    assert loaded_context.sha256 == recorded.sha256
    np.testing.assert_allclose(target_from_context(loaded_context).points, artifact.target.points)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert len(StackRunStore.load_placement_plans(destination)) == (0 if search_only else 1)
    # A branch is itself resumable; no geometry is read from current YAML.
    assert validate_planning_resume(destination, loaded_context).step == 1


@pytest.mark.parametrize("field", ["id", "pose", "points"])
def test_changed_component_geometry_invalidates_resume(tmp_path, field):
    source = tmp_path / "source.stack"
    recorded, _, _ = write_checkpoint_run(source)
    values = _values(recorded.profiles)
    item = values["target_geometry"]["components"][0]
    if field == "id":
        item["id"] = "different"
    elif field == "pose":
        item["pose"][0] += .01
    else:
        item["points"][0][0] += .01
    changed = replace(recorded, profiles=values)
    with pytest.raises(ValueError, match="resume context mismatch"):
        validate_planning_resume(source, changed)


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_real_context_builder_records_explicit_meshes_only_for_composites(platform):
    from stacking_tabletop import TabletopApplication
    from stacking_excavator import ExcavatorApplication
    application_type = TabletopApplication if platform == "tabletop" else ExcavatorApplication
    app = application_type.from_repository(ROOT)
    oid = app.inventory_profile.object_ids[0]
    scene = PickSceneSnapshot("world", {oid: pose()}, "test")
    inventory = app.load_inventory([oid], {oid: pose()})
    old = build_planning_context(app, inventory, scene, {}, {})
    assert "target_geometry" not in old.profiles
    app = replace(app, inventory_profile=replace(app.inventory_profile, target=profile()))
    inventory = app.load_inventory([oid], {oid: pose()})
    recorded = build_planning_context(app, inventory, scene, {}, {})
    assert isinstance(target_from_context(recorded), CompositeTargetVolume)
    assert recorded.profiles["target_geometry"]["components"][0]["id"] == "west"
    np.testing.assert_allclose(target_from_context(recorded).points, inventory.target.points)


def test_excavator_controller_reopens_recorded_composite_for_motion_replay(tmp_path):
    from test_excavator_operations import replay_run
    from stacking_excavator.gui.operations.controller import OperationsController as ExcavatorController
    source = tmp_path / "old.stack"
    record, old_context = replay_run(source)
    recorded = replace(old_context, profiles={**old_context.profiles,
        "inventory_profile": {**old_context.profiles["inventory_profile"], "target": _values(profile())},
        "target_geometry": encode_target(target_display(profile().geometry())),
    })
    destination = tmp_path / "composite-motion.stack"
    with StackRunStore.create_mcts(destination, MCTSDebugArtifact("stack", (), display()),
                                  metadata={"platform": "excavator"}) as writer:
        writer.append_run_context(recorded)
        writer.append_placement_plan(replace(record, context_sha256=recorded.sha256))
    before = destination.read_bytes()
    controller = ExcavatorController(ROOT)
    controller.load(destination)
    np.testing.assert_allclose(controller.frame().target.points, display().points)
    controller.select_plan_segment(0)
    controller.set_plan_sample(3)
    frame = controller.planning_frame()
    np.testing.assert_allclose(frame.target.points, display().points)
    np.testing.assert_allclose(frame.target_transform[:3], [-6., 3., 0.])
    assert any(item.instance_id == "plan/target" for item in frame.objects)
    assert len(frame.robot_parts) > 0
    assert destination.read_bytes() == before
