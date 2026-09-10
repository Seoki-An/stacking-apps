"""Recorded planner behavior at the common boundary, independent of legacy imports."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.artifacts import PlacementPlanRecord, encode_placement_plan, decode_placement_plan
from stacking_common.domain import Action, State
from stacking_common.planning import FrameTransform, PlacementPlanningProblem, MotionPlanSegment
from stacking_common.planning.frames import compose_poses, inverse_pose
from stacking_common.planning.native import NativeMotionPlannerAdapter

FIXTURE = Path(__file__).parent / "fixtures/legacy_planner"


def reference():
    return json.loads((FIXTURE / "reference.json").read_text())


def pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


@pytest.mark.parametrize("case_index", [0, 1])
def test_recorded_direct_and_regrasp_target_tracks_roundtrip(case_index):
    ref = reference()
    assert hashlib.sha256((FIXTURE / "motion.npz").read_bytes()).hexdigest() == ref["motion_sha256"]
    case = ref["cases"][case_index]
    object_id = case["action"]["stone_id"]
    action = Action(0, case["action"]["pose"], case["action"]["pose"], object_id=object_id)
    pick = np.concatenate(ref["pickup_poses"][str(object_id)])
    transform = FrameTransform("world", "stack", np.array([-6., 3., 0., 0., 0., 0., 1.]))
    problem = PlacementPlanningProblem.from_search(
        State([object_id], [], {}, [], []), action, pick_pose=pick, world_T_stack=transform)
    names = (["pick_approach", "pick_retreat", "place_approach", "place_retreat"]
             if case["mode"] == "direct" else
             ["pick_approach", "pick_retreat", "handoff_place", "handoff_retreat",
              "handoff_approach", "handoff_pick", "place_approach", "place_retreat"])
    with np.load(FIXTURE / "motion.npz", allow_pickle=False) as arrays:
        segments = []
        for i, name in enumerate(names):
            joints = arrays[f"step_{case['step']}_joints_{i}"]
            targets = arrays[f"step_{case['step']}_target_{i}"]
            samples = [NS(robot=NS(positions=q, gripper_opening=-0.4), frame_from_target=pose(t))
                       for q, t in zip(joints, targets)]
            segments.append(NS(stage=name, mode="free" if i % 4 in (0, 3) else "attached",
                               trajectory=NS(samples=samples)))
        candidate = NS(segments=segments, score=case["score"][0], grasp_events=[])
        result = NativeMotionPlannerAdapter(lambda _: NS(
            status="success", candidates=[candidate], selected_index=0)).plan(problem)
        assert result.succeeded
        record = PlacementPlanRecord(case["step"], 1, problem, result)
        restored = decode_placement_plan(json.loads(json.dumps(encode_placement_plan(record))))
        assert len(restored.result.segments) == case["section_count"]
        for i, segment in enumerate(restored.result.segments):
            np.testing.assert_array_equal(segment.positions, arrays[f"step_{case['step']}_joints_{i}"])
            expected = arrays[f"step_{case['step']}_target_{i}"]
            np.testing.assert_allclose(np.array(segment.target_poses)[:, :3], expected[:, :3, 3])
            np.testing.assert_allclose(Rotation.from_quat(np.array(segment.target_poses)[:, 3:]).as_matrix(),
                                       expected[:, :3, :3], atol=1e-12)
        assert restored.result.segments[2].scene_phase == ("place" if case_index == 0 else "handoff")
    # The stored run differs from today's module constant; never scale or replace it.
    assert ref["planning"]["place_z_offset"] == 0.1
    assert ref["production_config"]["qfunction"]["load"] is False


def test_optional_target_samples_are_owned_and_validated():
    p = np.array([0., 0., 0., 0., 0., 0., 1.])
    segment = MotionPlanSegment("transfer", np.zeros((2, 6)), target_poses=(None, p))
    p[0] = 20
    assert segment.target_poses[0] is None
    assert segment.target_poses[1][0] == 0
    with pytest.raises(ValueError, match="align"):
        MotionPlanSegment("transfer", np.zeros((2, 6)), target_poses=(p,))


def test_native_excavator_initializer_model_and_lifetime():
    import stacking_core as core
    urdf = core.load_urdf_model(str(FIXTURE / "excavator_kinematics.urdf"))
    model = urdf.kinematics
    state = core.KinematicState(1, model)
    state.positions = np.array([-.5, .4, -.7, -1., .1, .2])
    tool = model.find_link("grip_body")
    end = model.find_link("cs_rotate")
    end_from_tool = compose_poses(inverse_pose(state.link_pose(end)), state.link_pose(tool))
    robot = core.MotionRobot(state, tool, np.array([0., 0., 0., 0., 0., 0., 1.]))
    names = ["upper_body_joint", "boom_joint", "arm_joint", "bucket_joint", "tilt_joint", "rotate_joint"]
    core.configure_excavator_ik(robot, names, "cs_rotate", end_from_tool)
    target = state.link_pose(tool)
    del state, model, urdf
    seed = core.excavator_ik_seed(robot, target)
    assert seed is not None and np.all(np.isfinite(seed))
    seeded = robot.initial_state
    seeded.positions = seed
    config = core.InverseKinematicsConfig()
    config.initialization = core.InverseKinematicsInitialization.PROVIDED
    config.max_iters = 2000
    result = core.solve_inverse_kinematics(core.InverseKinematicsProblem(seeded, tool, target), config)
    assert result.status == core.SolveStatus.SUCCESS
    # Native IK tol bounds squared translation plus 1-cos(rotation), not metres.
    assert result.pos_error ** 2 + result.rot_error < config.tol
    with pytest.raises(ValueError, match="six joint names"):
        core.configure_excavator_ik(robot, names[:5], "cs_rotate", end_from_tool)
    unreachable = target.copy()
    unreachable[0] += 100
    assert core.excavator_ik_seed(robot, unreachable) is None
