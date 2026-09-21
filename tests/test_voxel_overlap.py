"""Optimized overlap must retain the original full-grid reward semantics."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.domain import ConvexGeometry, DirectionalTaper, StackObject, TargetGeometry
from stacking_common.domain.geometry import inverse_transform_points
from stacking_common.environment import VoxelOverlapEvaluator
from test_object_geometry import cube_geometry, identity_pose, stack_object, target_geometry


def full_grid_volume(evaluator, obj, pose):
    local = inverse_transform_points(evaluator._grid_points, pose)
    inside = obj.contains_local_points(local, tolerance=evaluator.resolution * 1e-6)
    return float(np.count_nonzero(inside)) * evaluator.resolution**3


@pytest.mark.parametrize('tapered', [False, True])
def test_overlap_matches_full_grid_for_rotated_multicomponent_objects(tapered):
    target = TargetGeometry(2., 2., 2., np.array([.2, -.3, -.8]),
                           DirectionalTaper.from_degrees(8., 3., 5., 10.)
                           if tapered else DirectionalTaper())
    evaluator = VoxelOverlapEvaluator(target, .1)
    cube = cube_geometry(.45)
    obj = StackObject(7, stack_object(7).body_model, (
        ConvexGeometry(cube.points + [-.35, 0., 0.], cube.faces),
        ConvexGeometry(cube.points + [.35, .1, .1], cube.faces),
    ))
    rng = np.random.default_rng(31)
    for _ in range(30):
        pose = np.r_[rng.uniform(-1.4, 1.4, 3), Rotation.random(random_state=rng).as_quat()]
        expected = full_grid_volume(evaluator, obj, pose)
        assert evaluator.object_containment(obj, pose) == float(np.clip(expected / obj.volume, 0., 1.))
        assert evaluator.target_coverage(obj, pose) == float(np.clip(expected / target.volume, 0., 1.))


@pytest.mark.parametrize('offset', [0., -0.5e-7, 0.5e-7, 2e-7, 4.])
def test_overlap_preserves_voxel_face_boundaries_and_empty_intersections(offset):
    evaluator = VoxelOverlapEvaluator(target_geometry(), .1)
    obj = stack_object(1)
    pose = identity_pose(.05 + offset)
    assert evaluator._intersection_volume(obj, pose) == full_grid_volume(evaluator, obj, pose)


def test_broad_phase_encloses_tolerance_at_acute_corners():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [1., 1e-8, 0.], [1., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    obj = StackObject(1, stack_object(1).body_model, [ConvexGeometry(points, faces)])
    target = TargetGeometry(8., .2, 2., np.zeros(3), DirectionalTaper())
    evaluator = VoxelOverlapEvaluator(target, .1)
    pose = identity_pose()
    pose[1] = .05
    local = inverse_transform_points(evaluator._grid_points, pose)
    inside = obj.contains_local_points(local, tolerance=evaluator.resolution * 1e-6)
    # Merely padding the vertex bounds by one voxel would drop valid points.
    assert np.any(inside & (evaluator._grid_points[:, 0] > 2.1))
    assert evaluator._intersection_volume(obj, pose) == full_grid_volume(evaluator, obj, pose)


def test_cache_reuses_intersection_across_metrics_and_tracks_pose_and_geometry(monkeypatch):
    evaluator = VoxelOverlapEvaluator(target_geometry(), .1)
    obj = stack_object(1)
    other = stack_object(1, .5)  # Same catalog ID, different immutable geometry.
    original = evaluator._compute_intersection_volume
    calls = []

    def compute(obj, pose):
        calls.append(pose.copy())
        return original(obj, pose)

    monkeypatch.setattr(evaluator, '_compute_intersection_volume', compute)
    pose = identity_pose()
    evaluator.object_containment(obj, pose)
    evaluator.target_coverage(obj, pose.copy())
    assert len(calls) == 1
    pose[0] = 1.
    assert evaluator._intersection_volume(obj, pose) == full_grid_volume(evaluator, obj, pose)
    assert len(calls) == 2
    assert evaluator._intersection_volume(other, pose) == full_grid_volume(evaluator, other, pose)
    assert len(calls) == 3
    # A malformed pose must not alias a cached valid pose with identical bytes.
    with pytest.raises(ValueError, match='shape'):
        evaluator.object_containment(obj, pose.reshape(1, 7))


def test_cache_evicts_old_entries_and_is_safe_for_shared_workers(monkeypatch):
    evaluator = VoxelOverlapEvaluator(target_geometry(), .1)
    monkeypatch.setattr(evaluator, '_CACHE_SIZE', 2)
    obj = stack_object(1)
    poses = [identity_pose(x) for x in (0., .3, .8, 2.)]
    expected = [full_grid_volume(evaluator, obj, pose) for pose in poses]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda i: evaluator._intersection_volume(obj, poses[i]),
                                list(range(4)) * 10))
    assert results == expected * 10
    assert len(evaluator._intersections) <= 2
    for pose, volume in zip(poses, expected):
        assert evaluator._intersection_volume(obj, pose) == volume


def test_broad_phase_removes_distant_voxels_without_changing_grid():
    evaluator = VoxelOverlapEvaluator(target_geometry(), .1)
    obj = stack_object(1, .2)
    original = evaluator._grid_points.copy()
    cropped = evaluator._candidate_grid_points(obj, identity_pose())
    assert len(cropped) < len(original) // 10
    assert np.array_equal(evaluator._grid_points, original)
    assert evaluator._intersection_volume(obj, identity_pose()) == full_grid_volume(evaluator, obj, identity_pose())
