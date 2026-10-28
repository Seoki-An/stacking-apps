from dataclasses import dataclass

import numpy as np
import pytest

from stacking_common.domain import (
    Action,
    ConvexGeometry,
    DirectionalTaper,
    ObjectInventory,
    StackObject,
    State,
    TargetGeometry,
)
from stacking_common.environment import RewardConfig, RewardWeights, StackingReward
from stacking_common.environment import (
    InventoryObservationAdapter,
    VoxelOverlapEvaluator,
)


@dataclass
class BodyModel:
    mass: float
    inertia: np.ndarray


def cube_geometry(size: float = 1.0) -> ConvexGeometry:
    half = size / 2.0
    points = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return ConvexGeometry(points, faces, sharpness=40.0)


def identity_pose(x: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def stack_object(object_id: int, size: float = 1.0) -> StackObject:
    model = BodyModel(mass=2.0, inertia=np.diag([0.2, 0.3, 0.4]))
    return StackObject(object_id, model, [cube_geometry(size)])


def target_geometry() -> TargetGeometry:
    return TargetGeometry(
        width=2.0,
        length=2.0,
        height=2.0,
        origin=np.array([0.0, 0.0, -1.0]),
        taper=DirectionalTaper(),
        support_overlap=0.0,
    )


def test_convex_geometry_calculates_volume_planes_and_samples() -> None:
    geometry = cube_geometry()

    assert np.isclose(geometry.volume, 1.0)
    assert np.allclose(geometry.extent, [1.0, 1.0, 1.0])
    assert geometry.contains(np.array([[0.0, 0.0, 0.0]])).tolist() == [True]
    assert geometry.contains(np.array([[0.6, 0.0, 0.0]])).tolist() == [False]
    with pytest.raises(ValueError, match="read-only"):
        geometry.points[0, 0] = 3.0
    points, normals, mask = geometry.vertex_surface_samples(10)
    assert points.shape == normals.shape == (10, 3)
    assert np.count_nonzero(mask) == 8
    assert np.allclose(np.linalg.norm(normals[mask], axis=1), 1.0)


@pytest.mark.parametrize("tolerance", [0., 1e-9, 1e-4, -1e-9])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("irregular", [False, True])
def test_contains_preserves_point_major_results_at_plane_boundaries(tolerance, layout, irregular):
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    cube = cube_geometry()
    rotation = Rotation.from_euler("xyz", [23., -37., 61.], degrees=True).as_matrix()
    geometry = ConvexGeometry(cube.points @ rotation.T, cube.faces)
    if irregular:
        vertices = np.random.default_rng(21).normal(size=(60, 3)) * [.2, .5, .1]
        geometry = ConvexGeometry(vertices, ConvexHull(vertices).simplices)
    centers, normals, _ = geometry.face_data()
    boundary = centers + tolerance * normals
    points = np.concatenate([
        np.random.default_rng(12).uniform(-1., 1., (4096, 3)),
        geometry.points, boundary,
        np.nextafter(boundary, np.inf), np.nextafter(boundary, -np.inf),
    ])
    if layout == "strided":
        buffer = np.zeros((len(points), 6))
        buffer[:, ::2] = points
        points = buffer[:, ::2]
    else:
        points = np.array(points, order=layout)
    for query in (points, points[:1], points[:0]):
        expected = np.all(
            query @ geometry._face_normals.T <= geometry._plane_offsets + tolerance,
            axis=1,
        )
        np.testing.assert_array_equal(geometry.contains(query, tolerance), expected)


def test_stack_object_keeps_native_model_and_geometry_separate() -> None:
    stack_object_value = stack_object(10)

    assert stack_object_value.object_id == 10
    assert stack_object_value.mass == 2.0
    assert np.isclose(stack_object_value.volume, 1.0)
    assert stack_object_value.mesh_arrays()[0].shape == (8, 3)
    assert stack_object_value.physical_features().shape == (14,)


def test_inventory_owns_order_identity_and_initial_poses() -> None:
    first = stack_object(10)
    second = stack_object(20, size=0.5)
    source_pose = identity_pose(1.0)
    inventory = ObjectInventory(
        [first, second],
        {10: source_pose, 20: identity_pose(2.0)},
        target_geometry(),
    )
    source_pose[0] = 9.0

    assert inventory.object_ids == (10, 20)
    assert inventory.object_index(20) == 1
    assert inventory.initial_pose(10)[0] == 1.0
    assert set(inventory.models) == {10, 20}
    with pytest.raises(ValueError, match="unique"):
        ObjectInventory(
            [first, first],
            {10: identity_pose()},
            target_geometry(),
        )


def test_inventory_observation_adapter_builds_geometry_arrays() -> None:
    inventory = ObjectInventory(
        [stack_object(10), stack_object(20, size=0.5)],
        {10: identity_pose(), 20: identity_pose(2.0)},
        target_geometry(),
    )
    class HeightMapRenderer:
        xlim = np.array([-1.0, 1.0])
        ylim = np.array([-2.0, 2.0])

        def render(self, state: State) -> np.ndarray:
            del state
            return np.zeros((2, 3), dtype=np.float32)

        def render_target(self) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    adapter = InventoryObservationAdapter(
        inventory,
        surface_sample_count=10,
        height_map_renderer=HeightMapRenderer(),
    )

    assert adapter.points.shape == (2, 8, 3)
    assert adapter.faces.shape == (2, 12, 3)
    assert adapter.object_surface_points.shape == (2, 10, 3)
    assert adapter.object_physical_features.shape == (2, 14)
    assert adapter.initial_pose(20)[0] == 2.0


def test_target_geometry_evaluates_object_and_target_overlap() -> None:
    target = target_geometry()
    stack_object_value = stack_object(10)
    overlap = VoxelOverlapEvaluator(target, resolution=0.1)

    assert np.isclose(target.volume, 8.0)
    assert np.isclose(
        overlap.object_containment(stack_object_value, identity_pose()),
        1.0,
        atol=1e-10,
    )
    assert np.isclose(
        overlap.target_coverage(stack_object_value, identity_pose()),
        0.125,
        atol=1e-10,
    )
    assert overlap.object_containment(
        stack_object_value, identity_pose(x=2.0)
    ) == 0.0
    with pytest.raises(ValueError, match="resolution"):
        VoxelOverlapEvaluator(target, resolution=3.0)


def test_target_taper_is_explicit_and_validated() -> None:
    taper = DirectionalTaper.from_degrees(
        width_positive=5.0,
        width_negative=5.0,
    )
    target = TargetGeometry(
        width=2.0,
        length=2.0,
        height=1.0,
        origin=np.zeros(3),
        taper=taper,
    )

    assert np.ptp(target.points[4:, 0]) < np.ptp(target.points[:4, 0])
    with pytest.raises(ValueError, match="read-only"):
        target.origin[0] = 1.0
    with pytest.raises(ValueError, match="closes"):
        TargetGeometry(
            width=1.0,
            length=1.0,
            height=1.0,
            origin=np.zeros(3),
            taper=DirectionalTaper.from_degrees(
                width_positive=45.0,
                width_negative=45.0,
            ),
        )


def test_common_inventory_and_target_drive_reward_without_legacy_adapters() -> None:
    stack_object_value = stack_object(10)
    inventory = ObjectInventory(
        [stack_object_value],
        {10: identity_pose()},
        target_geometry(),
    )
    state = State(
        object_ids=[10],
        placed_object_indices=[0],
        object_poses={10: identity_pose()},
        trajectories=[],
        action_history=[Action(0, identity_pose(), identity_pose())],
    )
    reward = StackingReward(
        inventory,
        VoxelOverlapEvaluator(inventory.target, resolution=0.1),
        RewardConfig(
            posegen_threshold=1.0,
            containment_threshold=0.5,
            weights=RewardWeights(
                stability=0.0,
                object_containment=1.0,
                target_overlap_increment=0.0,
                place_stability=0.0,
                large_object_lower=0.0,
                inward_orientation=0.0,
            ),
        ),
        support_evaluator=lambda *_: (0, False),
    )

    value, info = reward.compute(state)

    assert np.isclose(value, 0.5)
    assert np.isclose(info["object_IoU"], 1.0)
