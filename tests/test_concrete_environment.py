from dataclasses import dataclass, field

import numpy as np
import pytest

from stacking_common.artifacts import decode_observation, encode_observation
from stacking_common.domain import Action, StackObjectTrajectory, State
from stacking_common.environment import (
    ArrayObservationBuilder,
    ObservationConfig,
    PlacementResult,
    RewardConfig,
    RewardWeights,
    SimulationTransition,
    StackingReward,
)
from stacking_common.learning import encode_heightmap_observation


def pose(x: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, z, 0.0, 0.0, 0.0, 1.0])


@dataclass
class StackObject:
    object_id: int
    pose: np.ndarray
    volume: float
    normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )

    def representative_face_normals(self) -> tuple[np.ndarray, np.ndarray]:
        return self.normal[None, :], np.ones(1)


class Target:
    def __init__(self) -> None:
        self.points = np.zeros((3, 3))
        self.faces = np.array([[0, 1, 2]])
        self.width = 2.0
        self.length = 3.0
        self.height = 1.0
        self.origin = np.zeros(3)
        self.volume = 10.0
        self.calls: list[tuple[int, str]] = []

    def object_containment(
        self,
        stack_object: StackObject,
        object_pose: np.ndarray,
    ) -> float:
        del object_pose
        self.calls.append((stack_object.object_id, "object"))
        return 0.8

    def target_coverage(
        self,
        stack_object: StackObject,
        object_pose: np.ndarray,
    ) -> float:
        del object_pose
        self.calls.append((stack_object.object_id, "target"))
        return 0.1


class Inventory:
    def __init__(self) -> None:
        self.objects = [
            StackObject(10, pose(), 1.0),
            StackObject(20, pose(), 2.0),
        ]
        self.points = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        self.faces = np.array([[[0, 1, 1]], [[0, 1, 1]]])
        self.sharpness = np.array([0.2, 0.3])
        self.object_surface_points = np.zeros((2, 1, 3), dtype=np.float32)
        self.object_surface_normals = np.zeros((2, 1, 3), dtype=np.float32)
        self.object_surface_mask = np.ones((2, 1), dtype=bool)
        self.object_physical_features = np.zeros((2, 14), dtype=np.float32)
        self.target = Target()
        self.target_height_map = np.ones((2, 3), dtype=np.float32)
        self.xlim = np.array([-1.0, 1.0])
        self.ylim = np.array([-2.0, 2.0])

    def get_height_map(self, state: State) -> np.ndarray:
        del state
        return np.full((2, 3), 0.5, dtype=np.float32)

    def initial_pose(self, object_id: int) -> np.ndarray:
        return next(
            stack_object.pose
            for stack_object in self.objects
            if stack_object.object_id == object_id
        ).copy()


def test_array_observation_builds_object_layout() -> None:
    inventory = Inventory()
    builder = ArrayObservationBuilder(
        inventory,
        ObservationConfig(
            max_poses_per_object=2,
            height_map_resolution=(2, 3),
            render_height_map=True,
            action_sample_count=3,
        ),
    )
    current = State(
        object_ids=[10, 20],
        placed_object_indices=[0],
        object_poses={10: pose(x=2.0)},
        trajectories=[],
        action_history=[],
    )

    observation = builder.build(current)

    assert observation.scene_mask.tolist() == [True, False]
    assert observation.action_mask.tolist()[:2] == [0.0, 0.0]
    assert np.isneginf(observation.action_mask[2])
    assert observation.stacked_points[0, 0, 0] == 2.0
    assert np.all(np.isinf(observation.stacked_points[1]))
    assert observation.object_poses[0, 0] == 2.0
    assert observation.scene_height_map.shape == (2, 3)
    heightmap_values = encode_heightmap_observation(observation)
    assert heightmap_values["scene_height_map"].dtype == np.float16
    with pytest.raises(ValueError, match="read-only"):
        observation.pending_points[0, 0, 0] = 4.0

    artifact = encode_observation(observation)
    artifact["pending_points"][0, 0, 0] = 4.0
    assert observation.pending_points[0, 0, 0] == 0.0
    artifact["future_metadata"] = "ignored"
    decoded = decode_observation(artifact)
    assert decoded.object_ids.tolist() == [10, 20]

    with pytest.raises(ValueError, match="does not match observation"):
        builder.build(State([20, 10], [], {}, [], []))


def test_observation_config_rejects_invalid_array_sizes() -> None:
    with pytest.raises(ValueError, match="max poses"):
        ObservationConfig(0, (2, 3), False)
    with pytest.raises(ValueError, match="height-map resolution"):
        ObservationConfig(1, (2, 0), False)


def test_stacking_reward_calculates_object_overlap() -> None:
    inventory = Inventory()
    action = Action(
        object_index=1,
        object_id=20,
        pose=pose(),
        init_pose=pose(),
    )
    current = State(
        object_ids=[10, 20],
        placed_object_indices=[0, 1],
        object_poses={},
        trajectories=[],
        action_history=[action],
    )
    reward = StackingReward(
        inventory,
        inventory.target,
        RewardConfig(
            posegen_threshold=1.0,
            containment_threshold=0.5,
            weights=RewardWeights(
                stability=0.0,
                object_containment=2.0,
                target_overlap_increment=20.0,
                place_stability=0.0,
                large_object_lower=0.0,
                inward_orientation=0.0,
            ),
        ),
        support_evaluator=lambda *_: (1, True),
    )

    value, info = reward.compute(current)

    assert inventory.target.calls == [
        (10, "target"),
        (20, "object"),
    ]
    assert np.isclose(info["target_IoU"], 0.26)
    assert np.isclose(info["target_IoU_increment"], 0.16)
    assert np.isclose(info["object_IoU_reward"], 0.3)
    assert np.isclose(value, 3.8)


class Backend:
    def __init__(self) -> None:
        self.reset_ids: list[int] = []
        self.restored: State | None = None
        self.closed = False

    def reset(self, object_ids: list[int]) -> None:
        self.reset_ids = list(object_ids)

    def place(self, state: State, action: Action) -> PlacementResult:
        placed_pose = action.pose.copy()
        placed_pose[2] = 0.25
        trajectory = StackObjectTrajectory(action.object_id)
        trajectory.add_pose(placed_pose.copy())
        return PlacementResult(
            object_poses={action.object_id: placed_pose},
            trajectories={action.object_id: trajectory},
            c_feq=0.125,
            diagnostics={"backend": "test"},
        )

    def restore(self, state: State) -> None:
        self.restored = state

    def close(self) -> None:
        self.closed = True


def test_simulation_transition_owns_state_bookkeeping() -> None:
    backend = Backend()
    transition = SimulationTransition([10, 20], backend)

    initial = transition.reset()
    result = transition.step(Action(0, pose(), pose(z=1.0)))

    assert backend.reset_ids == [10, 20]
    assert initial.placed_object_indices == []
    assert result.placed_object_indices == [0]
    assert result.object_poses[10][2] == 0.25
    assert result.action_history[0].object_id == 10
    assert result.action_history[0].c_feq == 0.125
    assert result.action_history[0].diagnostics == {"backend": "test"}
    assert len(result.trajectories[-1][10]) == 1
    assert result.terminated is False


def test_simulation_transition_restores_and_rejects_duplicate_objects() -> None:
    backend = Backend()
    transition = SimulationTransition([10], backend)
    transition.reset()
    result = transition.step(Action(0, pose(), pose(z=1.0)))

    assert result.terminated is True
    transition.restore(result)
    assert backend.restored is not result

    with pytest.raises(ValueError, match="already been placed"):
        transition.step(Action(0, pose(), pose(z=1.0)))

    transition.close()
    assert backend.closed is True


def test_simulation_transition_validates_identity() -> None:
    backend = Backend()
    with pytest.raises(ValueError, match="unique"):
        SimulationTransition([10, 10], backend)

    transition = SimulationTransition([10], backend)
    with pytest.raises(ValueError, match="does not match"):
        transition.step(Action(0, pose(), pose(), object_id=20))


@pytest.mark.parametrize('consumer', ['fields', 'artifact', 'learning', 'pickle', 'deepcopy', 'replace'])
def test_deferred_observation_preserves_values_and_owns_state(consumer):
    import copy
    import pickle
    from dataclasses import fields
    from stacking_common.environment.observation import Observation

    builder = ArrayObservationBuilder(Inventory(), ObservationConfig(2, (2, 3), True))
    source = State([10, 20], [0], {10: pose(x=2.)}, [], [])
    expected = builder.build(source)
    original = builder.build
    calls = []
    def build(state):
        calls.append(state)
        return original(state)
    builder.build = build
    actual = builder.defer(source)
    assert isinstance(actual, Observation)
    assert not calls
    source.object_poses[10][0] = 99.
    source.placed_object_indices.clear()
    if consumer == 'artifact':
        actual = decode_observation(encode_observation(actual))
    elif consumer == 'learning':
        left, right = encode_heightmap_observation(actual), encode_heightmap_observation(expected)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    elif consumer == 'pickle':
        actual = pickle.loads(pickle.dumps(actual))
    elif consumer == 'deepcopy':
        actual = copy.deepcopy(actual)
    elif consumer == 'replace':
        from dataclasses import replace
        actual = replace(actual)
    for item in fields(Observation):
        np.testing.assert_equal(getattr(actual, item.name), getattr(expected, item.name))
    assert len(calls) == 1
    assert '_builder' not in vars(actual)


def test_deferred_observation_materializes_once_across_readers():
    from concurrent.futures import ThreadPoolExecutor
    builder = ArrayObservationBuilder(Inventory(), ObservationConfig(2, (2, 3), True))
    original = builder.build
    calls = []
    def build(state):
        calls.append(state)
        return original(state)
    builder.build = build
    observation = builder.defer(State([10, 20], [], {}, [], []))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: observation.scene_height_map, range(12)))
    assert len(calls) == 1
    assert all(value is results[0] for value in results)


@pytest.mark.parametrize('name', ['object_ids', 'object_poses', 'object_surface_points',
                                 'object_surface_normals', 'object_surface_mask',
                                 'object_physical_features'])
def test_deferred_optional_field_access_does_not_return_dataclass_default(name):
    builder = ArrayObservationBuilder(Inventory(), ObservationConfig(2, (2, 3), True))
    state = State([10, 20], [], {}, [], [])
    actual = builder.defer(state)
    assert '_builder' in vars(actual)
    np.testing.assert_array_equal(getattr(actual, name), getattr(builder.build(state), name))
    assert '_builder' not in vars(actual)
