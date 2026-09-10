from dataclasses import dataclass

import numpy as np
import pytest

from stacking_common.planning import (
    CoreSceneBuilder,
    SceneMobility,
    SceneObjectInstance,
)


class FakeMobility:
    STATIC = "native-static"
    KINEMATIC = "native-kinematic"
    DYNAMIC = "native-dynamic"


@dataclass
class FakeBodyInstance:
    id: int
    model: object
    pose: np.ndarray
    mobility: object


@dataclass
class FakeSceneSnapshot:
    frame_id: int
    bodies: list[FakeBodyInstance]


class FakeCore:
    Mobility = FakeMobility
    BodyInstance = FakeBodyInstance
    SceneSnapshot = FakeSceneSnapshot


def pose(x: float = 0.0) -> np.ndarray:
    return np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_scene_builder_maps_indices_models_and_roles() -> None:
    builder = CoreSceneBuilder(
        {10: "first-model", 20: "second-model"},
        frame_id=7,
        core_module=FakeCore,
    )
    ground = FakeBodyInstance(0, "ground", pose(), FakeMobility.STATIC)

    scene = builder.build(
        [
            SceneObjectInstance(0, 10, pose(1.0), SceneMobility.STATIC),
            SceneObjectInstance(1, 20, pose(2.0), SceneMobility.DYNAMIC),
        ],
        fixed_bodies=[ground],
    )

    assert scene.frame_id == 7
    assert [body.id for body in scene.bodies] == [0, 1, 2]
    assert [body.model for body in scene.bodies] == [
        "ground",
        "first-model",
        "second-model",
    ]
    assert scene.bodies[1].mobility == FakeMobility.STATIC
    assert scene.bodies[2].mobility == FakeMobility.DYNAMIC
    assert np.array_equal(scene.bodies[2].pose, pose(2.0))


def test_scene_object_owns_and_validates_scalar_last_pose() -> None:
    source = pose(1.0)
    stack_object = SceneObjectInstance(
        0, 10, source, SceneMobility.KINEMATIC
    )
    source[0] = 3.0

    assert stack_object.pose[0] == 1.0
    with pytest.raises(ValueError, match="shape"):
        SceneObjectInstance(0, 10, np.zeros(6), SceneMobility.DYNAMIC)
    with pytest.raises(ValueError, match="quaternion"):
        SceneObjectInstance(0, 10, np.zeros(7), SceneMobility.DYNAMIC)


def test_scene_builder_rejects_ambiguous_identity() -> None:
    builder = CoreSceneBuilder(
        {10: "model", 20: "other"},
        frame_id=1,
        core_module=FakeCore,
    )

    with pytest.raises(ValueError, match="indices"):
        builder.build(
            [
                SceneObjectInstance(0, 10, pose(), SceneMobility.STATIC),
                SceneObjectInstance(0, 20, pose(), SceneMobility.DYNAMIC),
            ]
        )
    with pytest.raises(ValueError, match="missing body model"):
        builder.build(
            [SceneObjectInstance(0, 99, pose(), SceneMobility.DYNAMIC)]
        )
