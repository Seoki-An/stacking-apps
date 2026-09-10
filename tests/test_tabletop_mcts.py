from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from stacking_common.actions import (
    HeightMapPlanarPoseSampler,
    MeshHeightInitializer,
    PlacementActionGenerator,
)
from stacking_common.domain import State
from stacking_common.search import (
    ActionGenerationRequest,
    MaskedActionGeneratorAdapter,
    EvaluationFidelity,
    MonteCarloTreeSearch,
)
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config import MCTSParallelConfig
from stacking_tabletop.search import LegacyActionBuilderSampler


ROOT = Path(__file__).resolve().parents[1]


def _state() -> State:
    pose = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    return State(
        object_ids=[10, 20],
        placed_object_indices=[0],
        object_poses={10: pose},
        trajectories=[],
        action_history=[],
    )


def _legacy_action(index, object_id, *, diagnostics=None):
    pose = np.array([float(max(index, 0)), 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    return SimpleNamespace(
        stone_idx=index,
        stone_id=object_id,
        pose=pose,
        init_pose=pose.copy(),
        solved_pose=pose.copy(),
        c_feq=0.25,
        c_gap=0.001,
        place_robustness_displacement=0.002,
        diagnostics=dict(diagnostics or {}),
    )


class _LegacyInventory:
    def __init__(self) -> None:
        self.states = []

    def update_from_state(self, state) -> None:
        self.states.append(state)


class _LegacyBuilder:
    def __init__(self, expected_inventory) -> None:
        self.expected_inventory = expected_inventory
        self.calls = []

    def get_action_samples(
        self,
        inventory,
        state,
        *,
        scene_height_map,
        n_action_samples,
        sampling_priority,
    ):
        assert inventory is self.expected_inventory
        assert inventory.states[-1] is state
        self.calls.append(
            {
                "state": state,
                "scene_height_map": scene_height_map,
                "n_action_samples": n_action_samples,
                "sampling_priority": sampling_priority,
            }
        )
        return (
            [
                _legacy_action(1, 20, diagnostics={"source": "posegen"}),
                _legacy_action(
                    0,
                    10,
                    diagnostics={"action_mask_reason": "outside_target"},
                ),
                _legacy_action(-1, -1),
            ],
            np.array([0.0, -np.inf, -np.inf]),
        )


def test_legacy_builder_is_hidden_behind_masked_action_adapter() -> None:
    inventory = _LegacyInventory()
    builder = _LegacyBuilder(inventory)
    sampler = LegacyActionBuilderSampler(builder, inventory)
    generator = MaskedActionGeneratorAdapter(sampler)
    height_map = np.arange(9, dtype=float).reshape(3, 3)

    result = generator.generate(
        _state(),
        SimpleNamespace(scene_height_map=height_map),
        ActionGenerationRequest(
            max_candidates=3,
            sampling_priority="lower_floor_first",
        ),
    )

    call = builder.calls[0]
    legacy_state = call["state"]
    assert legacy_state.stone_set == [10, 20]
    assert legacy_state.stone_seq == [0]
    np.testing.assert_allclose(
        legacy_state.stone_poses[10],
        _state().object_poses[10],
    )
    assert call["scene_height_map"] is height_map
    assert call["n_action_samples"] == 3
    assert call["sampling_priority"] == "lower_floor_first"

    assert result.attempts == 3
    assert len(result.candidates) == 1
    assert result.rejection_counts == {
        "outside_target": 1,
        "dummy_padding": 1,
    }
    action = result.candidates[0].action
    assert action.object_index == 1
    assert action.object_id == 20
    assert action.c_feq == 0.25
    assert action.c_gap == 0.001
    assert action.place_robustness_displacement == 0.002
    assert action.diagnostics == {"source": "posegen"}


def test_legacy_action_identity_must_match_common_inventory() -> None:
    inventory = _LegacyInventory()

    class WrongBuilder(_LegacyBuilder):
        def get_action_samples(self, inventory, state, **kwargs):
            del inventory, state, kwargs
            return [_legacy_action(1, 999)], np.array([0.0])

    generator = MaskedActionGeneratorAdapter(
        LegacyActionBuilderSampler(WrongBuilder(inventory), inventory)
    )

    with pytest.raises(ValueError, match="does not match"):
        generator.generate(
            _state(),
            SimpleNamespace(scene_height_map=None),
            ActionGenerationRequest(max_candidates=1),
        )


def test_tabletop_composition_builds_legacy_action_generator() -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = _LegacyInventory()
    builder = _LegacyBuilder(inventory)

    generator = application.build_legacy_action_generator(builder, inventory)

    assert isinstance(generator, MaskedActionGeneratorAdapter)
    assert isinstance(generator.sampler, LegacyActionBuilderSampler)


def test_tabletop_composition_builds_new_common_action_generator() -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = SimpleNamespace()
    pose_solver = SimpleNamespace()

    generator = application.build_action_generator(inventory, pose_solver)

    assert isinstance(generator, PlacementActionGenerator)
    assert generator.inventory is inventory
    assert generator.pose_solver is pose_solver
    assert isinstance(generator.planar_sampler, HeightMapPlanarPoseSampler)
    assert isinstance(generator.initializer, MeshHeightInitializer)


def test_tabletop_composition_loads_and_builds_native_mcts_path() -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = SimpleNamespace()
    pose_solver = SimpleNamespace()
    environments = {
        EvaluationFidelity.SHORT: SimpleNamespace(),
        EvaluationFidelity.LONG: SimpleNamespace(),
    }

    search = application.build_mcts(
        inventory,
        environments,
        pose_solver=pose_solver,
        prior_scorer=SimpleNamespace(),
    )

    assert isinstance(search, MonteCarloTreeSearch)
    assert search.action_generator.pose_solver is pose_solver
    assert search.root_action_generator.generator is search.action_generator
    assert application.default_mcts_request().iterations == 200
    assert application.mcts_profile.tree.generation_batch_size == 32
    assert application.mcts_profile.tree.final_validation_candidates == 8
    assert application.mcts_profile.tree.initial_depth1_steps == 6
    assert application.action_generation_profile.planar.yaw_angles == (0.0, np.pi)
    assert application.mcts_profile.tree.validate_root_before_search
    assert application.mcts_profile.tree.root_lookahead
    assert application.mcts_profile.tree.rejected_root_retry_rounds == 1
    assert application.mcts_profile.tree.final_selection.mode == "mean"
    assert (
        application.default_mcts_request().final_validation_reserve_seconds
        == 30.0
    )
    assert application.mcts_profile.parallel.enabled
    assert application.mcts_profile.parallel.effective_workers == 12
    assert application.mcts_profile.root_cem.height_map_score_weight == 1.0
    assert np.isclose(
        application.mcts_profile.root_cem.mutation_yaw_std,
        np.deg2rad(5.0),
    )
    assert application.short_simulation_profile.placement.structural.max_time == 4.0
    assert (
        application.short_simulation_profile.placement.structural.freeze_radius
        == 0.10
    )
    assert (
        application.short_simulation_profile.placement.support_chain_z_tolerance
        == 0.005
    )
    assert application.simulation_profile.placement.structural.max_time == 15.0
    assert application.simulation_profile.placement.structural.freeze_radius is None


def test_tabletop_parallel_execution_can_be_disabled() -> None:
    assert MCTSParallelConfig(enabled=False, workers=4).effective_workers == 1


def test_tabletop_composition_enables_independent_workers_by_default(
    monkeypatch,
) -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = SimpleNamespace()
    transition_evaluator = SimpleNamespace()
    pose_solvers = []
    expected_workers = application.mcts_profile.parallel.effective_workers

    def build_transition_workers(self, worker_inventory, worker_count):
        assert worker_inventory is inventory
        assert worker_count == expected_workers
        return transition_evaluator

    def build_pose_solver(self, worker_inventory, *, core_module=None):
        assert worker_inventory is inventory
        assert core_module is None
        solver = SimpleNamespace(worker=len(pose_solvers))
        pose_solvers.append(solver)
        return solver

    monkeypatch.setattr(
        TabletopApplication,
        "build_parallel_transition_evaluator",
        build_transition_workers,
    )
    monkeypatch.setattr(
        TabletopApplication,
        "build_pose_solver",
        build_pose_solver,
    )

    search = application.build_mcts(
        inventory,
        prior_scorer=SimpleNamespace(),
        value_estimator=SimpleNamespace(),
    )

    assert search.transition_evaluator is transition_evaluator
    assert search.action_generator.pose_solvers == tuple(pose_solvers)
    assert len(pose_solvers) == expected_workers
    search.close()
