from types import SimpleNamespace
from pathlib import Path

import numpy as np
import yaml
from stacking_common.actions import MeshHeightInitializer
from stacking_common.actions.scored import ScoredPlanarPoseSampler, ScoredPlacementActionGenerator
from stacking_common.profiles.mcts import load_mcts_profile
from stacking_common.search import (
    EvaluationFidelity,
    MonteCarloTreeSearch,
)
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config import MCTSParallelConfig


ROOT = Path(__file__).resolve().parents[1]
MCTS_CONFIG = ROOT / "apps/tabletop/config"


def test_tabletop_cpu_and_cuda_profiles_preserve_platform_settings() -> None:
    cpu_path = MCTS_CONFIG / "mcts_cpu.yml"
    cuda_path = MCTS_CONFIG / "mcts_cuda.yml"
    cpu = yaml.safe_load(cpu_path.read_text())
    cuda = yaml.safe_load(cuda_path.read_text())

    assert cpu["parallel"]["posegen_backend"] == "cpu"
    assert cpu["parallel"]["simulation_backend"] == "cpu"
    assert cuda["parallel"]["posegen_backend"] == "cuda"
    assert cuda["parallel"]["simulation_backend"] == "cuda"
    for section in cpu:
        if section != "parallel":
            assert cuda[section] == cpu[section]
    for key in ("enabled", "workers"):
        assert cuda["parallel"][key] == cpu["parallel"][key]

    assert (
        TabletopApplication.from_repository(ROOT).mcts_profile
        == load_mcts_profile(cpu_path)
    )
    assert (
        TabletopApplication.from_repository(ROOT, mcts_config=cuda_path).mcts_profile
        == load_mcts_profile(cuda_path)
    )


def _canonical_inventory(application):
    return application.load_inventory([1], {1: np.array([0., 0., .1, 0., 0., 0., 1.])})


def test_tabletop_composition_builds_new_common_action_generator() -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = _canonical_inventory(application)
    pose_solver = SimpleNamespace()

    generator = application.build_action_generator(inventory, pose_solver)

    assert isinstance(generator, ScoredPlacementActionGenerator)
    assert generator.inventory is inventory
    assert generator.pose_solver is pose_solver
    assert isinstance(generator.planar_sampler, ScoredPlanarPoseSampler)
    assert isinstance(generator.initializer, MeshHeightInitializer)
    generator.close()


def test_tabletop_composition_loads_and_builds_native_mcts_path() -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = _canonical_inventory(application)
    pose_solver = SimpleNamespace()
    environments = {
        EvaluationFidelity.SHORT: SimpleNamespace(close=lambda: None),
        EvaluationFidelity.LONG: SimpleNamespace(close=lambda: None),
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
    search.close()


def test_tabletop_parallel_execution_can_be_disabled() -> None:
    assert MCTSParallelConfig(enabled=False, workers=4).effective_workers == 1


def test_tabletop_composition_enables_independent_workers_by_default(
    monkeypatch,
) -> None:
    application = TabletopApplication.from_repository(ROOT)
    inventory = _canonical_inventory(application)
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
