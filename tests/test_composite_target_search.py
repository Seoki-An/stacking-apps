"""Bounded native CPU searches through composite sampling, physics and promotion."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest

pytest.importorskip("stacking_core")

from stacking_common.domain import CompositeTargetGeometry, ConvexGeometry, TargetComponent
from stacking_common.artifacts import StackRunStore
from stacking_common.search import MCTSNode, MCTSSession
from stacking_common.profiles.inventory import CompositeTargetProfile, TargetComponentProfile
from stacking_common.visualization import MCTSDebugArtifact, target_display
from stacking_common.visualization.mcts import project_mcts_step
from stacking_tabletop import TabletopApplication

from test_composite_target_posegen import box, inventory_for, pose


def target_case(kind):
    def component(name, width, length, x=0., y=0.):
        shape = box(width, length, .3)
        return TargetComponent(name, ConvexGeometry(shape.points, shape.faces), pose(x, y))

    if kind == "disconnected":
        pieces = (component("west", .3, .3, -.3), component("east", .3, .3, .3))
    elif kind == "overlapping":
        pieces = (component("west", .3, .3, -.08), component("east", .3, .3, .08))
    elif kind == "l_shape":
        pieces = (component("horizontal", .5, .2, y=-.15),
                  component("vertical", .2, .5, x=-.15))
    else:
        raise ValueError(kind)
    return CompositeTargetGeometry(pieces)


def bounded_application(workers=1):
    app = TabletopApplication.from_repository(Path(__file__).resolve().parents[1])
    # Keep this geometry/physics migration fixture independent of the optional
    # proposal-locality restriction. Its box solves may travel between seeds.
    app = replace(app, action_generation_profile=replace(app.action_generation_profile,
        generator=replace(app.action_generation_profile.generator,
                          max_initial_xy_displacement_ratio=None)))
    search = app.mcts_profile
    return replace(app, mcts_profile=replace(search,
        tree=replace(search.tree, max_depth=2, max_children=2, root_max_children=4,
                     generation_batch_size=4, min_feasible_children=1,
                     evaluation_batch_size=4, rejected_root_retry_rounds=0,
                     expansion_retry_batches=1),
        request=replace(search.request, iterations=4, max_time_seconds=60.),
        root_cem=replace(search.root_cem, population=8, elite=4, keep=4, iterations=1),
        parallel=replace(search.parallel, workers=workers, enabled=workers > 1,
                         posegen_backend="cpu", simulation_backend="cpu")))


@pytest.mark.parametrize("kind", ["disconnected", "overlapping", "l_shape"])
@pytest.mark.parametrize("workers", [1, 2])
def test_native_composite_search_and_promotion(kind, workers, tmp_path):
    app = bounded_application(workers)
    inventory, _ = inventory_for(target_case(kind), count=3)
    environment = app.build_environment(inventory, app.build_core_backend(inventory))
    reset = environment.reset()
    root = MCTSNode(state=reset.state, observation=reset.observation)
    engine = app.build_mcts(inventory, rng=random.Random(0))
    session = MCTSSession(engine, root)
    steps = []
    previous_coverage = 0.
    try:
        for step in range(2):
            result = session.search(app.default_mcts_request())
            assert result.backend_failure is None
            assert result.selected is not None, (result.root.rejected_expansions,
                [(child.rejection_reason, child.info) for child in result.root.children])
            assert not result.timed_out
            assert result.transition_workers == workers
            selected = result.selected
            assert selected.selectable and selected.evaluated
            assert len(selected.state.placed_object_indices) == step + 1
            assert np.isfinite(selected.reward)
            assert selected.observation.target_kind == "composite"
            assert selected.info["simulation_settled"]
            assert result.short_simulation_jobs == 0
            assert result.final_validation_attempts == result.descendant_validation_attempts == 0
            assert selected.info["deepest_feasible_depth"] >= selected.depth
            # Layer admission applies to optimized proposals, not settled poses.
            assert "settled_active_floor_checked" not in selected.info
            assert selected.info["place_robustness_evaluated"]
            assert not selected.info["place_robustness_failed"]
            assert selected.info["posegen_target_component_count"] == 2
            assert selected.info["posegen_target_containment"] == "whole_stone_in_one_component"
            assert selected.info["target_volume_method"] == "voxel_union"
            coverage = selected.info["target_IoU"]
            assert previous_coverage < coverage < 1.
            previous_coverage = coverage
            for value in selected.state.object_poses.values():
                assert inventory.target.contains(value[None, :3])[0]
            if kind == "disconnected" and step == 0:
                assert {part for child in result.ranked_candidates
                        for part in child.info["target_component_ids"]} == {"west", "east"}
            steps.append(project_mcts_step(result.root, selected, step=step + 1, frame_id="stack"))
            session.promote(selected)
    finally:
        engine.close()
        environment.close()

    artifact = MCTSDebugArtifact("stack", tuple(steps), target_display(inventory.target))
    path = StackRunStore.save_mcts(artifact, tmp_path / f"{kind}.stack")
    loaded = StackRunStore.load_mcts(path)
    assert loaded.schema_version == 2
    assert len(loaded.steps) == 2 and all(step.succeeded for step in loaded.steps)
    assert [item.component_id for item in loaded.target.components] == list(inventory.target.component_ids)
    for actual, expected in zip(loaded.steps, steps, strict=True):
        assert actual.selected_candidate_id == expected.selected_candidate_id
        candidate = next(item for item in actual.candidates if item.selected)
        assert candidate.timeline.tracks


@pytest.mark.parametrize("kind", ["disconnected", "overlapping", "l_shape"])
def test_tabletop_cli_composite_with_catalog_stones(kind, monkeypatch, tmp_path):
    """Exercise real catalog loading and CLI reporting/writing, not only boxes."""
    app = bounded_application(workers=2)
    target = CompositeTargetProfile(tuple(TargetComponentProfile(
        item.component_id, nodes=item.geometry.points, pose=item.pose)
        for item in target_case(kind).components))
    app = replace(app, inventory_profile=replace(app.inventory_profile, target=target))
    script = app.repository_root / "apps/tabletop/scripts/debug/mcts.py"
    spec = importlib.util.spec_from_file_location("composite_native_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the bounded geometry fixture (locality gate disabled), native assets,
    # solvers, physics thresholds, reporting and persistence.
    monkeypatch.setattr(TabletopApplication, "from_repository", lambda *a, **k: app)
    monkeypatch.setattr(module, "load_mcts_profile", lambda path: app.mcts_profile)
    path = tmp_path / "catalog-composite.stack"
    monkeypatch.setattr(sys, "argv", [str(script), "--config",
        str(app.repository_root / "apps/tabletop/config/mcts_cpu.yml"),
        "--object-ids", "1", "2", "3", "--steps", "2", "--iterations", "4",
        "--max-search-time", "60", "--no-pose-from-scan", "--save-run", str(path)])
    module.main()
    artifact = StackRunStore.load_mcts(path)
    assert artifact.schema_version == 2
    assert len(artifact.steps) == 2
    assert all(step.succeeded for step in artifact.steps)
    assert len(artifact.target.components) == 2


def test_native_composite_displacement_gate_reports_exhaustion():
    """The locality policy can exhaust a pool; do not silently bypass it."""
    app = bounded_application()
    app = replace(app, action_generation_profile=replace(app.action_generation_profile,
        generator=replace(app.action_generation_profile.generator,
                          max_initial_xy_displacement_ratio=.75)))
    inventory, _ = inventory_for(target_case('disconnected'), count=3)
    environment = app.build_environment(inventory, app.build_core_backend(inventory))
    reset = environment.reset()
    engine = app.build_mcts(inventory, rng=random.Random(0))
    try:
        result = engine.search(MCTSNode(state=reset.state, observation=reset.observation),
                               app.default_mcts_request())
        assert result.selected is None
        assert result.evaluated_candidates == 0
        assert engine.root_action_generator.last_statistics.rejection_counts['initial_xy_displacement'] > 0
    finally:
        engine.close()
        environment.close()
