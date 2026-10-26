"""Composite geometry/posegen must not silently use single-volume search rules."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from stacking_common.domain import CompositeTargetGeometry, TargetComponent
from stacking_common.workflows.search import (
    StackingSearchApplication,
    require_search_target,
)

from test_object_geometry import cube_geometry, identity_pose, target_geometry


@pytest.fixture
def composite_target():
    return CompositeTargetGeometry((
        TargetComponent("left", cube_geometry(), identity_pose(-1.0)),
        TargetComponent("right", cube_geometry(), identity_pose(1.0)),
    ))


@pytest.mark.parametrize("method,extra", [
    ("build_mcts", ()),
    ("build_search_environments", ()),
    ("build_parallel_transition_evaluator", (2,)),
])
def test_composite_search_stops_before_native_allocation(composite_target, method, extra):
    # No profiles or native backend are needed: reject before accessing them.
    application = StackingSearchApplication()
    inventory = SimpleNamespace(target=composite_target)
    with pytest.raises(NotImplementedError, match="full MCTS viewer and saved-run"):
        getattr(application, method)(inventory, *extra)


def test_search_guard_preserves_single_targets_and_duck_typed_fixtures():
    require_search_target(target_geometry())
    require_search_target(SimpleNamespace(width=1.0, length=1.0, height=1.0))


def test_composite_direct_pose_solver_factory_is_available(composite_target, monkeypatch):
    import stacking_common.workflows.search as workflow

    application = StackingSearchApplication()
    application.mcts_profile = SimpleNamespace(parallel=SimpleNamespace(posegen_backend="cpu"))
    application.posegen_profile = object()
    inventory = SimpleNamespace(target=composite_target)
    native_module = object()
    solver = object()

    def build(received_inventory, profile, *, core_module):
        assert received_inventory is inventory
        assert profile is application.posegen_profile
        assert core_module is native_module
        return solver

    monkeypatch.setattr(workflow, "CorePlacementPoseSolver", build)
    assert application.build_pose_solver(inventory, core_module=native_module) is solver


def test_excavator_cli_rejects_composite_before_creating_run(
    composite_target, tmp_path, monkeypatch, capsys,
):
    from stacking_excavator import ExcavatorApplication
    from stacking_excavator.planning import cli

    repository = Path(__file__).resolve().parents[1]
    application = ExcavatorApplication.from_repository(repository)
    monkeypatch.setattr(ExcavatorApplication, "from_repository", lambda *a, **k: application)
    monkeypatch.setattr(ExcavatorApplication, "load_inventory",
                        lambda *a, **k: SimpleNamespace(target=composite_target))
    monkeypatch.setattr(cli, "load_native_solver_profile",
                        lambda *a, **k: pytest.fail("native planning must not initialize"))
    destination = tmp_path / "composite.stack"
    assert cli.main([
        "--worker", "--repository", str(repository), "--search-only",
        "--no-pose-from-scan", "--object-ids", str(application.inventory_profile.object_ids[0]),
        "--save-run", str(destination),
    ]) == 2
    assert not destination.exists()
    assert "full MCTS viewer and saved-run support" in capsys.readouterr().err
