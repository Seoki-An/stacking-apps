"""Legacy normal/strict floor modes must agree across search and planning."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stacking_common.actions import ActiveFloorFilter, ScanOrientationFilter, TargetContainmentFilter
from stacking_common.environment import VoxelOverlapEvaluator
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config.action import load_action_generation_profile
import stacking_tabletop.composition as composition


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [None, False, True, "false", 1])
def test_floor_first_profile_requires_boolean(tmp_path, value):
    values = yaml.safe_load((ROOT / "apps/tabletop/config/action_generation.yml").read_text())
    if value is None:
        values["filtering"].pop("strict_floor_first", None)
    else:
        values["filtering"]["strict_floor_first"] = value
    path = tmp_path / "action.yml"
    path.write_text(yaml.safe_dump(values))
    if value is not None and not isinstance(value, bool):
        with pytest.raises(TypeError, match="strict_floor_first must be boolean"):
            load_action_generation_profile(path, max_poses_per_object=32, banned_object_ids=())
    else:
        profile = load_action_generation_profile(path, max_poses_per_object=32, banned_object_ids=())
        assert profile.strict_floor_first is (False if value is None else value)


@pytest.mark.parametrize("strict", [False, True])
def test_floor_mode_is_shared_by_search_and_planning(monkeypatch, strict):
    app = TabletopApplication.from_repository(ROOT)
    app = replace(app, action_generation_profile=replace(
        app.action_generation_profile, strict_floor_first=strict))
    inventory = app.load_inventory([1], {1: np.array([0., 0., 0., 0., 0., 0., 1.])})
    containment = TargetContainmentFilter(VoxelOverlapEvaluator(inventory.target, .003), .5)
    generator = app.build_action_generator(inventory, SimpleNamespace(), filters=(containment,))
    assert any(isinstance(f, ActiveFloorFilter) for f in generator.filters) is strict
    orientation_filter = next(f for f in generator.filters if isinstance(f, ScanOrientationFilter))
    assert orientation_filter.floor_policy is generator.planar_sampler.floor_policy
    assert containment in generator.filters
    assert generator.planar_sampler.floor_policy is not None

    # Exercise real app composition without starting native motion/simulation.
    backend = SimpleNamespace(reset=lambda ids: None)
    monkeypatch.setattr(composition, "CorePlacementBackend", SimpleNamespace(from_profile=lambda *a, **k: backend))
    monkeypatch.setattr(composition, "TabletopNativePlanningBackend", lambda *a, **k: SimpleNamespace(solve=lambda problem: None))
    coordinator = app.build_planning_coordinator(inventory)
    assert (coordinator.safety is not None) is strict
    assert coordinator.placement_backend is backend
    assert coordinator.world_T_stack is app.planning_profile.world_T_stack
