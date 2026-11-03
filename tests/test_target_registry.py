"""Named platform targets, compatible inline inputs and frozen resume geometry."""

from pathlib import Path
import shutil
import importlib.util
import sys

import pytest
import yaml

from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.planning_context import _values
from stacking_common.domain import CompositeTargetGeometry
from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.profiles.targets import load_target_registry, target_shape_choices
from stacking_tabletop import TabletopApplication
from stacking_excavator import ExcavatorApplication

ROOT = Path(__file__).resolve().parents[1]


def test_excavator_plan17_wall_preserves_recorded_dimensions_and_taper():
    from stacking_common.profiles.inventory import target_profile_from_record
    import numpy as np

    # Captured from plan17's planning context; no local .stack dependency.
    expected = target_profile_from_record({
        "width": 2., "length": 6., "height": 2.5, "origin": [0., 0., 0.],
        "support_overlap": .02,
        "taper": {name: 0.08726646259971647 for name in (
            "width_positive", "width_negative", "length_positive", "length_negative")},
    })
    registry = load_target_registry(ROOT / "apps/excavator/config/target_shapes.yml")
    actual, name = registry.select("default")
    assert name == "default"
    assert registry.default == "default"
    assert _values(actual) == _values(expected)
    np.testing.assert_array_equal(actual.geometry().points, expected.geometry().points)


def test_excavator_l_shaped_wall_overlaps_and_forms_one_region():
    import numpy as np
    from stacking_common.actions.target_regions import TargetRegions

    registry = load_target_registry(ROOT / "apps/excavator/config/target_shapes.yml")
    profile, name = registry.select("l_shaped_wall")
    geometry = profile.geometry()
    assert name == "l_shaped_wall"
    assert registry.default == "default"
    assert isinstance(geometry, CompositeTargetGeometry)
    assert len(geometry.components) == 2
    regions = TargetRegions(geometry)
    assert len(regions.regions) == 1
    assert set(regions.regions[0].region_id) == {"north_arm", "east_arm"}
    # Interior overlap remains present from ground to the tapered wall tops.
    overlap = np.array([[-1., -1., z] for z in (0., 1.5, 3.)])
    assert all(component.world_geometry.contains(overlap).all()
               for component in geometry.components)
    # Both arms are included, but the concave notch is not convexified.
    np.testing.assert_array_equal(geometry.contains(np.array([
        [-1., 1., 0.], [1., -1., 0.], [1., 1., 0.], [3., 0., 0.],
    ])), [True, True, False, False])


@pytest.mark.parametrize("platform,application,default,composite", [
    ("tabletop", TabletopApplication, "tapered_wall", "two_piers"),
    ("excavator", ExcavatorApplication, "default", "two_walls"),
    ("excavator", ExcavatorApplication, "default", "l_shaped_wall"),
])
def test_platform_registry_defaults_selection_and_recorded_restore(
    platform, application, default, composite, monkeypatch,
):
    path = ROOT / f"apps/{platform}/config/inventory.yml"
    before = path.read_bytes()
    assert "target" not in yaml.safe_load(before)
    configured = load_inventory_profile(path)
    assert configured.target_name == default
    assert target_shape_choices(path)[0] == default
    selected = application.from_repository(ROOT, target_shape=composite).inventory_profile
    assert selected.target_name == composite
    assert isinstance(selected.target.geometry(), CompositeTargetGeometry)
    assert selected.object_ids == configured.object_ids
    with pytest.raises(ValueError, match="unknown target shape"):
        application.from_repository(ROOT, target_shape="missing")
    recorded = _values(selected)
    monkeypatch.setattr("stacking_common.profiles.targets.load_target_registry",
                        lambda *a: pytest.fail("resume must not consult the registry"))
    restored = application.from_repository(ROOT, recorded_target=recorded).inventory_profile
    assert _values(restored) == recorded
    with pytest.raises(ValueError, match="recorded target"):
        application.from_repository(ROOT, recorded_target=recorded, target_shape="box")
    assert path.read_bytes() == before


def test_inline_inventory_remains_compatible_without_target_name_in_context(tmp_path):
    source = ROOT / "apps/tabletop/config"
    values = yaml.safe_load((source / "inventory.yml").read_text())
    shape = yaml.safe_load((source / "target_shapes.yml").read_text())["shapes"]["tapered_wall"]
    values["target"] = shape
    path = tmp_path / "inventory.yml"
    path.write_text(yaml.safe_dump(values))
    profile = load_inventory_profile(path)
    assert profile.target_name is None
    assert "target_name" not in _values(profile)
    assert target_shape_choices(path) == (None, ())
    # A legacy saved profile remains hash-compatible after the file migration.
    recorded = _values(profile)
    del values["target"]
    path.write_text(yaml.safe_dump(values))
    assert _values(load_inventory_profile(path, recorded_target=recorded)) == recorded


@pytest.mark.parametrize("change", ["units", "default", "empty", "name", "shape", "unknown"])
def test_invalid_registry_fails_clearly(change, tmp_path):
    values = yaml.safe_load((ROOT / "apps/tabletop/config/target_shapes.yml").read_text())
    if change == "units":
        values["units"] = "mm"
    elif change == "default":
        values["default"] = "missing"
    elif change == "empty":
        values["shapes"] = {}
    elif change == "name":
        values["shapes"][" "] = values["shapes"]["box"]
    elif change == "shape":
        values["shapes"]["box"] = []
    else:
        values["typo"] = 1
    path = tmp_path / "target_shapes.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError):
        load_target_registry(path)


@pytest.mark.parametrize("platform,chosen", [
    ("tabletop", "two_piers"), ("excavator", "two_walls"),
    ("excavator", "l_shaped_wall"),
])
def test_gui_selection_passes_name_without_editing_registry(platform, chosen, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog as TabletopDialog
    from stacking_excavator.gui.operations.planning_run import PlanningRunDialog as ExcavatorDialog
    qt = QApplication.instance() or QApplication([])
    path = ROOT / f"apps/{platform}/config/target_shapes.yml"
    before = path.read_bytes()
    dialog = (TabletopDialog if platform == "tabletop" else ExcavatorDialog)(ROOT)
    try:
        assert not dialog.running
        dialog.output_path.setText(str(tmp_path / "new.stack"))
        dialog.search_only.setChecked(True)
        dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(False))
        dialog.target_shape.setCurrentIndex(dialog.target_shape.findData(chosen))
        assert dialog.target_preview.title.text() == chosen
        expected = load_inventory_profile(ROOT / f"apps/{platform}/config/inventory.yml",
                                          target_shape=chosen).target.geometry()
        import numpy as np
        np.testing.assert_allclose(dialog.target_preview.target.points, expected.points)
        args = dialog.arguments()
        assert args[args.index("--target-shape") + 1] == chosen
        assert path.read_bytes() == before
    finally:
        dialog.close()


def test_excavator_resume_keeps_named_target_after_registry_removed(tmp_path, monkeypatch):
    from stacking_common.search import MonteCarloTreeSearch
    from stacking_excavator.planning.cli import main
    from test_excavator_search_resume import controlled_search

    repo = tmp_path / "repository"
    config = repo / "apps/excavator/config"
    shutil.copytree(ROOT / "apps/excavator/config", config)
    (repo / "apps/excavator/assets").symlink_to(ROOT / "apps/excavator/assets", target_is_directory=True)
    source, destination = tmp_path / "first.stack", tmp_path / "resumed.stack"
    monkeypatch.setattr(MonteCarloTreeSearch, "search", controlled_search([], interrupt_at=2))
    with pytest.raises(KeyboardInterrupt):
        main(["--worker", "--repository", str(repo), "--search-only", "--no-pose-from-scan",
              "--target-shape", "box", "--object-ids", "55", "50", "--steps", "2",
              "--iterations", "1", "--save-run", str(source)])
    before = source.read_bytes()
    checkpoint = StackRunStore.load_planning_checkpoint(source)
    old = StackRunStore.load_run_context(source, checkpoint.context_sha256)
    assert old.profiles["inventory_profile"]["target_name"] == "box"
    (config / "target_shapes.yml").unlink()
    monkeypatch.setattr(MonteCarloTreeSearch, "search", controlled_search([]))
    assert main(["--worker", "--repository", str(repo), "--resume-run", str(source),
                 "--steps", "1", "--save-run", str(destination)]) == 0
    new = StackRunStore.load_planning_checkpoint(destination)
    assert new.context_sha256 == checkpoint.context_sha256
    assert new.step == 2
    assert source.read_bytes() == before


def test_batch_prepares_composite_with_union_volume(tmp_path):
    from stacking_excavator.planning.shape_batch import prepare_batch
    output = tmp_path / "batch"
    manifest = prepare_batch(ROOT, output, shapes=["two_walls"],
        config=ROOT / "apps/excavator/config/mcts_cpu.yml")
    job, = manifest["jobs"]
    assert job["target_volume_m3"] > 0.
    assert job["target_volume_method"] == "voxel_union"
    inventory = load_inventory_profile(output / "two_walls/repository/apps/excavator/config/inventory.yml")
    assert inventory.target_name == "two_walls"
    assert isinstance(inventory.target.geometry(), CompositeTargetGeometry)


def test_tabletop_cli_resolves_named_target_before_loading_assets(monkeypatch):
    script = ROOT / "apps/tabletop/scripts/debug/mcts.py"
    spec = importlib.util.spec_from_file_location("target_registry_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--config",
        str(ROOT / "apps/tabletop/config/mcts_cpu.yml"), "--target-shape", "two_piers",
        "--object-ids", "1", "--steps", "1"])

    class InventoryReached(Exception):
        pass

    def load_inventory(self, *args):
        assert self.inventory_profile.target_name == "two_piers"
        assert isinstance(self.inventory_profile.target.geometry(), CompositeTargetGeometry)
        raise InventoryReached

    monkeypatch.setattr(TabletopApplication, "load_inventory", load_inventory)
    with pytest.raises(InventoryReached):
        module.main()


def test_tabletop_resume_selector_is_locked_and_sends_no_target_override(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog
    qt = QApplication.instance() or QApplication([])
    # Unit-test the passive form without launching its normal resume process.
    monkeypatch.setattr(PlanningRunDialog, "start", lambda self: None)
    dialog = PlanningRunDialog(ROOT, source=tmp_path / "recorded.stack",
                               destination=tmp_path / "resumed.stack")
    try:
        assert not dialog.target_shape.isEnabled()
        assert "--target-shape" not in dialog.arguments()
    finally:
        dialog.close()
