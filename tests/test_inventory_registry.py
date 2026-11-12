"""Named asset catalogs remain independent of targets and survive checkpoint resume."""

from pathlib import Path

import pytest
import yaml

from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.inventory import inventory_choices, load_inventory_profile
from stacking_excavator import ExcavatorApplication

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "apps/excavator/config"


def test_registry_selects_catalog_independently_of_target_and_keeps_files_unchanged():
    path = CONFIG / "inventory.yml"
    before = path.read_bytes()
    assert inventory_choices(path) == ("real", ("real", "augmented"))
    real = ExcavatorApplication.from_repository(ROOT).inventory_profile
    augmented = ExcavatorApplication.from_repository(ROOT, inventory_name="augmented",
                                                      target_shape="l_shaped_wall").inventory_profile
    assert real.inventory_name == "real" and len(real.object_ids) == 49
    assert augmented.inventory_name == "augmented" and len(augmented.object_ids) == 100
    assert augmented.model_directory == Path("apps/excavator/assets/objects_augmented")
    assert augmented.target_name == "l_shaped_wall"
    assert 93 in augmented.object_ids and 93 not in real.object_ids
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="unknown inventory"):
        ExcavatorApplication.from_repository(ROOT, inventory_name="missing")


def test_standalone_single_inventory_is_compatible():
    path = CONFIG / "inventory_augmented.yml"
    profile = load_inventory_profile(path)
    assert inventory_choices(path) == (None, ())
    assert profile.inventory_name is None
    assert "inventory_name" not in _values(profile)
    assert len(profile.object_ids) == 100
    with pytest.raises(ValueError, match="no named inventory choices"):
        load_inventory_profile(path, inventory_name="augmented")


@pytest.mark.parametrize("change", ["units", "default", "empty", "name", "mapping", "mixed"])
def test_invalid_inventory_registries_fail_clearly(tmp_path, change):
    values = yaml.safe_load((CONFIG / "inventory.yml").read_text())
    if change == "units":
        values["units"] = "mm"
    elif change == "default":
        values["default"] = "missing"
    elif change == "empty":
        values["inventories"] = {}
    elif change == "name":
        values["inventories"][" "] = values["inventories"]["real"]
    elif change == "mapping":
        values["inventories"]["real"] = []
    else:
        values["objects"] = values["inventories"]["real"]
    path = tmp_path / "inventory.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises((ValueError, TypeError)):
        inventory_choices(path)


@pytest.mark.parametrize("quoted_numbers", [False, True])
def test_legacy_recorded_inventory_keeps_context_hash_when_registry_default_changes(tmp_path, quoted_numbers):
    values = yaml.safe_load((CONFIG / "inventory.yml").read_text())
    shape = yaml.safe_load((CONFIG / "target_shapes.yml").read_text())["shapes"]["default"]
    path = tmp_path / "inventory.yml"
    path.write_text(yaml.safe_dump(dict(units="si", objects=values["inventories"]["real"], target=shape)))
    recorded = _values(load_inventory_profile(path))
    values["default"] = "augmented"
    if quoted_numbers:
        values["inventories"]["real"]["ids"] = [str(i) for i in values["inventories"]["real"]["ids"]]
        values["inventories"]["real"]["density"] = "1000.0"
    path.write_text(yaml.safe_dump(values))
    restored = load_inventory_profile(path, recorded_target=recorded, recorded_inventory=recorded)
    assert _values(restored) == recorded
    assert restored.inventory_name is None
    with pytest.raises(ValueError, match="resume context mismatch"):
        load_inventory_profile(path, inventory_name="augmented",
                               recorded_target=recorded, recorded_inventory=recorded)


def test_named_resume_ignores_new_default_and_rejects_another_selection(tmp_path):
    recorded = _values(load_inventory_profile(CONFIG / "inventory.yml", inventory_name="augmented"))
    values = yaml.safe_load((CONFIG / "inventory.yml").read_text())
    path = tmp_path / "inventory.yml"
    path.write_text(yaml.safe_dump(values))  # Default is real, while recorded choice is augmented.
    restored = load_inventory_profile(path, recorded_target=recorded, recorded_inventory=recorded)
    assert _values(restored) == recorded
    with pytest.raises(ValueError, match="resume preserves recorded inventory"):
        load_inventory_profile(path, inventory_name="real",
                               recorded_target=recorded, recorded_inventory=recorded)


def test_gui_inventory_dropdown_passes_named_choice_and_keeps_target_independent(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from stacking_excavator.gui.operations.planning_run import PlanningRunDialog

    qt = QApplication.instance() or QApplication([])
    dialog = PlanningRunDialog(ROOT)
    try:
        assert [dialog.inventory_choice.itemText(i) for i in range(dialog.inventory_choice.count())] == [
            "real (49 stones)", "augmented (100 stones)"]
        assert dialog.inventory_choice.currentData() == "real"
        dialog.inventory_choice.setCurrentIndex(dialog.inventory_choice.findData("augmented"))
        dialog.target_shape.setCurrentIndex(dialog.target_shape.findData("l_shaped_wall"))
        dialog.output_path.setText(str(tmp_path / "augmented.stack"))
        dialog.search_only.setChecked(True)
        dialog.pose_from_scan.setCurrentIndex(dialog.pose_from_scan.findData(False))
        arguments = dialog.arguments()
        assert arguments[arguments.index("--inventory")+1] == "augmented"
        assert arguments[arguments.index("--target-shape")+1] == "l_shaped_wall"
        assert "--search-only" in arguments and "--no-pose-from-scan" in arguments
        assert not dialog.running and not dialog.destination.exists()
    finally:
        dialog.close()
