"""Target selection previews retain real taper, transforms, and component gaps."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from stacking_common.artifacts import StackRunStore
from stacking_common.profiles.targets import load_target_registry
from stacking_common.visualization.contracts import target_display
from stacking_common.rendering.target_preview import TargetPreview, preview_meshes

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_all_registered_previews_render_without_loading_object_assets(platform):
    app = QApplication.instance() or QApplication([])
    preview = TargetPreview()
    preview.resize(320, 390)
    registry = load_target_registry(ROOT / f"apps/{platform}/config/target_shapes.yml")
    try:
        for name, profile in registry.shapes.items():
            target = target_display(profile.geometry())
            preview.set_target(target, name)
            assert preview.title.text() == name
            assert not preview.grab().isNull()
            meshes = preview_meshes(target)
            # A composite's world transform is already applied; do not apply it twice.
            if hasattr(target, "components"):
                assert len(meshes) == len(target.components)
                for (points, faces), component in zip(meshes, target.components):
                    np.testing.assert_allclose(points, component.world_geometry.points)
                    np.testing.assert_array_equal(faces, component.world_geometry.faces)
            else:
                np.testing.assert_allclose(meshes[0][0], target.points)
    finally:
        preview.close()


@pytest.mark.parametrize("shape,expected_runs", [("box", 1), ("two_walls", 2)])
def test_top_view_keeps_gap_between_separate_walls(shape, expected_runs):
    app = QApplication.instance() or QApplication([])
    registry = load_target_registry(ROOT / "apps/excavator/config/target_shapes.yml")
    preview = TargetPreview()
    try:
        preview.set_target(target_display(registry.shapes[shape].geometry()), shape)
        preview.resize(320, 390)
        preview.show()
        app.processEvents()
        canvas = preview.canvas
        image = canvas.grab().toImage()
        y = int(canvas.height() * .83 - 13)  # Middle of the top-view panel.
        filled = np.array([image.pixelColor(x, y).name() == "#9bc4e5"
                           for x in range(image.width())])
        runs = np.count_nonzero(np.diff(np.r_[False, filled].astype(int)) == 1)
        assert runs == expected_runs
    finally:
        preview.close()


def test_resume_preview_uses_recorded_target_when_registry_changes(tmp_path, monkeypatch):
    from stacking_tabletop.gui.operations.planning_run import PlanningRunDialog
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(PlanningRunDialog, "start", lambda self: None)
    recorded = {"width": .31, "length": .47, "height": .62, "origin": [0., 0., 0.]}
    context = SimpleNamespace(profiles={"inventory_profile": {
        "target": recorded, "target_name": "previous_geometry"}})
    monkeypatch.setattr(StackRunStore, "load_planning_checkpoint",
                        lambda path: SimpleNamespace(context_sha256="recorded"))
    monkeypatch.setattr(StackRunStore, "load_run_context", lambda path, digest: context)
    dialog = PlanningRunDialog(ROOT, source=tmp_path / "recorded.stack",
                               destination=tmp_path / "continued.stack")
    try:
        assert not dialog.target_shape.isEnabled()
        assert dialog.target_preview.title.text() == "previous_geometry"
        assert dialog.target_preview.target.width == .31
        assert dialog.target_preview.target.length == .47
        assert dialog.target_preview.target.height == .62
    finally:
        dialog.close()
