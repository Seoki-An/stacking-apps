"""Keyboard review must follow catalog IDs, preserve view angle, and keep models lit."""

import hashlib
from itertools import product
import json

import numpy as np
import pytest

from stacking_common.visualization.asset_fit_viewer import AssetFitViewer


def catalog(directory):
    pv = pytest.importorskip("pyvista")
    models = {}
    for oid in (1, 3, 59):
        obj = directory / f"model_{oid}.obj"
        obj.write_text("# sharpness: 2\no component\n"+
                       "".join(f"v {x} {y} {z}\n" for x, y, z in product((-1., 1.), repeat=3)))
        ply = directory / f"model_{oid}_mesh.ply"
        pv.Sphere(radius=2.).save(ply)
        models[str(oid)] = dict(kind="synthetic" if oid == 59 else "real",
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (obj, ply)})
    (directory / "manifest.json").write_text(json.dumps(dict(status="complete", models=models)))


def test_keyboard_navigation_wraps_noncontiguous_ids_and_preserves_shared_camera(tmp_path):
    catalog(tmp_path)
    viewer = AssetFitViewer(tmp_path, start_id=1, off_screen=True,
                           screenshot_directory=tmp_path / "screenshots")

    def key(name):
        viewer.plot.iren.interactor.SetKeySym(name)
        viewer.plot.iren.key_press_event()

    try:
        viewer.plot.camera.Azimuth(25.)
        initial = np.subtract(viewer.plot.camera_position[0], viewer.plot.camera_position[1])
        initial /= np.linalg.norm(initial)
        for name, expected in (("Right", 3), ("space", 59), ("Right", 1),
                               ("Left", 59), ("Home", 1), ("End", 59)):
            key(name)
            assert viewer.object_id == expected
            actual = np.subtract(viewer.plot.camera_position[0], viewer.plot.camera_position[1])
            actual /= np.linalg.norm(actual)
            np.testing.assert_allclose(actual, initial, atol=1e-12)
            assert all(len(renderer.lights) > 0 for renderer in viewer.plot.renderers)
            assert all(renderer.camera is viewer.plot.renderers[0].camera for renderer in viewer.plot.renderers)
        key("s")
        assert (tmp_path / "screenshots/stone_059_fit.png").is_file()
    finally:
        viewer.close()


def test_viewer_rejects_unknown_id_and_changed_geometry(tmp_path):
    catalog(tmp_path)
    with pytest.raises(ValueError, match="not in this catalog"):
        AssetFitViewer(tmp_path, start_id=2, off_screen=True)
    viewer = AssetFitViewer(tmp_path, start_id=1, off_screen=True)
    try:
        (tmp_path / "model_3.obj").write_text("changed")
        with pytest.raises(ValueError, match="catalog asset changed"):
            viewer.next()
    finally:
        viewer.close()
