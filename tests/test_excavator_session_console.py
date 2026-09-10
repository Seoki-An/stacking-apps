"""Operations workflow integration without a NUC or a hardware publisher."""
import json
from pathlib import Path
import time
import numpy as np
import pytest
from stacking_excavator.acquisition.capture import CaptureFrame, CropRequest, digest
from stacking_excavator.acquisition.storage import create_capture, prepare_crop

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, seconds=20):
    from PySide6.QtTest import QTest
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate(), 'timed out waiting for console worker'


def test_background_perception_review_simulation_and_reopen(tmp_path):
    from PySide6.QtWidgets import QApplication
    from stacking_excavator.gui.operations.session import SessionConsole
    from stacking_excavator.execution.cli import SimulationClock
    app = QApplication.instance() or QApplication([])
    console = SessionConsole(ROOT)
    with np.load(ROOT / 'apps/excavator/assets/perception/model_10.npz') as a:
        points = a['xyz'] + [0, 0, 1]
    calibration = digest({'test': 'console'})
    frame = CaptureFrame('f0', 'world', 1, 1, np.eye(4), np.eye(4), calibration, points, (0.,)*6, 0.)
    crop = CropRequest('scene', 'world', np.eye(4), (5.,)*3, (0.,)*3, calibration, 0, (10,))
    create_capture(tmp_path / 'raw', 'console-scene', 'scene', 0, [frame])
    prepare_crop(tmp_path / 'raw', tmp_path / 'crop', 'console-crop', crop, maximum_timestamp_skew_ns=0)
    pose = np.eye(4); pose[:3, 3] = [0, 0, 1]
    priors = tmp_path / 'priors.json'
    priors.write_text(json.dumps({'poses': {'10': pose.tolist()}, 'scene_revision': 0, 'ground_height': 0}))
    session = tmp_path / 'session'
    try:
        console.open_session(session)
        console.product.setText(str(tmp_path / 'crop'))
        console.priors.setText(str(priors))
        console.identify()
        assert console.running
        wait_for(lambda: not console.running, 30)
        assert console.proposal is not None, console.perception_info.toPlainText()
        assert console.proposal['valid'], console.proposal
        assert console.cloud_path is not None
        assert console.accept_button.isEnabled()
        from PySide6.QtTest import QSignalSpy
        changes = QSignalSpy(console.changed)
        console.layers['Observed cloud'].setChecked(False)
        assert changes.count() == 1
        console.layers['Observed cloud'].setChecked(True)
        console.decide(True)
        assert console.runtime.review.snapshot()['revision'] == 1
        assert not console.accept_button.isEnabled()
        console.load_demo(True)
        console.start()
        # Advance the same runtime deterministically; Qt remains responsible
        # for process lifecycle and button/review wiring above.
        console.timer.stop()
        clock = SimulationClock()
        clock.seconds = console.runtime.clock()
        console.runtime.clock = console.runtime.device.clock = clock
        for _ in range(2000):
            clock.advance()
            console.tick()
            if console.runtime.state['status'] == 'succeeded':
                break
        assert console.runtime.state['status'] == 'succeeded'
        assert 'sample_completed' in console.history_text.toPlainText()
        assert console.joints.item(0, 1).text()
    finally:
        console.close()
        console.deleteLater()
    reopened = SessionConsole(ROOT)
    try:
        reopened.open_session(session)
        assert reopened.runtime.review.snapshot()['revision'] == 1
        assert reopened.proposal['valid']
        assert reopened.cloud_path is not None
        assert not reopened.accept_button.isEnabled()
        reopened.tick()
        assert 'succeeded' in reopened.history_text.toPlainText()
        assert reopened.runtime.device.stopped
    finally:
        reopened.close()
        reopened.deleteLater()


def test_perception_rendering_keeps_frame_roi_and_shared_material(tmp_path):
    import pyvista as pv
    from PySide6.QtWidgets import QApplication
    from stacking_common.profiles.visualization import load_visualization_profile
    from stacking_excavator.gui.operations.scene_view import OperationsSceneView
    app = QApplication.instance() or QApplication([])
    view = OperationsSceneView(None, ROOT / 'apps/excavator/assets/stone',
                              load_visualization_profile(ROOT / 'apps/excavator/config/visualization.yml'))
    # Resolve actual catalog path through the inventory profile.
    from stacking_common.profiles.inventory import load_inventory_profile
    view.asset_directory = ROOT / load_inventory_profile(ROOT / 'apps/excavator/config/inventory.yml').model_directory
    proposal = dict(sha256='a'*64, kind='inhand', frame='stone_grab_geometry', scene_revision=0,
        objects=[dict(object_id=10, initial_pose=np.eye(4).tolist(), pose=np.eye(4).tolist(), valid=True)],
        crop=dict(half_extents=[1, 2, 3], margins=[.1]*3, reference_T_box=np.eye(4).tolist()))
    points = tmp_path / 'processed.ply'
    pv.PolyData(np.array([[0., 0., 0.], [.1, .1, .1]])).save(points)
    try:
        view.render_perception(proposal, points, layers={}, reset_camera=True)
        assert 'perception/cloud' in view.actors
        assert 'perception/crop' in view.actors
        actor = view.actors['perception/pose/10']
        assert actor.prop.opacity == pytest.approx(.3)
        assert actor.prop.specular == 0
        view.set_object_geometry_mode('ply')
        view.render_perception(proposal, points, layers={'Initial poses': False}, reset_camera=False)
        assert 'perception/initial_pose/10' not in view.actors
        assert 'perception/pose/10' in view.actors
    finally:
        view.close()
