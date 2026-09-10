"""Target taper survives storage, projection, and both operation renderers."""

from dataclasses import replace
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from stacking_common.artifacts import StackRunStore, decode_mcts_debug, encode_mcts_debug
from stacking_common.domain import DirectionalTaper
from stacking_common.visualization import MCTSDebugArtifact, TargetVolume
from stacking_common.visualization.operations import OperationsController


def tapered_target():
    return TargetVolume(
        origin=[1., 2., 3.], width=2., length=3.5, height=2.5,
        structure_offset=[.1, -.2, .3],
        taper=DirectionalTaper.from_degrees(5., 7., 9., 3.),
    )


def test_directional_taper_round_trips_and_old_artifacts_default_to_zero(tmp_path):
    target = tapered_target()
    artifact = MCTSDebugArtifact("stack", (), target)
    encoded = encode_mcts_debug(artifact)
    decoded = decode_mcts_debug(encoded)
    assert decoded.target.taper == target.taper
    np.testing.assert_array_equal(decoded.target.points, target.points)
    path = tmp_path / "target.stack"
    StackRunStore.save_mcts(artifact, path)
    assert StackRunStore.load_mcts(path).target.taper == target.taper

    del encoded["target"]["taper"]
    old = decode_mcts_debug(encoded).target
    assert old.taper == DirectionalTaper()
    np.testing.assert_allclose(old.points[4:] - old.points[:4], [[0., 0., 2.5]] * 4)


def test_planning_target_translation_preserves_taper_and_offset():
    target = tapered_target()
    controller = OperationsController(MCTSDebugArtifact("stack", (), target))
    record = SimpleNamespace(problem=SimpleNamespace(metadata={
        "planning_T_search": [-6., 3., 0., 0., 0., 0., 1.],
    }))
    projected = controller._planning_target(record)
    assert projected.taper == target.taper
    np.testing.assert_allclose(projected.points, target.points + [-6., 3., 0.])


@pytest.mark.parametrize("platform", ["tabletop", "excavator"])
def test_viewer_draws_the_correct_asymmetric_tapered_vertices(platform):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    module = importlib.import_module(f"stacking_{platform}.gui.operations.scene_view")
    target = tapered_target()
    view = Mock()
    module.OperationsSceneView._add_target(view, target)
    mesh = view.add_mesh.call_args.args[0]
    points = mesh.points - target.origin - target.structure_offset
    expected_top = np.array([
        [-1. + 2.5 * np.tan(np.deg2rad(7.)), -1.75 + 2.5 * np.tan(np.deg2rad(3.)), 2.5],
        [1. - 2.5 * np.tan(np.deg2rad(5.)), -1.75 + 2.5 * np.tan(np.deg2rad(3.)), 2.5],
        [1. - 2.5 * np.tan(np.deg2rad(5.)), 1.75 - 2.5 * np.tan(np.deg2rad(9.)), 2.5],
        [-1. + 2.5 * np.tan(np.deg2rad(7.)), 1.75 - 2.5 * np.tan(np.deg2rad(9.)), 2.5],
    ])
    np.testing.assert_allclose(points[4:], expected_top)
    assert mesh.n_cells == 5  # Four walls and ceiling; no coplanar floor.
    assert mesh.n_open_edges == 4

    # Existing zero-taper tabletop shells retain exactly the same vertices.
    module.OperationsSceneView._add_target(view, replace(target, taper=DirectionalTaper()))
    box = view.add_mesh.call_args.args[0]
    np.testing.assert_allclose(box.points[4:] - box.points[:4], [[0., 0., 2.5]] * 4)


def test_ablation_target_top_is_approximately_1_563_metres_wide():
    from stacking_excavator import ExcavatorApplication
    from pathlib import Path
    profile = ExcavatorApplication.from_repository(Path(__file__).resolve().parents[1]).inventory_profile.target
    target = TargetVolume(profile.origin, profile.width, profile.length, profile.height,
                          taper=profile.taper)
    assert np.ptp(target.points[4:, 0]) == pytest.approx(1.56255668237038)
