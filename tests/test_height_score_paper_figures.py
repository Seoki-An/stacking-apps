"""Lower-surface rasterization must retain real mesh heights and empty space."""

import numpy as np
from types import SimpleNamespace

from height_score_paper_figures import candidate_bottom_patch, rasterize_bottom_height, render_maps
from height_score_boundary import boundary_display_map, extended_score_fields
from stacking_common.visualization.sampling import initial_pose_score_view
from test_initial_pose_score_view import config, scene


def test_vector_overlay_preserves_alpha_blend_at_score_and_target_boundaries(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib import colormaps

    score = np.array([[0., 1.], [2., 3.]])
    bottom = np.full((3, 3), np.nan)
    bottom[1, 1] = .2  # One bottom cell crosses both score axes and the target boundary.
    view = SimpleNamespace(x=np.array([-.5, .5]), y=np.array([-.5, .5]),
        fields={"score": score, "support_likelihood": score.copy()}, weights={"support_likelihood": 1.},
        valid=np.ones((2, 2), dtype=bool), scene_height=np.zeros((3, 3)),
        scene_xlim=np.array([-1.2, 1.2]), scene_ylim=np.array([-1.2, 1.2]),
        initial_xy=np.array([0., 0.]), target_bounds=(np.array([-.4, -.4]), np.array([.4, .4])))
    captured = []

    def save(fig, path, **kwargs):
        if path.name == "height_score_overview.pdf":
            captured.append(fig.axes[3].collections[-1])  # Total score, bottom-left panel.

    monkeypatch.setattr(Figure, "savefig", save)
    render_maps(tmp_path, view, bottom)
    overlay = captured[0]
    assert not overlay.get_rasterized()
    colors = overlay.get_facecolors()
    np.testing.assert_allclose(colors[:, 3], 1.)
    for point, expected in (
            ((-.2, -.2), .5*np.ones(3)+.5*np.array(colormaps["viridis"](0.))[:3]),
            ((.2, .2), .5*np.ones(3)+.5*np.array(colormaps["viridis"](1.))[:3]),
            ((-.5, -.5), np.full(3, .5))):  # Outside target: white overlay over black.
        matches = [i for i, path in enumerate(overlay.get_paths()) if path.contains_point(point)]
        assert len(matches) == 1
        np.testing.assert_allclose(colors[matches[0], :3], expected)
    np.testing.assert_array_equal(view.fields["score"], score)


def test_candidate_bottom_patch_preserves_metric_coordinates_holes_and_height_differences():
    view = SimpleNamespace(scene_xlim=(-2., 2.), scene_ylim=(-3., 3.), initial_xy=(.5, -1.))
    bottom = np.full((7, 5), np.nan)
    bottom[2:5, 1:4] = [[4., 4.2, np.nan], [4.1, np.nan, 4.3], [4.2, 4.4, 4.5]]
    original = bottom.copy()
    x, y, heights, minimum = candidate_bottom_patch(view, bottom)
    np.testing.assert_allclose(x, [-1.5, -.5, .5])
    np.testing.assert_allclose(y, [0., 1., 2.])
    np.testing.assert_allclose(heights, bottom[2:5, 1:4]-4., equal_nan=True)
    assert minimum == 4.
    np.testing.assert_array_equal(bottom, original)


def test_bottom_map_selects_lower_surface_and_applies_recorded_pose():
    xy = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    lower = np.column_stack((xy, .2 * xy[:, 0] + .1 * xy[:, 1]))
    upper = np.column_stack((xy, np.full(4, 2.)))
    points = np.vstack((lower, upper))
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    pose = np.array([.4, -.2, 3., 0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
    x, y = np.linspace(-1.5, 1.5, 25), np.linspace(-1.5, 1.5, 25)
    bottom = rasterize_bottom_height(points, faces, pose, x, y)
    xx, yy = np.meshgrid(x, y)
    local_x, local_y = yy + .2, .4 - xx
    inside = (np.abs(local_x) < .99) & (np.abs(local_y) < .99)
    outside = (np.abs(local_x) > 1.01) | (np.abs(local_y) > 1.01)
    np.testing.assert_allclose(bottom[inside], (3. + .2*local_x + .1*local_y)[inside], atol=1e-12)
    assert np.isnan(bottom[outside]).all()


def test_bottom_map_preserves_empty_space_instead_of_convexifying_mesh():
    square = np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [.2, .2, 0.], [-.2, .2, 0.]])
    points = np.vstack((square + [-.6, 0., 0.], square + [.6, 0., 0.]))
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    axis = np.linspace(-1., 1., 21)
    bottom = rasterize_bottom_height(points, faces, [0., 0., 0., 0., 0., 0., 1.], axis, axis)
    assert np.isfinite(bottom[10, 4]) and np.isfinite(bottom[10, 16])
    assert np.isnan(bottom[10, 10])  # A whole-stone convex hull would incorrectly fill this gap.


def test_boundary_continuation_retains_scores_and_mask_but_exposes_rim(config):
    inventory, state = scene()
    pose = np.array([.01, .04, .3, 0., 0., 0., 1.])
    view = initial_pose_score_view(inventory, state, 20, pose, **config)
    saved = view.fields['target_boundary'].copy()
    mask = view.valid.copy()
    result = boundary_display_map(inventory, state, 1, pose, view,
                                 config['planar_config'], config['settings'])
    known = np.isfinite(saved)
    np.testing.assert_array_equal(result['score'][known], saved[known])
    np.testing.assert_array_equal(view.fields['target_boundary'], saved)
    np.testing.assert_array_equal(view.valid, mask)
    xx, yy = np.meshgrid(view.x, view.y)
    lower, upper = result['lower'], result['upper']
    outside_anchor = (xx < lower[..., 0]) | (xx > upper[..., 0]) | (yy < lower[..., 1]) | (yy > upper[..., 1])
    rim = outside_anchor & ~known & np.isfinite(result['score'])
    assert rim.any()
    np.testing.assert_allclose(result['score'][rim], 1.)


def test_all_display_scores_reach_target_and_total_matches_components(config):
    inventory, state = scene()
    pose = np.array([.01, .04, .3, 0., 0., 0., 1.])
    view = initial_pose_score_view(inventory, state, 20, pose, **config)
    saved = {key: value.copy() for key, value in view.fields.items()}
    original_mask = view.valid.copy()
    boundary = boundary_display_map(inventory, state, 1, pose, view,
                                    config['planar_config'], config['settings'])
    fields = extended_score_fields(view, boundary)
    xx, yy = np.meshgrid(view.x, view.y)
    lower, upper = view.target_bounds
    inside = (xx >= lower[0]) & (xx <= upper[0]) & (yy >= lower[1]) & (yy <= upper[1])
    assert (inside & ~view.valid).any()
    for name, values in fields.items():
        assert np.isfinite(values[inside]).all()
        np.testing.assert_array_equal(values[view.valid], view.fields[name][view.valid])
    total = sum(fields[key] * weight for key, weight in view.weights.items())
    np.testing.assert_allclose(total[inside], fields['score'][inside], rtol=1e-12, atol=1e-12)
    for name in view.fields:
        np.testing.assert_array_equal(saved[name], view.fields[name])
    np.testing.assert_array_equal(view.valid, original_mask)
