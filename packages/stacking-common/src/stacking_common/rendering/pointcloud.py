"""Portable offline point-cloud figures from canonical arrays and transforms."""

from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData, validate_rigid_transform


COLORS = ('#18a999', '#ef8354', '#548cfa', '#b873d9', '#e9b949', '#8797a5')


def render_clouds(directory, layers, camera_poses, *, title, frame='base_link', object_poses=None):
    """Write standalone interactive HTML and a PNG; decimation is display-only.

    layers maps labels to Nx3 arrays; camera_poses and object_poses map labels
    to transforms in the rendered frame.
    The caller owns provenance. This renderer knows nothing about camera drivers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure = go.Figure()
    png = plt.figure(figsize=(12, 8), facecolor='#f5f7fb')
    axes = png.add_subplot(111, projection='3d')
    extents = []
    for i, (label, points) in enumerate(layers.items()):
        points = PointCloudData(points).points
        displayed = points[::max(1, int(np.ceil(len(points)/12000)))]
        if not len(displayed):
            continue
        extents.append(displayed)
        color = COLORS[i % len(COLORS)]
        figure.add_trace(go.Scatter3d(x=displayed[:, 0].tolist(), y=displayed[:, 1].tolist(),
            z=displayed[:, 2].tolist(), mode='markers', name=label,
            marker=dict(size=1.5, color=color, opacity=.75)))
        axes.scatter(*displayed.T, s=.5, alpha=.6, color=color, label=label, rasterized=True)
    for label, value in camera_poses.items():
        pose = validate_rigid_transform(value)
        # Symbolic frustum, not a claim about a particular lens's field of view.
        corners = np.array([[-.025, -.018, .045], [.025, -.018, .045],
                            [.025, .018, .045], [-.025, .018, .045]]) @ pose[:3, :3].T + pose[:3, 3]
        origin = pose[:3, 3]
        segments = []
        for i in range(4):
            segments.extend([origin, corners[i], corners[(i+1) % 4], [np.nan]*3])
        segments = np.asarray(segments)
        figure.add_trace(go.Scatter3d(x=segments[:, 0].tolist(), y=segments[:, 1].tolist(),
            z=segments[:, 2].tolist(), mode='lines', line=dict(color='#24334a', width=3),
            name=label + ' camera', showlegend=False))
        axes.plot(*segments.T, color='#24334a', linewidth=1.)
        axes.text(*origin, label, fontsize=8)
        extents.append(np.vstack((origin, corners)))
    for label, value in (object_poses or {}).items():
        pose = validate_rigid_transform(value)
        origin = pose[:3, 3]
        ends = origin + .025 * pose[:3, :3].T
        for axis_name, end, color in zip('XYZ', ends, ('#d62728', '#2ca02c', '#1f77b4')):
            figure.add_trace(go.Scatter3d(
                x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
                mode='lines+markers', name=f'{label} {axis_name}',
                line=dict(color=color, width=7), marker=dict(color=color, size=3)))
            axes.plot([origin[0], end[0]], [origin[1], end[1]],
                      [origin[2], end[2]], color=color, linewidth=2.5)
        axes.text(*origin, label, fontsize=8)
        extents.append(np.vstack((origin, ends)))
    if extents:
        all_points = np.concatenate(extents)
        centre = (all_points.min(axis=0) + all_points.max(axis=0))/2
        half = max(np.ptp(all_points, axis=0).max()/2, .01) * 1.08
        for setter, value in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), centre):
            setter(value-half, value+half)
    axes.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
    axes.set_box_aspect((1, 1, 1))
    axes.view_init(elev=32, azim=-58)
    axes.legend(loc='upper left', markerscale=5, fontsize=8)
    png.text(.04, .025, f'Frame: {frame} · Display subsampling only · Camera frusta are symbolic', color='#48566a')
    png.tight_layout()
    png.savefig(directory / 'overview.png', dpi=170)
    plt.close(png)
    figure.update_layout(title=title, template='plotly_white',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=65), legend=dict(itemsizing='constant'))
    figure.write_html(directory / 'overview.html', include_plotlyjs=True, full_html=True)
    # A separate close view keeps the scene legible when cameras are far above it.
    surfaces = np.concatenate([PointCloudData(value).points for value in layers.values()])
    surfaces = surfaces[::max(1, int(np.ceil(len(surfaces)/60000)))]
    detail = plt.figure(figsize=(13, 6), facecolor='#f5f7fb')
    spatial = detail.add_subplot(121, projection='3d')
    top = detail.add_subplot(122)
    if len(surfaces):
        spatial.scatter(*surfaces.T, c=surfaces[:, 2], cmap='viridis', s=1., rasterized=True)
        color = top.scatter(surfaces[:, 0], surfaces[:, 1], c=surfaces[:, 2], cmap='viridis', s=1., rasterized=True)
        detail.colorbar(color, ax=top, shrink=.7, label='Z (m)')
        spatial.set_box_aspect(np.maximum(np.ptp(surfaces, axis=0), .001))
    spatial.view_init(elev=38, azim=-62)
    spatial.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Observed surfaces')
    top.set(xlabel='X (m)', ylabel='Y (m)', title='Top view', aspect='equal')
    detail.suptitle(title + ' · height colors', fontsize=14)
    detail.tight_layout()
    detail.savefig(directory / 'surfaces.png', dpi=170)
    plt.close(detail)
