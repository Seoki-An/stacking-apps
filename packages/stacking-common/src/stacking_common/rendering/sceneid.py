"""Compare portable SceneID fixture observations and posed model surfaces."""
from html import escape
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


COLORS = {'Initial': '#ef8354', 'Proposed': '#548cfa', 'Ground truth': '#18a999'}


def comparison_layers(directory, metadata, initial, truth, result):
    """Use entity IDs, never result order, to place body-frame model samples.

    Call after load_fixture has verified the fixture assets. No fitting or
    alignment is performed here, and observation RGB values remain unchanged.
    """
    directory = Path(directory)
    if result['frame'] != metadata['frame']:
        raise ValueError('result frame does not match fixture')
    with np.load(directory/'observation.npz', allow_pickle=False) as data:
        observation = dict(label='Captured RGB', points=data['points'].copy(),
                           colors=data['colors'].copy(), group='Captured RGB')
    models = {}
    for stone in metadata['stones']:
        with np.load(directory/stone['model_file'], allow_pickle=False) as data:
            models[stone['entity_id']] = data['surface_points'].copy()
    labels = {s['entity_id']: s['stone_id'] for s in metadata['stones']}
    layers = [observation]
    for group, poses in [('Initial', initial['stones']),
                         ('Proposed', result.get('proposed_poses', [])),
                         ('Ground truth', truth['stones'])]:
        seen = set()
        for record in poses:
            entity = record['entity_id']
            if entity not in models or entity in seen:
                raise ValueError('result contains unknown or duplicate entity IDs')
            seen.add(entity)
            pose = np.asarray(record['pose'], dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError('invalid visualization pose')
            points = models[entity] @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
            layers.append(dict(label=f'{group}: stone {labels[entity]} (entity {entity})',
                               points=points, color=COLORS[group], group=group))
    return layers


def result_title(result):
    status = result.get('status', 'scan evaluation only')
    accepted = 'accepted snapshot' if result.get('accepted_snapshot') else 'NO accepted snapshot'
    return f'SceneID — {status} — {accepted}'


def write_comparison(path, layers, result):
    """Save a self-contained report with RGB and per-stone pose-layer toggles."""
    import plotly.graph_objects as go

    figure = go.Figure()
    selected = 'Proposed' if any(x['group'] == 'Proposed' for x in layers) else 'Initial'
    for layer in layers:
        step = max(1, int(np.ceil(len(layer['points'])/20000)))
        p = layer['points'][::step]
        color = layer.get('color')
        if 'colors' in layer:
            rgb = np.rint(np.clip(layer['colors'][::step], 0, 1)*255).astype(int)
            color = [f'rgb({r},{g},{b})' for r, g, b in rgb]
        figure.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode='markers', name=layer['label'],
            visible=True if layer['group'] in ('Captured RGB', selected) else 'legendonly',
            marker=dict(size=2, color=color, opacity=.85)))
    buttons = [dict(label='RGB only', method='update', args=[{'visible': [
        True if x['group'] == 'Captured RGB' else 'legendonly' for x in layers]}])]
    for group in COLORS:
        if any(x['group'] == group for x in layers):
            buttons.append(dict(label=group, method='update', args=[{'visible': [
                True if x['group'] in ('Captured RGB', group) else 'legendonly' for x in layers]}]))
    figure.update_layout(title=escape(result_title(result)), template='plotly_white',
        scene=dict(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
        updatemenus=[dict(type='buttons', direction='right', buttons=buttons, x=0, y=1.08)],
        margin=dict(t=120, b=65), legend=dict(itemsizing='constant'),
        annotations=[dict(text=f"Frame: {escape(result['frame'])} · Model surface samples · "
            'Click legend to toggle stones/layers · Display subsampling only',
            x=0, y=-.1, xref='paper', yref='paper', showarrow=False)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True)
    return path


def show_comparison(layers, result):
    """Open a desktop viewer; keys select poses without changing the camera."""
    import open3d as o3d

    clouds = []
    for layer in layers:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(layer['points']))
        if 'colors' in layer:
            cloud.colors = o3d.utility.Vector3dVector(layer['colors'])
        else:
            color = layer['color'].lstrip('#')
            cloud.paint_uniform_color([int(color[i:i+2], 16)/255 for i in (0, 2, 4)])
        clouds.append(cloud)
    state = dict(group='Proposed' if any(x['group'] == 'Proposed' for x in layers) else 'Initial',
                 rgb=True, truth=False)
    active = []
    viewer = o3d.visualization.VisualizerWithKeyCallback()

    def refresh(view):
        for cloud in active:
            view.remove_geometry(cloud, reset_bounding_box=False)
        active.clear()
        for layer, cloud in zip(layers, clouds):
            if (layer['group'] == state['group'] or
                (state['rgb'] and layer['group'] == 'Captured RGB') or
                (state['truth'] and layer['group'] == 'Ground truth')):
                view.add_geometry(cloud, reset_bounding_box=False)
                active.append(cloud)
        print(f"SceneID view: {state['group']}; RGB={state['rgb']}; GT overlay={state['truth']}", flush=True)
        return True

    def select(group):
        def callback(view):
            state['group'] = group
            if not group:
                state.update(rgb=True, truth=False)
            return refresh(view)
        return callback

    def toggle(key):
        def callback(view):
            state[key] = not state[key]
            return refresh(view)
        return callback

    try:
        if not viewer.create_window(window_name=result_title(result), width=1280, height=800):
            raise RuntimeError('Open3D requires a graphical desktop. Open visualization/overview.html instead.')
        for key, group in [('0', ''), ('1', 'Initial'), ('2', 'Proposed'), ('3', 'Ground truth')]:
            viewer.register_key_callback(ord(key), select(group))
        viewer.register_key_callback(ord('C'), toggle('rgb'))
        viewer.register_key_callback(ord('G'), toggle('truth'))
        refresh(viewer)
        viewer.reset_view_point(True)
        viewer.get_render_option().background_color = np.array([.08, .09, .11])
        viewer.get_render_option().point_size = 2.
        view = viewer.get_view_control()
        view.set_front([0., -1., 1.]); view.set_up([0., 0., 1.]); view.set_zoom(.7)
        print('0: RGB only | 1: initial (orange) | 2: proposed (blue) | 3: GT (green) | '
              'C: RGB toggle | G: GT overlay | Q: close', flush=True)
        viewer.run()
    finally:
        viewer.destroy_window()
