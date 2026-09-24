"""Standalone inspection of separately labelled geometric evidence."""
from pathlib import Path
import numpy as np
from stacking_common.pointcloud import PointCloudData


def render_scene_layers(path, layers):
    """Render layer/group toggles with source-row IDs; subsample display only.

    Each layer has points, source_indices, group_ids, label, color and visible.
    Group -1 means background. Group IDs describe regions, not object identities.
    """
    import plotly.graph_objects as go
    figure = go.Figure()
    groups = sorted({int(g) for layer in layers for g in np.unique(layer['group_ids']) if g >= 0})
    modes = [('Stone regions', None), ('Whole scene', -1)] + [(f'Region {g}', g) for g in groups]
    visibility = []
    for mode, group in modes:
        mode_visibility = []
        for layer in layers:
            points = PointCloudData(layer['points']).points
            selected = (np.ones(len(points), bool) if group == -1 else
                        layer['group_ids'] >= 0 if group is None else layer['group_ids'] == group)
            ids = np.flatnonzero(selected)
            ids = ids[::max(1, int(np.ceil(len(ids)/16000)))]
            p = points[ids]
            default = True if layer['visible'] else 'legendonly'
            mode_visibility.append(default)
            figure.add_trace(go.Scatter3d(
                x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                customdata=layer['source_indices'][ids].tolist(),
                name=f"{layer['label']} ({int(selected.sum()):,})", mode='markers',
                marker=dict(size=2, color=layer['color'], opacity=.8),
                visible=default if len(visibility) == 0 else False,
                hovertemplate='Source row %{customdata}<br>X %{x:.4f} m<br>Y %{y:.4f} m<br>Z %{z:.4f} m<extra>%{fullData.name}</extra>'))
        visibility.append(mode_visibility)
    buttons = []
    for i, (label, _) in enumerate(modes):
        visible = [False]*len(figure.data)
        start = i*len(layers)
        visible[start:start+len(layers)] = visibility[i]
        buttons.append(dict(label=label, method='update', args=[{'visible': visible}]))
    figure.update_layout(
        title='FEMTO + wrist RGB scene · separate evidence layers · accuracy unverified',
        template='plotly_white', margin=dict(l=0, r=0, b=50, t=100),
        updatemenus=[dict(buttons=buttons, x=0, y=1.06, xanchor='left')],
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        legend=dict(itemsizing='constant'),
        annotations=[dict(text='Frame: base_link · Click legend to toggle layers · Possible extensions are unverified · Display subsampling only',
                          x=0, y=-.07, xref='paper', yref='paper', showarrow=False)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
