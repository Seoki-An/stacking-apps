"""Spatial disagreement figures from canonical height grids, in metres."""

from itertools import combinations
from pathlib import Path

import numpy as np


def render_height_differences(directory, xy, common, heights, labels, *, frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pairs = list(combinations(range(len(labels)), 2))
    differences = [(heights[j]-heights[i])*1000 for i, j in pairs]
    limit = max(1., max(float(np.max(np.abs(delta))) for delta in differences))
    rows, cols = (len(pairs)+1)//2, min(2, len(pairs))
    titles = [f'{labels[j]} − {labels[i]}' for i, j in pairs]
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False, constrained_layout=True)
    interactive = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    x, y = xy[:, 0, 0], xy[0, :, 1]
    for index, (delta, title) in enumerate(zip(differences, titles)):
        z = np.full(common.shape, np.nan)
        z[common] = delta
        ax = axes.ravel()[index]
        artist = ax.pcolormesh(x, y, z.T, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='nearest')
        ax.set(xlabel='X (m)', ylabel='Y (m)', title=title, aspect='equal')
        interactive.add_trace(go.Heatmap(x=x, y=y, z=z.T, coloraxis='coloraxis',
            hovertemplate='X=%{x:.3f} m<br>Y=%{y:.3f} m<br>ΔZ=%{z:.3f} mm<extra></extra>'),
            row=index//cols+1, col=index%cols+1)
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)
    title = f'Table height disagreement · {frame} · common observed cells · no correction'
    fig.suptitle(title)
    fig.colorbar(artist, ax=list(axes.ravel()[:len(pairs)]), label='ΔZ (mm)', shrink=.7)
    directory = Path(directory)
    fig.savefig(directory/'table_differences.png', dpi=160)
    plt.close(fig)
    interactive.update_xaxes(title_text='X (m)')
    interactive.update_yaxes(title_text='Y (m)')
    interactive.update_layout(title=title, height=400*rows, template='plotly_white',
        coloraxis=dict(colorscale='RdBu_r', cmin=-limit, cmax=limit, colorbar_title='ΔZ (mm)'))
    interactive.write_html(directory/'table_differences.html', include_plotlyjs=True)
