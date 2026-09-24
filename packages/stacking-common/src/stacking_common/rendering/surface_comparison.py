"""Cross-sections and signed height residuals on identical observed cells."""
from pathlib import Path
import numpy as np


def render_surface_comparison(directory, regions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    titles = []
    for i in range(len(regions)):
        titles.extend([f'Region {i}: before ΔZ', f'Region {i}: after ΔZ', f'Region {i}: X cross-section', f'Region {i}: Y cross-section'])
    interactive = make_subplots(rows=len(regions), cols=4, subplot_titles=titles, horizontal_spacing=.065, vertical_spacing=.10)
    fig, axes = plt.subplots(len(regions), 4, figsize=(22, 4.5*len(regions)), squeeze=False, constrained_layout=True)
    limit = max(1., max(float(np.max(np.abs((a[key]-a['reference'])[a['common']]*1000)))
                       for a in regions for key in ('rgb_before', 'rgb_after')))
    colors = ['#788594', '#548cfa', '#e9a328', '#18a999']
    keys = ['reference', 'reference_after', 'rgb_before', 'rgb_after']
    names = ['FEMTO before', 'FEMTO after', 'RGB before alignment', 'RGB after shared alignment']
    for row, a in enumerate(regions):
        x, y = a['xy'][:, 0, 0]*1000, a['xy'][0, :, 1]*1000
        for col, key in enumerate(('rgb_before', 'rgb_after')):
            delta = np.where(a['common'], (a[key]-a['reference'])*1000, np.nan)
            artist = axes[row, col].pcolormesh(x, y, delta.T, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='nearest')
            axes[row, col].set(xlabel='X (mm)', ylabel='Y (mm)', aspect='equal')
            fig.colorbar(artist, ax=axes[row, col], label='RGB − FEMTO Z (mm)', shrink=.75)
            interactive.add_trace(go.Heatmap(x=x.tolist(), y=y.tolist(), z=delta.T.tolist(), coloraxis='coloraxis',
                                           hovertemplate='X %{x:.1f} mm<br>Y %{y:.1f} mm<br>ΔZ %{z:.2f} mm<extra></extra>'), row=row+1, col=col+1)
            interactive.update_xaxes(title_text='X (mm)', row=row+1, col=col+1)
            interactive.update_yaxes(title_text='Y (mm)', row=row+1, col=col+1)
        for axis, fixed in enumerate(a['slice_indices']):
            col = axis+2
            if fixed < 0:
                axes[row, col].text(.1, .5, 'No sufficiently supported central strip', transform=axes[row, col].transAxes)
                continue
            coordinate = x if axis == 0 else y
            selector = (slice(None), int(fixed)) if axis == 0 else (int(fixed), slice(None))
            support = a['common'][selector]
            for key, name, color in zip(keys, names, colors):
                z = np.where(support, a[key][selector]*1000, np.nan)
                axes[row, col].plot(coordinate, z, '.-', color=color, label=name, linewidth=1)
                interactive.add_trace(go.Scatter(x=coordinate.tolist(), y=z.tolist(), name=name, legendgroup=key,
                                                showlegend=(row == 0 and axis == 0), connectgaps=False,
                                                mode='lines+markers', line=dict(color=color)), row=row+1, col=col+1)
            axes[row, col].set(xlabel=('X' if axis == 0 else 'Y')+' (mm)', ylabel='Z (mm)')
            axes[row, col].grid(alpha=.2)
            interactive.update_xaxes(title_text=('X' if axis == 0 else 'Y')+' (mm)', row=row+1, col=col+1)
            interactive.update_yaxes(title_text='Z (mm)', row=row+1, col=col+1)
        for col in range(4):
            axes[row, col].set_title(titles[row*4+col])
    axes[0, 2].legend(fontsize=8)
    title = 'Stone surface comparison · one shared rigid fit · same interior cells · diagnostic only'
    fig.suptitle(title)
    directory = Path(directory)
    fig.savefig(directory/'surface_comparison.png', dpi=140)
    plt.close(fig)
    interactive.update_layout(title=title, height=390*len(regions), width=1650, template='plotly_white',
                              coloraxis=dict(colorscale='RdBu_r', cmin=-limit, cmax=limit, colorbar_title='ΔZ mm'),
                              legend=dict(orientation='h', y=-.08))
    interactive.write_html(directory/'overview.html', include_plotlyjs=True, full_html=True)
