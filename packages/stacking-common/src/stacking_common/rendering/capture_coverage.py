"""Portable, interactive capture footprint figure."""
from pathlib import Path
from html import escape

import plotly.graph_objects as go


def coverage_summary(coverage):
    center = next((f for f in coverage['footprints'] if f['label'] == 'center' and f['bounded']), None)
    def dimensions(span):
        return f'{span[0]*100:.1f} × {span[1]*100:.1f} cm'
    text = f"RGB footprint at assumed {coverage['distance_m']*100:.0f} cm target depth"
    if center:
        text += f"; center {dimensions(center['span_m'])}"
    span = coverage['combined_bounding_span_m']
    text += f'; all-view bounding span {dimensions(span)}' if span else '; all-view span unbounded on this plane'
    return text


def render_coverage(coverage, path):
    figure = go.Figure()
    unbounded = [f['label'] for f in coverage['footprints'] if not f['bounded']]
    for footprint in coverage['footprints']:
        if not footprint['bounded']:
            continue
        xy = footprint['plane_xy_m']
        figure.add_trace(go.Scatter(x=[p[0]*100 for p in xy]+[xy[0][0]*100],
            y=[p[1]*100 for p in xy]+[xy[0][1]*100], mode='lines',
            name=footprint['label'], fill='toself', opacity=.3))
    figure.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='look-at target',
                                marker=dict(color='black', size=10)))
    route_status = coverage.get('route_status', 'validated')
    notice = ('Route IK validated; collisions not checked.' if route_status == 'validated' else
              'DESIRED CAMERA COVERAGE ONLY — robot route ' + route_status + '. Capture is unavailable.')
    figure.update_layout(title=notice + '<br>' + coverage_summary(coverage).replace('; ', '<br>'),
        template='plotly_white', xaxis_title='Reference-plane X (cm)',
        yaxis=dict(title='Reference-plane Y (cm)', scaleanchor='x', scaleratio=1),
        margin=dict(t=160, b=180),
        annotations=[dict(text='Plane perpendicular to the central camera axis at the assumed target depth.'
            '<br>Not a measured table plane. Bounding span includes gaps; it is not stereo overlap.'
            '<br>Occlusion, texture, stereo overlap and reconstruction crops reduce usable PCD coverage.'
            + ('<br><a href="preview.html">Open robot route preview</a>' if route_status == 'validated' else
               '<br>' + escape(coverage.get('route_error', 'Reachability has not been checked.'))) +
            (f'<br>Unbounded footprints (not drawn): {", ".join(unbounded)}' if unbounded else ''),
            x=0, y=-.2, xref='paper', yref='paper', showarrow=False, align='left')])
    figure.write_html(Path(path), include_plotlyjs=True)
