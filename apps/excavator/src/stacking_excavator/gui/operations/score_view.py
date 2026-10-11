"""Top-view comparison of scene height and initial-placement score terms."""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)


FIELDS = (
    ('score', 'Total score', 'higher is better'),
    ('height', 'Normalized support height', 'lower is better'),
    ('support_likelihood', 'Support likelihood', 'higher is better'),
    ('fill_area', 'Fill area', 'higher is better'),
    ('target_boundary', 'Target boundary', 'higher is better'),
    ('planar_bias', 'Planar bias', 'higher is better'),
    ('placement_height', 'Minimum placement Z [m]', 'geometric clearance; not initial Z'),
)


class InitialPoseScoreDialog(QDialog):
    """A fixed step/candidate snapshot; no computation during MCTS or replay."""

    def __init__(self, data, *, step, candidate, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle(f'XY score maps · step {step} · object {candidate.object_id} · #{candidate.rank}')
        self.resize(1100, 820)
        layout = QVBoxLayout(self)
        self.description = QLabel(
            f'Step {step} · candidate #{candidate.rank} · object {candidate.object_id}\n'
            'Fixed recorded initial orientation · pre-search scene · normal floor gates\n'
            'Recomputed with recorded settings; height normalization uses only this orientation. '
            'Retry relaxation is not reconstructed.\n'
            'White: candidate footprint · magenta: initial XY · yellow: target base.')
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        controls = QHBoxLayout()
        self.field = QComboBox()
        for name, label, _ in FIELDS:
            self.field.addItem(label, name)
        self.weighted = QCheckBox('Weighted contribution')
        self.weighted.setToolTip('Multiply the displayed term by its recorded signed score weight')
        save = QPushButton('Save PNG…')
        save.clicked.connect(self._save)
        for widget in (self.field, self.weighted, save):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.plot = QtInteractor(self, shape=(1, 2))
        layout.addWidget(self.plot.interactor)
        self.field.currentIndexChanged.connect(self.render_map)
        self.weighted.toggled.connect(self.render_map)
        self.render_map()

    def render_map(self, *_):
        data = self.data
        name, label, preference = FIELDS[self.field.currentIndex()]
        self.weighted.setEnabled(name in data.weights)
        values = data.fields[name].copy()
        if self.weighted.isChecked() and name in data.weights:
            weight = data.weights[name]
            values *= weight
            label = f'{name} contribution (weight {data.weights[name]:g})'
            preference = 'higher is better'
        mask = data.valid.copy()
        values[~mask] = np.nan
        finite = values[np.isfinite(values)]
        text = ('No admissible grid cells' if not len(finite) else
                f'{len(finite)} admissible cells · min {finite.min():.4g} · max {finite.max():.4g}')
        self.summary.setText(f'{preference} · {text}\n'
            'Gray cells are masked. These values rank seeds; they do not guarantee a valid placement.')
        self.plot.suppress_rendering = True
        try:
            self.plot.clear()
            for panel in range(2):
                self.plot.subplot(0, panel)
                self.plot.set_background('#252525')
                if panel == 0:
                    x = np.linspace(*data.scene_xlim, data.scene_height.shape[1])
                    y = np.linspace(*data.scene_ylim, data.scene_height.shape[0])
                    raster = data.scene_height
                    title = 'Scene height [m]'
                else:
                    x, y, raster, title = data.x, data.y, values, label
                self._heatmap(x, y, raster, title)
                lo, hi = data.target_bounds
                corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                                    [hi[0], hi[1]], [lo[0], hi[1]]])
                self._outline(corners, '#f4d35e')
                for polygon in data.footprint:
                    self._outline(polygon, 'white')
                self.plot.add_points(np.array([[*data.initial_xy, .05]]), color='magenta',
                    point_size=12, render_points_as_spheres=True)
                self.plot.add_text(title, font_size=10, color='white')
                self.plot.view_xy()
                self.plot.enable_parallel_projection()
                # Equal camera bounds make side-by-side XY comparisons direct.
                self.plot.reset_camera(bounds=(data.scene_xlim[0], data.scene_xlim[1],
                    data.scene_ylim[0], data.scene_ylim[1], 0., .05))
                self.plot.show_bounds(bounds=(data.scene_xlim[0], data.scene_xlim[1],
                    data.scene_ylim[0], data.scene_ylim[1], 0., .05),
                    xtitle='X [m]', ytitle='Y [m]', show_zaxis=False,
                    show_zlabels=False, color='white', font_size=10)
        finally:
            self.plot.suppress_rendering = False
        self.plot.render()

    def _heatmap(self, x, y, values, title):
        data = self.data
        # Sampling grids may have a single row or column.
        lo, hi = data.target_bounds
        dx = x[1] - x[0] if len(x) > 1 else hi[0] - lo[0]
        dy = y[1] - y[0] if len(y) > 1 else hi[1] - lo[1]
        grid = pv.ImageData(dimensions=(len(x) + 1, len(y) + 1, 1),
            spacing=(dx, dy, 1.), origin=(x[0] - dx / 2, y[0] - dy / 2, 0.))
        grid.cell_data['value'] = values.ravel()
        finite = values[np.isfinite(values)]
        limits = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
        if limits[0] == limits[1]:
            limits = (limits[0] - .5, limits[1] + .5)
        self.plot.add_mesh(grid, scalars='value', cmap='viridis', clim=limits,
            nan_color='#555555', lighting=False,
            scalar_bar_args={'title': title, 'fmt': '%.3g', 'n_labels': 3, 'color': 'white',
                             'title_font_size': 11, 'label_font_size': 10})

    def _outline(self, polygon, color):
        points = np.column_stack((np.vstack((polygon, polygon[:1])),
                                  np.full(len(polygon) + 1, .025)))
        self.plot.add_mesh(pv.lines_from_points(points), color=color, line_width=2)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save XY score maps', 'xy-score-maps.png', 'PNG (*.png)')
        if path:
            self.plot.screenshot(path)

    def closeEvent(self, event):
        self.plot.close()
        super().closeEvent(event)
