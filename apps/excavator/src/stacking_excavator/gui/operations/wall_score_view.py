"""On-demand inspection of two-sided wall proposals and surface completion."""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout


class WallScoreDialog(QDialog):
    def __init__(self, data, *, step, candidate, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Two-sided wall · step {step} · object {candidate.object_id} · #{candidate.rank}')
        self.resize(1200, 700)
        layout = QVBoxLayout(self)
        description = QLabel('Recorded initial orientation, reconstructed pre-search scene. '
            'Left: eligible XY seeds colored by proposal score; magenta: recorded seed. '
            'Middle/right: exterior surface completion (0–1), with lower gaps weighted more in planning. '
            'These maps describe surface fit; they do not certify structural bonding.')
        description.setWordWrap(True)
        layout.addWidget(description)
        self.plot = QtInteractor(self, shape=(1, 3))
        layout.addWidget(self.plot.interactor)
        save = QPushButton('Save PNG…')
        save.clicked.connect(self._save)
        layout.addWidget(save)
        for panel in range(3):
            self.plot.subplot(0, panel)
            self.plot.set_background('#252525')
            if panel == 0:
                lower, upper = data.target_bounds
                corners = np.array([[lower[0], lower[1], 0.], [upper[0], lower[1], 0.],
                                    [upper[0], upper[1], 0.], [lower[0], upper[1], 0.],
                                    [lower[0], lower[1], 0.]])
                self.plot.add_mesh(pv.lines_from_points(corners), color='#f4d35e')
                if len(data.proposal_xy):
                    points = pv.PolyData(np.column_stack((data.proposal_xy, np.zeros(len(data.proposal_xy)))))
                    points['score'] = data.proposal_scores
                    self.plot.add_mesh(points, scalars='score', cmap='viridis', point_size=10,
                                       render_points_as_spheres=True, scalar_bar_args={'title': 'Proposal score'})
                self.plot.add_points(np.array([[*data.initial_xy, .01]]), color='magenta', point_size=14)
                title, horizontal, vertical = 'XY proposals', 'X [m]', 'Y [m]'
            else:
                dy, dz = data.y[1] - data.y[0], data.z[1] - data.z[0]
                grid = pv.ImageData(dimensions=(len(data.y) + 1, len(data.z) + 1, 1),
                    spacing=(dy, dz, 1.), origin=(data.y[0] - dy / 2, data.z[0] - dz / 2, 0.))
                grid.cell_data['completion'] = np.where(data.valid, data.quality[panel - 1], np.nan).ravel()
                self.plot.add_mesh(grid, scalars='completion', clim=(0., 1.), cmap='viridis',
                    nan_color='#555555', lighting=False, scalar_bar_args={'title': f'Side {panel} completion'})
                title, horizontal, vertical = ('−X exterior' if panel == 1 else '+X exterior'), 'Y [m]', 'Z [m]'
            self.plot.add_text(title, font_size=10, color='white')
            self.plot.view_xy()
            self.plot.enable_parallel_projection()
            self.plot.show_bounds(xtitle=horizontal, ytitle=vertical, show_zaxis=False,
                                  show_zlabels=False, color='white')

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save wall maps', 'wall-maps.png', 'PNG (*.png)')
        if path:
            self.plot.screenshot(path)

    def closeEvent(self, event):
        self.plot.close()
        super().closeEvent(event)
