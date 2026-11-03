"""Lightweight Qt previews of canonical target geometry, without an OpenGL view."""

import numpy as np
from scipy.spatial import ConvexHull

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from stacking_common.domain import TargetGeometry
from stacking_common.visualization.contracts import CompositeTargetVolume, TargetVolume


def preview_meshes(target):
    """Keep posed components separate so gaps and concavities remain visible."""
    if isinstance(target, CompositeTargetVolume):
        return tuple((item.world_geometry.points + target.structure_offset,
                      item.world_geometry.faces) for item in target.components)
    if not isinstance(target, TargetVolume):
        raise TypeError("target preview requires canonical target display geometry")
    geometry = TargetGeometry(target.width, target.length, target.height,
                              target.origin + target.structure_offset,
                              taper=target.taper, support_overlap=0.)
    return ((geometry.points, geometry.faces),)


class TargetPreview(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Target preview", parent)
        self.target = None
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.canvas = _TargetCanvas()
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.details)
        self.setMinimumWidth(290)
        self.setMaximumWidth(380)

    def set_target(self, target, name):
        meshes = preview_meshes(target)
        bounds = np.ptp(np.vstack([points for points, _ in meshes]), axis=0)
        self.target = target
        self.title.setText(name)
        self.details.setText(
            "Extent X × Y × Z: " + " × ".join(f"{value:.3g}" for value in bounds) + " m\n"
            + (f"{len(meshes)} components" if len(meshes) > 1 else "1 component"))
        self.canvas.meshes = meshes
        self.canvas.update()

    def show_error(self, message):
        self.target = None
        self.title.setText("Preview unavailable")
        self.details.setText(message)
        self.canvas.meshes = ()
        self.canvas.update()


class _TargetCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.meshes = ()
        self.setMinimumSize(260, 290)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f2f4f7"))
        if not self.meshes:
            painter.end()
            return
        eye = np.array([1.6, -2.5, 1.7])
        eye /= np.linalg.norm(eye)
        right = np.cross([0., 0., 1.], eye)
        right /= np.linalg.norm(right)
        up = np.cross(eye, right)
        self._draw(painter, QRectF(12, 24, self.width()-24, self.height()*.61-38),
                   np.array([right, up]), eye)
        self._draw(painter, QRectF(12, self.height()*.66, self.width()-24, self.height()*.34-26),
                   np.array([[1., 0., 0.], [0., 1., 0.]]), None)
        painter.setPen(QColor("#252525"))
        painter.drawText(12, 17, "3D")
        painter.drawText(12, int(self.height()*.66)-7, "Top (XY)")
        painter.end()

    def _draw(self, painter, rect, basis, eye):
        cloud = np.vstack([points for points, _ in self.meshes])
        projected = cloud @ basis.T
        center = (projected.min(axis=0) + projected.max(axis=0)) / 2
        span = np.maximum(np.ptp(projected, axis=0), 1e-12)
        scale = min(rect.width()/span[0], rect.height()/span[1]) * .86

        def polygon(points):
            xy = (points @ basis.T - center) * scale
            return QPolygonF([QPointF(rect.center().x()+x, rect.center().y()-y) for x, y in xy])

        if eye is None:
            # Project each piece independently; a hull over the union fills L notches.
            for points, _ in self.meshes:
                outline = points[ConvexHull(points[:, :2]).vertices]
                painter.setPen(QPen(QColor("#355c80"), 1.))
                painter.setBrush(QColor("#9bc4e5"))
                painter.drawPolygon(polygon(outline))
        else:
            faces = []
            for points, indices in self.meshes:
                for face in indices:
                    triangle = points[face]
                    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                    norm = np.linalg.norm(normal)
                    if norm <= 1e-15:
                        continue
                    if np.dot(normal, triangle.mean(axis=0)-points.mean(axis=0)) < 0:
                        normal = -normal
                    shade = .65 + .35 * max(float(np.dot(normal/norm, eye)), 0.)
                    faces.append((float(triangle.mean(axis=0) @ eye), triangle, shade))
            for _, triangle, shade in sorted(faces, key=lambda item: item[0]):
                color = QColor.fromRgbF(.48*shade, .69*shade, .86*shade)
                painter.setPen(QPen(color, .6))
                painter.setBrush(color)
                painter.drawPolygon(polygon(triangle))
        # Fixed axis markers clarify the view without obscuring small targets.
        origin = QPointF(rect.left()+22, rect.bottom()-14)
        painter.setPen(QPen(QColor("#355c80"), 1.2))
        for index, label in enumerate("XY" if eye is None else "XYZ"):
            x, y = basis[:, index] * 20
            end = origin + QPointF(float(x), float(-y))
            painter.drawLine(origin, end)
            painter.drawText(end + QPointF(2, -2), label)
