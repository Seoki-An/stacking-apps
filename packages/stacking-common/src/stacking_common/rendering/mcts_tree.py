"""Optional Qt renderer for canonical MCTS tree snapshots."""

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout, QWidget,
)

from stacking_common.visualization.tree import displayed_tree, tree_positions


class TreeCanvas(QGraphicsView):
    def __init__(self, scene, parent):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#20252b"))

    def zoom(self, factor):
        scale = self.transform().m11() * factor
        if .0001 <= scale <= 20:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()


class MCTSTreeView(QWidget):
    """Branch diagram: depth downwards, larger nodes for more visits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._step = None
        self.nodes = ()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.scene = QGraphicsScene(self)
        self.canvas = TreeCanvas(self.scene, self)
        self.details = QLabel("Hover or click a node for details. Wheel to zoom; drag to pan.")
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.scene.selectionChanged.connect(self._selection_changed)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        controls = QHBoxLayout()
        for label, slot in (("Fit tree", self.fit_tree), ("Zoom +", lambda: self.canvas.zoom(1.3)),
                            ("Zoom −", lambda: self.canvas.zoom(1 / 1.3))):
            button = QPushButton(label)
            button.clicked.connect(slot)
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.details)

    def set_step(self, step):
        if step is self._step:
            return
        self._step = step
        self.scene.clear()
        self.nodes = () if step is None else displayed_tree(step)
        self.details.setText("Hover or click a node for details. Wheel to zoom; drag to pan.")
        if step is None:
            self.summary.setText("No search step loaded.")
            return
        count = len(self.nodes)
        depth = max(node.depth for node in self.nodes)
        scope = ("Retained search tree" if step.tree_nodes else
                 "Partial tree — this file saved root candidates only; deeper branches are unavailable")
        self.summary.setText(
            f"Step {step.step} · {scope}\n"
            f"{count} nodes · depth {depth} · Green: evaluated · Red: failed · "
            "Blue: pending · Gray: unknown · Gold outline: selected root action. "
            "Node size: visits. Pre-node generation rejections are not branches.")
        positions = tree_positions(self.nodes)
        points = {key: (x * 42., y * 100.) for key, (x, y) in positions.items()}
        edges = QPainterPath()
        for node in self.nodes[1:]:
            edges.moveTo(*points[node.parent_id])
            edges.lineTo(*points[node.node_id])
        edge_pen = QPen(QColor("#73808d"))
        edge_pen.setCosmetic(True)
        self.scene.addPath(edges, edge_pen)
        colors = {"evaluated": "#60bd68", "failed": "#f17c67",
                  "pending": "#5da5da", "unknown": "#999999"}
        max_visits = max((node.visits or 0 for node in self.nodes), default=0)
        for node in self.nodes:
            x, y = points[node.node_id]
            radius = 5 + (9 * math.log1p(node.visits or 0) / math.log1p(max_visits)
                          if max_visits else 0)
            pen = QPen(QColor("#ffd166" if node.selected else "#c4ccd4"))
            pen.setWidthF(3 if node.selected else .7)
            item = self.scene.addEllipse(x-radius, y-radius, radius*2, radius*2,
                                         pen, QColor(colors[node.outcome]))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            item.setData(0, node.node_id)
            item.setToolTip(self._node_text(node))
            item.setZValue(1)
        root_label = self.scene.addSimpleText("Root")
        root_label.setBrush(QColor("#ffffff"))
        root_label.setPos(points[0][0] - 14, -38)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-24, -24, 24, 24))
        self.fit_tree()

    @staticmethod
    def _node_text(node):
        def number(value):
            return "not recorded" if value is None else f"{value:.4g}"
        return (
            f"Node {node.node_id} · parent {node.parent_id} · depth {node.depth}\n"
            f"Object: {node.object_id if node.object_id is not None else 'root'} · {node.outcome}"
            f"{' · selected root action' if node.selected else ''}\n"
            f"Visits: {number(node.visits)} · mean value: {number(node.mean_value)} · "
            f"reward: {number(node.reward)} · prior: {number(node.prior)}"
            f"{chr(10) + node.rejection_reason if node.rejection_reason else ''}")

    def _selection_changed(self):
        selected = self.scene.selectedItems()
        if selected:
            self.details.setText(self._node_text(self.nodes[selected[0].data(0)]))

    def fit_tree(self):
        if self.nodes:
            self.canvas.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_tree()
