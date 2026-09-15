"""Interactive cluster-to-stone ordering for fixed-Femto field scans."""

from __future__ import annotations

import numpy as np
import open3d as o3d


TEXT_LABEL_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
LABEL_SCALE_M = 0.0003
LABEL_HEIGHT_M = 0.015
COORDINATE_FRAME_SIZE_M = 0.02


def validate_cluster_assignments(
    selected_ids: list[int | None],
    removed_clusters: set[int],
) -> dict[int, int]:
    """Return kept cluster-to-stone assignments after validating uniqueness."""
    assignments = {
        cluster_index: int(stone_id)
        for cluster_index, stone_id in enumerate(selected_ids)
        if cluster_index not in removed_clusters and stone_id is not None
    }
    if len(set(assignments.values())) != len(assignments):
        raise ValueError("Each kept cluster must have a unique stone ID")
    return assignments


def _text_mesh(text: str, position: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.t.geometry.TriangleMesh.create_text(text)
    mesh = mesh.scale(LABEL_SCALE_M, center=[0.0, 0.0, 0.0]).to_legacy()
    mesh.paint_uniform_color([0.05, 0.05, 0.05])
    mesh.rotate(TEXT_LABEL_ROTATION, center=[0.0, 0.0, 0.0])
    triangles = np.asarray(mesh.triangles).copy()
    triangles[:, [1, 2]] = triangles[:, [2, 1]]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.translate(position)
    return mesh


def assign_cluster_ids_gui(
    clusters: list[o3d.geometry.PointCloud],
    stone_ids: list[int],
    suggested_assignments: dict[int, int],
) -> dict[int, int]:
    """Review Femto clusters and return ``{cluster_index: stone_id}``."""
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QHeaderView,
            QLabel,
            QMessageBox,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )
    except ImportError as error:
        raise RuntimeError(
            "PySide6 is required for stone ordering; use "
            "--automatic-ordering for headless operation"
        ) from error

    app = QApplication.instance() or QApplication([])
    preview = o3d.visualization.Visualizer()
    if not preview.create_window(
        window_name="Femto clusters - assign IDs in the ordering GUI",
        width=1100,
        height=720,
    ):
        raise RuntimeError(
            "Could not open the cluster preview; use --automatic-ordering "
            "when no display is available"
        )

    palette = np.array(
        [
            [0.90, 0.20, 0.20],
            [0.20, 0.55, 0.95],
            [0.20, 0.75, 0.35],
            [0.95, 0.60, 0.15],
            [0.65, 0.30, 0.90],
            [0.10, 0.75, 0.75],
        ]
    )
    preview_clusters = []
    label_positions = []
    labels = []
    for index, cluster in enumerate(clusters):
        colored = o3d.geometry.PointCloud(cluster)
        colored.paint_uniform_color(palette[index % len(palette)])
        preview_clusters.append(colored)
        preview.add_geometry(colored)

        position = np.asarray(cluster.points).mean(axis=0)
        position = position + [0.0, 0.0, LABEL_HEIGHT_M]
        label_positions.append(position)
        suggested = suggested_assignments.get(index)
        text = str(index) if suggested is None else f"{index}:{suggested}"
        label = _text_mesh(text, position)
        labels.append(label)
        preview.add_geometry(label, reset_bounding_box=False)

    preview.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=COORDINATE_FRAME_SIZE_M
        )
    )
    render_options = preview.get_render_option()
    render_options.background_color = np.asarray([0.92, 0.92, 0.92])
    render_options.point_size = 3.0

    dialog = QDialog()
    dialog.setWindowTitle("Assign Stone IDs to Femto Clusters")
    dialog.resize(780, 520)
    layout = QVBoxLayout(dialog)
    layout.addWidget(
        QLabel(
            "Review the numbered clusters, assign one known stone ID to each, "
            "and mark unrelated clusters as Skip."
        )
    )

    table = QTableWidget(len(clusters), 7)
    table.setHorizontalHeaderLabels(
        ["Cluster", "Points", "X [m]", "Y [m]", "Z [m]", "Stone ID", "Skip"]
    )
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    layout.addWidget(table)

    combos = []
    skip_checks = []
    for row, cluster in enumerate(clusters):
        points = np.asarray(cluster.points)
        center = points.mean(axis=0)
        values = [
            str(row),
            str(len(points)),
            f"{center[0]:.4f}",
            f"{center[1]:.4f}",
            f"{center[2]:.4f}",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, column, item)

        combo = QComboBox()
        combo.addItem("-", None)
        for stone_id in stone_ids:
            combo.addItem(str(stone_id), int(stone_id))
        suggested = suggested_assignments.get(row)
        suggested_index = combo.findData(suggested)
        if suggested_index >= 0:
            combo.setCurrentIndex(suggested_index)
        table.setCellWidget(row, 5, combo)
        combos.append(combo)

        skip = QCheckBox()
        skip.setStyleSheet("margin-left: auto; margin-right: auto;")
        table.setCellWidget(row, 6, skip)
        skip_checks.append(skip)

    def update_label(row: int):
        preview.remove_geometry(labels[row], reset_bounding_box=False)
        stone_id = combos[row].currentData()
        text = str(row) if stone_id is None else f"{row}:{stone_id}"
        labels[row] = _text_mesh(text, label_positions[row])
        if not skip_checks[row].isChecked():
            preview.add_geometry(labels[row], reset_bounding_box=False)

    for row, combo in enumerate(combos):
        combo.currentIndexChanged.connect(
            lambda _index, row=row: update_label(row)
        )

    def toggle_cluster(row: int, state: int):
        if bool(state):
            preview.remove_geometry(
                preview_clusters[row], reset_bounding_box=False
            )
            preview.remove_geometry(labels[row], reset_bounding_box=False)
        else:
            preview.add_geometry(preview_clusters[row], reset_bounding_box=False)
            preview.add_geometry(labels[row], reset_bounding_box=False)

    for row, skip in enumerate(skip_checks):
        skip.stateChanged.connect(
            lambda state, row=row: toggle_cluster(row, state)
        )

    timer = QTimer()
    timer.timeout.connect(lambda: (preview.poll_events(), preview.update_renderer()))
    timer.start(30)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)

    def accept_if_valid():
        selected = [combo.currentData() for combo in combos]
        removed = {
            index for index, checkbox in enumerate(skip_checks)
            if checkbox.isChecked()
        }
        try:
            validate_cluster_assignments(selected, removed)
        except ValueError as error:
            QMessageBox.warning(dialog, "Invalid stone ordering", str(error))
            return
        dialog.accept()

    buttons.accepted.connect(accept_if_valid)
    buttons.rejected.connect(dialog.reject)

    try:
        if dialog.exec() != QDialog.Accepted:
            raise RuntimeError("Stone ordering cancelled")
        selected = [combo.currentData() for combo in combos]
        removed = {
            index for index, checkbox in enumerate(skip_checks)
            if checkbox.isChecked()
        }
        return validate_cluster_assignments(selected, removed)
    finally:
        timer.stop()
        preview.destroy_window()
