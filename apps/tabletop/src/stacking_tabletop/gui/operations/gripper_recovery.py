"""Operator-confirmed, bounded recovery outside an execution worker."""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox

from stacking_tabletop.execution.gripper_recovery import recover_gripper


def confirm_gripper_recovery(parent):
    return QMessageBox.question(
        parent, 'Recover gripper',
        'Confirm the robot is stopped and any held stone is supported or removed. '
        'Reboot interrupts gripping torque. Recovery leaves torque off and does '
        'not resume execution. Continue?',
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    ) == QMessageBox.StandardButton.Yes


def connect_recovery():
    from stacking_tabletop.ros.hardware import TabletopRosTransport
    return TabletopRosTransport(node_name='tabletop_gripper_recovery')


class GripperRecoveryWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent=None, *, transport_factory=connect_recovery):
        super().__init__(parent)
        self.transport_factory = transport_factory

    def run(self):
        try:
            self.completed.emit(recover_gripper(self.transport_factory))
        except Exception as error:
            self.failed.emit(str(error))
