"""Shared excavator GUI theme."""

DARK_STYLESHEET = """
QWidget {
    background-color: #171b21;
    color: #e1e6ec;
    selection-background-color: #355b82;
    selection-color: #ffffff;
}
QMainWindow, QSplitter, QGroupBox {
    background-color: #171b21;
}
QGroupBox {
    border: 1px solid #343c47;
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 9px;
    padding: 0 4px;
}
QPushButton {
    background-color: #272e37;
    border: 1px solid #414b58;
    border-radius: 4px;
    padding: 5px 8px;
}
QPushButton:hover {
    background-color: #323b47;
    border-color: #5b7188;
}
QPushButton:pressed {
    background-color: #20262e;
}
QPushButton:disabled {
    background-color: #1d2229;
    border-color: #2a3038;
    color: #68717c;
}
QTreeWidget, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #11151a;
    border: 1px solid #343c47;
    border-radius: 3px;
}
QHeaderView::section {
    background-color: #252b34;
    color: #d9dfe6;
    border: 0;
    border-right: 1px solid #343c47;
    border-bottom: 1px solid #343c47;
    padding: 4px;
}
QProgressBar {
    background-color: #11151a;
    border: 1px solid #343c47;
    border-radius: 4px;
    color: #e1e6ec;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #3d729e;
    border-radius: 3px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
}
QCheckBox::indicator:unchecked {
    background-color: #11151a;
    border: 1px solid #4a5563;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #3d729e;
    border: 1px solid #6495bd;
    border-radius: 3px;
}
QSplitter::handle {
    background-color: #2d343e;
}
QSplitter::handle:hover {
    background-color: #506071;
}
QToolTip {
    background-color: #252b34;
    color: #f2f5f8;
    border: 1px solid #566373;
}
"""
