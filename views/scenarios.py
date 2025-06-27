from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


def display() -> QWidget:
    """Create and return the scenario planning widget."""
    widget = QWidget()
    layout = QVBoxLayout()

    title = QLabel("Scenario Planning")
    title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00eaff;")

    description = QLabel("Explore different scenarios for hotel management.")
    description.setStyleSheet("font-size: 16px; color: #ffffff;")

    layout.addWidget(title)
    layout.addWidget(description)

    widget.setLayout(layout)
    return widget
