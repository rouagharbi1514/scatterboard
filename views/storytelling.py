from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


def display() -> QWidget:
    """Create a QWidget for storytelling with data."""
    widget = QWidget()
    layout = QVBoxLayout()

    title = QLabel("Storytelling with Data")
    title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00eaff;")
    layout.addWidget(title)

    description = QLabel(
        "This section will provide insights and narratives based on the KPI data."
    )
    description.setStyleSheet("font-size: 16px; color: #ffffff;")
    layout.addWidget(description)

    # Additional storytelling elements can be added here

    widget.setLayout(layout)
    return widget
