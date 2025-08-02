# This file has been temporarily modified to remove the torch dependency,
# which was causing installation issues.
# The original code has been commented out and will be restored once the
# torch installation issue is resolved.

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class ForecastScatterView(QWidget):
    def __init__(self, dataframe=None):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        label = QLabel("Forecast view is temporarily disabled due to a torch installation issue.")
        layout.addWidget(label)
