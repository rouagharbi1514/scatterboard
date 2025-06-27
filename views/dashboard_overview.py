# views/dashboard_overview.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from views.utils import data_required, get_df

@data_required
def display():
    """Dashboard overview display function"""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    
    # Add a simple label for now
    label = QLabel("Dashboard Overview")
    label.setStyleSheet("font-size: 24pt; font-weight: bold;")
    layout.addWidget(label)
    
    return widget