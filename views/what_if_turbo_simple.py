# views/what_if_turbo_simple.py
"""
Simplified What-If Turbo Panel
=============================

Minimal version to avoid layout and import issues.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QPushButton,
)

import pandas as pd
import numpy as np
from views.utils import data_required


class SimpleWhatIfTurboPanel(QWidget):
    """Simplified What-If Turbo Panel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What-If Turbo Analysis")
        self.setMinimumWidth(400)

        # Initialize data
        self.current_scenario = {
            'room_rate': 150.0,
            'occupancy_pct': 75.0,
            'housekeeping_staff': 10,
            'fb_staff': 8,
        }

        self._setup_ui()
        self._update_kpis()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("What-If Turbo Analysis")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        layout.addWidget(title)

        # Input controls
        layout.addWidget(self._create_input_controls())

        # KPI display
        layout.addWidget(self._create_kpi_display())

        # Info section
        info_label = QLabel("Adjust the controls above to see real-time KPI updates.")
        info_label.setStyleSheet("color: #7f8c8d; font-size: 12px; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()

    def _create_input_controls(self) -> QGroupBox:
        """Create input control section."""
        group = QGroupBox("Scenario Inputs")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        layout = QGridLayout(group)

        # Room Rate Slider
        layout.addWidget(QLabel("Room Rate ($):"), 0, 0)
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 500)
        self.rate_slider.setValue(150)
        self.rate_label = QLabel("$150")
        self.rate_label.setMinimumWidth(60)
        self.rate_label.setStyleSheet("font-weight: bold; color: #27ae60;")

        rate_layout = QHBoxLayout()
        rate_layout.addWidget(self.rate_slider)
        rate_layout.addWidget(self.rate_label)
        layout.addLayout(rate_layout, 0, 1)

        # Occupancy Percentage
        layout.addWidget(QLabel("Occupancy (%):"), 1, 0)
        self.occupancy_spin = QDoubleSpinBox()
        self.occupancy_spin.setRange(0.0, 100.0)
        self.occupancy_spin.setValue(75.0)
        self.occupancy_spin.setSuffix("%")
        layout.addWidget(self.occupancy_spin, 1, 1)

        # Housekeeping Staff
        layout.addWidget(QLabel("Housekeeping Staff:"), 2, 0)
        self.housekeeping_spin = QSpinBox()
        self.housekeeping_spin.setRange(1, 50)
        self.housekeeping_spin.setValue(10)
        layout.addWidget(self.housekeeping_spin, 2, 1)

        # F&B Staff
        layout.addWidget(QLabel("F&B Staff:"), 3, 0)
        self.fb_spin = QSpinBox()
        self.fb_spin.setRange(1, 30)
        self.fb_spin.setValue(8)
        layout.addWidget(self.fb_spin, 3, 1)

        # Connect signals
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        self.occupancy_spin.valueChanged.connect(self._update_kpis)
        self.housekeeping_spin.valueChanged.connect(self._update_kpis)
        self.fb_spin.valueChanged.connect(self._update_kpis)

        return group

    def _create_kpi_display(self) -> QGroupBox:
        """Create KPI display section."""
        group = QGroupBox("Key Performance Indicators")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        layout = QGridLayout(group)

        # KPI labels
        self.revpar_label = QLabel("RevPAR: $112.50")
        self.goppar_label = QLabel("GOPPAR: $67.50")
        self.profit_delta_label = QLabel("Profit Δ: $0.00")
        self.occupancy_rooms_label = QLabel("Occupied Rooms: 75")

        # Style KPI labels
        kpi_style = """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border: 3px solid #3498db;
                border-radius: 8px;
                background: #f8f9fa;
                color: #2c3e50;
                margin: 5px;
            }
        """

        for label in [self.revpar_label, self.goppar_label,
                      self.profit_delta_label, self.occupancy_rooms_label]:
            label.setStyleSheet(kpi_style)

        layout.addWidget(self.revpar_label, 0, 0)
        layout.addWidget(self.goppar_label, 0, 1)
        layout.addWidget(self.profit_delta_label, 1, 0)
        layout.addWidget(self.occupancy_rooms_label, 1, 1)

        return group

    def _on_rate_changed(self, value: int):
        """Handle room rate slider change."""
        self.rate_label.setText(f"${value}")
        self._update_kpis()

    def _update_kpis(self):
        """Update KPI calculations and display."""
        try:
            # Get current values
            rate = self.rate_slider.value()
            occupancy_pct = self.occupancy_spin.value()
            housekeeping_staff = self.housekeeping_spin.value()
            fb_staff = self.fb_spin.value()

            # Simple calculations
            occupancy_decimal = occupancy_pct / 100
            revpar = rate * occupancy_decimal

            # Simplified GOPPAR calculation (assume 60% flow-through)
            goppar = revpar * 0.6

            # Staff cost impact (simplified)
            baseline_staff_cost = (10 * 100) + (8 * 80)  # $100/housekeeper, $80/fb
            current_staff_cost = (housekeeping_staff * 100) + (fb_staff * 80)
            staff_cost_delta = current_staff_cost - baseline_staff_cost

            # Profit delta (per day)
            baseline_revpar = 150 * 0.75  # $150 rate, 75% occupancy
            revpar_delta = revpar - baseline_revpar
            profit_delta = (revpar_delta * 0.6) - (staff_cost_delta / 30)  # Assume 30 days

            # Update labels
            self.revpar_label.setText(f"RevPAR: ${revpar:.2f}")
            self.goppar_label.setText(f"GOPPAR: ${goppar:.2f}")

            # Color code profit delta
            delta_color = "#27ae60" if profit_delta >= 0 else "#e74c3c"
            delta_sign = "+" if profit_delta >= 0 else ""
            self.profit_delta_label.setText(f"Profit Δ: {delta_sign}${profit_delta:.2f}")
            self.profit_delta_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 16px;
                    font-weight: bold;
                    padding: 12px;
                    border: 3px solid {delta_color};
                    border-radius: 8px;
                    background: #f8f9fa;
                    color: #2c3e50;
                    margin: 5px;
                }}
            """)

            # Occupied rooms (assume 100 total rooms)
            occupied_rooms = 100 * occupancy_decimal
            self.occupancy_rooms_label.setText(f"Occupied Rooms: {occupied_rooms:.0f}")

        except Exception as e:
            print(f"Error updating KPIs: {e}")


@data_required
def display() -> QWidget:
    """Create and return simplified What-If Turbo panel widget."""
    try:
        return SimpleWhatIfTurboPanel()
    except Exception as e:
        print(f"Error creating Simple What-If Turbo panel: {e}")

        # Return basic fallback widget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        error_label = QLabel(f"Error loading What-If Turbo: {str(e)}")
        error_label.setStyleSheet("""
            QLabel {
                color: #e74c3c;
                font-size: 14pt;
                padding: 20px;
                background-color: #f8f9fa;
                border: 2px solid #e74c3c;
                border-radius: 10px;
            }
        """)
        layout.addWidget(error_label)
        return widget
