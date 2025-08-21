# flake8: noqa
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
    QFrame,
    QGraphicsDropShadowEffect,
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

    # ──────────────────────────────────────────────────────────
    # UI
    def _apply_card_shadow(self, widget: QWidget, blur: int = 18, alpha: int = 26, y: int = 6):
        """Soft shadow for cards (purely visual)."""
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, y)
        shadow.setColor(Qt.black)
        # Reduce alpha via stylesheet-friendly color (handled by palette)
        widget.setGraphicsEffect(shadow)

    def _setup_ui(self):
        """Setup the user interface."""
        # Root style (design-only)
        self.setObjectName("whatIfRoot")
        self.setStyleSheet("""
            /* Root background: soft, modern */
            #whatIfRoot {
                background:
                    radial-gradient(420px 280px at 10% 8%, #EEF4FF 0%, transparent 60%),
                    radial-gradient(520px 340px at 90% 92%, #F6FAFF 0%, transparent 60%),
                    qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #F9FBFE, stop:1 #F2F6FF);
                font-family: "Inter","Segoe UI", Arial, sans-serif;
                color: #0F172A;
            }

            /* Title */
            #titleLabel {
                font-size: 18pt;
                font-weight: 800;
                color: #0F172A;
                letter-spacing: .2px;
            }

            /* Cards (GroupBox relooked as cards) */
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E6EAF1;
                border-radius: 14px;
                margin-top: 18px;   /* space for title */
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                top: 6px;
                padding: 2px 8px;
                color: #1D4ED8;
                background: #F3F6FF;
                border: 1px solid #E0E7FF;
                border-radius: 999px;
                font-weight: 700;
                letter-spacing: .2px;
            }

            /* Labels */
            QLabel {
                color: #0F172A;
            }

            /* Slider – slim & accessible */
            QSlider::groove:horizontal {
                border: 0;
                height: 6px;
                background: #E8EEF7;
                border-radius: 3px;
                margin: 8px 0;
            }
            QSlider::handle:horizontal {
                background: #2563EB;
                border: 2px solid #FFFFFF;
                width: 18px; height: 18px;
                margin: -9px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal { background: #9CC2FF; border-radius: 3px; }

            /* SpinBoxes – clean */
            QDoubleSpinBox, QSpinBox {
                background: #FFFFFF;
                border: 1px solid #D8E3F5;
                border-radius: 10px;
                padding: 6px 10px;
                min-height: 32px;
                font-weight: 600;
            }
            QDoubleSpinBox:hover, QSpinBox:hover { border-color: #C9D7F0; }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border: 2px solid #2563EB;
                padding: 5px 9px; /* maintain size on focus */
            }

            /* KPI tiles – fine & modern */
            .kpi {
                font-size: 14px;
                font-weight: 700;
                padding: 10px 12px;
                border: 1px solid #E6EAF1;
                border-radius: 12px;
                background:
                    qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #FFFFFF, stop:1 #FAFCFF);
                color: #0F172A;
                margin: 4px;
            }
            .kpiGood { border-color: #9EE6B0; background: #F6FFF8; }
            .kpiBad  { border-color: #FFC1C1; background: #FFF7F7; }

            /* Info hint */
            #hint {
                color: #475569;
                font-size: 11pt;
                background: #F6FAFF;
                border: 1px solid #E6EAF1;
                border-radius: 12px;
                padding: 10px 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title + accent
        title_wrap = QVBoxLayout()
        title = QLabel("What-If Turbo Analysis")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_wrap.addWidget(title)

        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2563EB, stop:1 #5B7CFF);
                border-radius: 2px;
            }""")
        title_wrap.addWidget(accent)
        layout.addLayout(title_wrap)

        # Input controls
        inputs = self._create_input_controls()
        self._apply_card_shadow(inputs)
        layout.addWidget(inputs)

        # KPI display
        kpis = self._create_kpi_display()
        self._apply_card_shadow(kpis)
        layout.addWidget(kpis)

        # Info section
        info_label = QLabel("Adjust the controls above to see real-time KPI updates.")
        info_label.setObjectName("hint")
        layout.addWidget(info_label)

        layout.addStretch()

    # ──────────────────────────────────────────────────────────
    # Sections
    def _create_input_controls(self) -> QGroupBox:
        """Create input control section."""
        group = QGroupBox("Scenario Inputs")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(12, 10, 12, 12)

        # Row 0: Room Rate
        rate_row = QHBoxLayout()
        rate_row.setSpacing(10)
        rate_row.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(QLabel("Room Rate ($):"), 0, 0)
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 500)
        self.rate_slider.setValue(150)
        self.rate_label = QLabel("$150")
        self.rate_label.setMinimumWidth(60)
        self.rate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rate_label.setStyleSheet("font-weight: 800; color: #1D4ED8;")

        rate_row.addWidget(self.rate_slider)
        rate_row.addWidget(self.rate_label)
        grid.addLayout(rate_row, 0, 1)

        # Row 1: Occupancy
        grid.addWidget(QLabel("Occupancy (%):"), 1, 0)
        self.occupancy_spin = QDoubleSpinBox()
        self.occupancy_spin.setRange(0.0, 100.0)
        self.occupancy_spin.setValue(75.0)
        self.occupancy_spin.setSuffix("%")
        grid.addWidget(self.occupancy_spin, 1, 1)

        # Row 2: Housekeeping
        grid.addWidget(QLabel("Housekeeping Staff:"), 2, 0)
        self.housekeeping_spin = QSpinBox()
        self.housekeeping_spin.setRange(1, 50)
        self.housekeeping_spin.setValue(10)
        grid.addWidget(self.housekeeping_spin, 2, 1)

        # Row 3: F&B
        grid.addWidget(QLabel("F&B Staff:"), 3, 0)
        self.fb_spin = QSpinBox()
        self.fb_spin.setRange(1, 30)
        self.fb_spin.setValue(8)
        grid.addWidget(self.fb_spin, 3, 1)

        # Signals (unchanged logic)
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        self.occupancy_spin.valueChanged.connect(self._update_kpis)
        self.housekeeping_spin.valueChanged.connect(self._update_kpis)
        self.fb_spin.valueChanged.connect(self._update_kpis)

        return group

    def _create_kpi_display(self) -> QGroupBox:
        """Create KPI display section."""
        group = QGroupBox("Key Performance Indicators")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(12, 10, 12, 12)

        # KPI labels (same variables, new style classes)
        self.revpar_label = QLabel("RevPAR: $112.50")
        self.revpar_label.setProperty("class", "kpi")
        self.goppar_label = QLabel("GOPPAR: $67.50")
        self.goppar_label.setProperty("class", "kpi")
        self.profit_delta_label = QLabel("Profit Δ: $0.00")
        self.profit_delta_label.setProperty("class", "kpi")
        self.occupancy_rooms_label = QLabel("Occupied Rooms: 75")
        self.occupancy_rooms_label.setProperty("class", "kpi")

        grid.addWidget(self.revpar_label, 0, 0)
        grid.addWidget(self.goppar_label, 0, 1)
        grid.addWidget(self.profit_delta_label, 1, 0)
        grid.addWidget(self.occupancy_rooms_label, 1, 1)

        return group

    # ──────────────────────────────────────────────────────────
    # Logic (unchanged)
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

            # Update labels (same variables)
            self.revpar_label.setText(f"RevPAR: ${revpar:.2f}")
            self.goppar_label.setText(f"GOPPAR: ${goppar:.2f}")

            # Profit delta chip with good/bad accent (design-only)
            delta_color_class = "kpiGood" if profit_delta >= 0 else "kpiBad"
            delta_sign = "+" if profit_delta >= 0 else ""
            self.profit_delta_label.setText(f"Profit Δ: {delta_sign}${profit_delta:.2f}")
            # keep base kpi style + add good/bad modifier
            self.profit_delta_label.setStyleSheet((
                "QLabel {"
                "  font-size: 14px; font-weight: 700; padding: 10px 12px;"
                "  border: 1px solid %s; border-radius: 12px;"
                "  background: %s; color: #0F172A; margin: 4px;"
                "}"
            ) % (("#9EE6B0", "#F6FFF8") if profit_delta >= 0 else ("#FFC1C1", "#FFF7F7")))

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
