# views/what_if.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QCheckBox, QGroupBox, QPushButton, QFrame, QSizePolicy,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPalette

import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

from views.utils import create_plotly_widget, data_required, format_currency

# Matrix cache
ELASTICITY_MATRIX = None
BASELINE_DATA = None
ROOM_TYPES = []


def _collapsible(text: str) -> QWidget:
    """Create a collapsible explanation panel.

    Args:
        text: The explanation text to display

    Returns:
        A widget containing a toggle button and collapsible text panel
    """
    container = QWidget()
    container.setStyleSheet("""
        QWidget {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin: 5px 0;
        }
    """)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(8)

    # Toggle button
    toggle_btn = QPushButton("📋 Show explanation")
    toggle_btn.setStyleSheet("""
        QPushButton {
            background-color: #27ae60;
            color: white;
            padding: 8px 15px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11pt;
            border: none;
            text-align: left;
        }
        QPushButton:hover {
            background-color: #229954;
        }
        QPushButton:pressed {
            background-color: #1e8449;
        }
    """)

    # Explanation label with improved styling
    explanation = QLabel(text)
    explanation.setWordWrap(True)
    explanation.setStyleSheet("""
        QLabel {
            background-color: rgba(39, 174, 96, 0.1);
            color: #2c3e50;
            padding: 15px;
            border-radius: 6px;
            font-size: 11pt;
            line-height: 1.5;
            border: 1px solid rgba(39, 174, 96, 0.3);
        }
    """)
    explanation.setVisible(False)

    # Add widgets to layout
    layout.addWidget(toggle_btn)
    layout.addWidget(explanation)

    # Connect toggle button
    def toggle_explanation():
        is_visible = explanation.isVisible()
        explanation.setVisible(not is_visible)
        toggle_btn.setText("📋 Hide explanation" if not is_visible else "📋 Show explanation")

    toggle_btn.clicked.connect(toggle_explanation)

    return container


class WhatIfPanel(QWidget):
    """Panel for What-If scenario simulations"""
    fallback_needed = Signal(dict)  # Signal to trigger server fallback

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set a modern background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

        # Main layout for the widget
        main_wrapper_layout = QVBoxLayout(self)
        main_wrapper_layout.setContentsMargins(5, 5, 5, 5)

        # Create scroll area with enhanced styling
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show for clarity
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                background-color: #f9f9f9;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #f0f2f5;
                width: 14px;
                border-radius: 7px;
                border: 1px solid #ddd;
            }
            QScrollBar::handle:vertical {
                background-color: #3498db;
                border-radius: 6px;
                min-height: 25px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #2980b9;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #1f4e79;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: transparent;
            }
        """)

        # Create content widget for the scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Main content layout
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Title - compact and modern
        title = QLabel("What-If Turbo Simulation")
        title.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #2c3e50;
            padding: 6px 0;
            border-bottom: 2px solid #3498db;
            margin-bottom: 8px;
        """)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Top section: Controls and KPIs - improved layout
        top_section_layout = QHBoxLayout()
        top_section_layout.setSpacing(12)

        # Controls area - more compact
        controls_group = QGroupBox("Scenario Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                color: #2980b9;
                border: 2px solid #aaddff;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 6px;
                background-color: #eaf6ff;
                border-radius: 4px;
            }
        """)
        controls_group.setMaximumWidth(380)  # Slightly smaller width
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(8)

        # Room rate sliders - more compact
        rate_group = QGroupBox("Room Rate Adjustments (SAR)")
        rate_group.setStyleSheet("""
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #3498db;
                border: 1px solid #cceeff;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
                background-color: #f8fcff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        rate_layout = QVBoxLayout(rate_group)
        rate_layout.setSpacing(6)

        self.room_sliders = {}
        for room_type in ["Standard", "Deluxe", "Suite", "Presidential"]:
            slider_layout = QHBoxLayout()
            label = QLabel(f"{room_type}:")
            label.setStyleSheet("font-size: 11pt; font-weight: 500; min-width: 90px; color: #333;")
            slider_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-50, 50)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.setFixedWidth(200) # Slightly wider slider
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: 1px solid #c0c0c0;
                    height: 8px;
                    background: #e0e0e0;
                    margin: 2px 0;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #3498db;
                    border: 1px solid #2980b9;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
                QSlider::sub-page:horizontal {
                    background: #3498db;
                    border-radius: 4px;
                }
            """)
            slider.valueChanged.connect(self._on_control_changed)

            value_label = QLabel("0 SAR")
            value_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 50px; color: #2c3e50;")

            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)

            self.room_sliders[room_type] = (slider, value_label)
            rate_layout.addLayout(slider_layout)

        controls_layout.addWidget(rate_group)

        # Occupancy slider
        occ_group = QGroupBox("Target Occupancy (%)")
        occ_group.setStyleSheet("""
            QGroupBox {
                font-size: 12pt;
                font-weight: bold;
                color: #3498db;
                border: 1px solid #cceeff;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px;
                background-color: #f8fcff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
        """)
        occ_layout = QHBoxLayout(occ_group)
        occ_layout.setSpacing(10)

        self.occ_slider = QSlider(Qt.Horizontal)
        self.occ_slider.setRange(50, 100)
        self.occ_slider.setValue(80)
        self.occ_slider.setTickPosition(QSlider.TicksBelow)
        self.occ_slider.setTickInterval(5)
        self.occ_slider.setFixedWidth(200)
        self.occ_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #c0c0c0;
                height: 8px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
        """)
        self.occ_slider.valueChanged.connect(self._on_control_changed)

        self.occ_label = QLabel("80%")
        self.occ_label.setStyleSheet("font-size: 11pt; font-weight: bold; min-width: 50px; color: #2c3e50;")

        occ_layout.addWidget(self.occ_slider)
        occ_layout.addWidget(self.occ_label)

        controls_layout.addWidget(occ_group)

        # Staffing controls - more compact
        staff_group = QGroupBox("Staffing Levels (FTE)")
        staff_group.setStyleSheet("""
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #3498db;
                border: 1px solid #cceeff;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
                background-color: #f8fcff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        staff_layout = QVBoxLayout(staff_group)
        staff_layout.setSpacing(6)

        self.staff_spinboxes = {}
        for dept in ["Housekeeping", "F&B"]:
            staff_row = QHBoxLayout()
            label = QLabel(f"{dept}:")
            label.setStyleSheet("font-size: 11pt; font-weight: 500; min-width: 90px; color: #333;")
            staff_row.addWidget(label)

            spinbox = QSpinBox()
            spinbox.setRange(-5, 5)
            spinbox.setValue(0)
            spinbox.setPrefix("+" if spinbox.value() >= 0 else "")
            spinbox.setSuffix(" FTE")
            spinbox.setStyleSheet("""
                QSpinBox {
                    font-size: 11pt;
                    font-weight: bold;
                    min-width: 70px;
                    height: 28px;
                    border: 1px solid #cceeff;
                    border-radius: 5px;
                    padding-left: 5px;
                    background-color: #ffffff;
                    color: #2c3e50;
                }
                QSpinBox::up-button, QSpinBox::down-button {
                    width: 20px;
                    border: 1px solid #cceeff;
                    border-radius: 3px;
                    background-color: #eaf6ff;
                }
                QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                    background-color: #d8edff;
                }
            """)
            spinbox.valueChanged.connect(lambda v, sb=spinbox: sb.setPrefix("+" if v >= 0 else ""))
            spinbox.valueChanged.connect(self._on_control_changed)

            staff_row.addWidget(spinbox)
            self.staff_spinboxes[dept.lower()] = spinbox
            staff_layout.addLayout(staff_row)

        controls_layout.addWidget(staff_group)

        # Promo checkboxes - more compact
        promo_group = QGroupBox("Promotional Bundles")
        promo_group.setStyleSheet("""
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #3498db;
                border: 1px solid #cceeff;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
                background-color: #f8fcff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        promo_layout = QVBoxLayout(promo_group)
        promo_layout.setSpacing(4)

        self.promo_checkboxes = {}
        for promo_id, promo_name in [
            ("spa_discount", "10% Spa Discount"),
            ("breakfast", "2-for-1 Breakfast"),
            ("resort_credit", "$50 Resort Credit"),
            ("late_checkout", "Late Checkout (2pm)")
        ]:
            checkbox = QCheckBox(promo_name)
            checkbox.setChecked(False)
            checkbox.setStyleSheet("font-size: 11pt; font-weight: 500; color: #333;")
            checkbox.stateChanged.connect(self._on_control_changed)

            self.promo_checkboxes[promo_id] = checkbox
            promo_layout.addWidget(checkbox)

        controls_layout.addWidget(promo_group)
        controls_layout.addStretch(1) # Push content to top

        top_section_layout.addWidget(controls_group)

        # Results area - more compact and refined
        results_group = QGroupBox("Financial Impact (30-day Outlook)")
        results_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                color: #27ae60;
                border: 2px solid #a8e6cf;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 6px;
                background-color: #e6ffee;
                border-radius: 4px;
            }
        """)
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)

        # KPI tiles - horizontal layout for better space usage
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(10)

        self.kpi_widgets = {}
        kpi_metrics = ["RevPAR", "GOPPAR", "Revenue", "Cost", "Profit"]

        for metric in kpi_metrics:
            kpi_frame = QFrame()
            kpi_frame.setFrameShape(QFrame.StyledPanel)
            kpi_frame.setLineWidth(1)
            kpi_frame.setFixedSize(120, 70) # More compact size
            kpi_frame.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                                stop: 0 #e8f5e9, stop: 1 #ffffff);
                    border: 2px solid #81c784;
                    border-radius: 8px;
                    padding: 4px;
                }
            """)

            kpi_layout_internal = QVBoxLayout(kpi_frame)
            kpi_layout_internal.setContentsMargins(2, 2, 2, 2)
            kpi_layout_internal.setSpacing(1)

            title_label = QLabel(metric)
            title_label.setStyleSheet("font-size: 8pt; font-weight: 600; color: #2e7d32;")
            title_label.setAlignment(Qt.AlignCenter)

            value_label = QLabel("--")
            value_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #222; padding: 2px 0;")
            value_label.setAlignment(Qt.AlignCenter)

            kpi_layout_internal.addWidget(title_label)
            kpi_layout_internal.addWidget(value_label)

            self.kpi_widgets[metric.lower()] = value_label
            kpi_layout.addWidget(kpi_frame)

        results_layout.addLayout(kpi_layout)

        # Reduce spacing
        results_layout.addSpacing(8)

        # Waterfall chart placeholder - more compact
        self.waterfall_chart = QWidget()
        self.waterfall_chart.setMinimumHeight(280) # Reduced height
        self.waterfall_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        results_layout.addWidget(self.waterfall_chart)

        top_section_layout.addWidget(results_group)
        top_section_layout.setStretch(0, 1) # Controls take available space
        top_section_layout.setStretch(1, 2) # Results take more space

        main_layout.addLayout(top_section_layout)

        # Add explanation after the main content for better flow
        explanation_text = (
            "**What-If Turbo Simulation** - Test different scenarios and see their financial impact instantly.\n\n"

            "**💰 Key Metrics Explained:**\n"
            "• **RevPAR** (Revenue per Available Room) = Total Revenue ÷ Total Available Rooms\n"
            "• **GOPPAR** (Gross Operating Profit per Available Room) = Gross Operating Profit ÷ Total Available Rooms\n"
            "• **Revenue/Cost/Profit** show percentage changes from your baseline performance\n\n"

            "**🎯 How to Use:**\n"
            "• **Room Rates**: Adjust prices by ±50 SAR for each room type\n"
            "• **Occupancy**: Set target occupancy between 50-100%\n"
            "• **Staffing**: Modify housekeeping/F&B staff by ±5 FTE\n"
            "• **Promotions**: Select promotional bundles to activate\n\n"

            "**📊 Reading Results:**\n"
            "• **Green percentages** = Positive impact on performance\n"
            "• **Red percentages** = Negative impact requiring consideration\n"
            "• All projections are based on 30-day outlook for strategic planning"
        )
        explanation_widget = _collapsible(explanation_text)
        explanation_widget.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(explanation_widget)

        # Add bottom padding and scroll hint
        bottom_hint = QLabel("📊 Scroll up/down to see all controls and results")
        bottom_hint.setStyleSheet("""
            QLabel {
                background-color: #e8f4fd;
                color: #2980b9;
                padding: 8px;
                border-radius: 4px;
                font-size: 9pt;
                font-style: italic;
                text-align: center;
                border: 1px solid #d4e6f1;
            }
        """)
        bottom_hint.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(bottom_hint)

        # Add some bottom spacing
        main_layout.addSpacing(20)

        # Add scroll area to main wrapper
        main_wrapper_layout.addWidget(scroll_area)

        # Connect fallback signal
        self.fallback_needed.connect(self._handle_fallback)

        # Load matrix (synchronous)
        self._load_elasticity_matrix()

    def _load_elasticity_matrix(self):
        """Load elasticity matrix with fallback to default values"""
        global ELASTICITY_MATRIX, BASELINE_DATA, ROOM_TYPES

        # Use default values for demonstration
        ROOM_TYPES = ["Standard", "Deluxe", "Suite", "Presidential"]

        # Create a simple elasticity matrix (room_types x metrics)
        # Metrics: occupancy impact, revenue impact, cost impact, profit impact, satisfaction
        # Added more rows to match potential staff and promo impacts for local_calc
        ELASTICITY_MATRIX = np.array([
            [0.8, 1.0, 0.5, 0.9, 0.7],  # Standard (Occupancy, Rev, Cost, Profit, Sat)
            [0.7, 1.2, 0.6, 1.1, 0.8],  # Deluxe
            [0.6, 1.5, 0.7, 1.3, 0.9],  # Suite
            [0.5, 2.0, 0.8, 1.5, 1.0],  # Presidential
            # Placeholder rows for staff impacts (e.g., impact on [Occ, Rev, Cost, Profit, Sat])
            [0.01, 0.0, 0.05, -0.05, 0.1], # Housekeeping staff change impact
            [0.005, 0.01, 0.03, -0.02, 0.08], # F&B staff change impact
            # Placeholder rows for promo impacts (e.g., impact on [Occ, Rev, Cost, Profit, Sat])
            [0.02, 0.05, 0.01, 0.04, 0.03], # Spa Discount impact
            [0.03, 0.06, 0.02, 0.04, 0.05], # Breakfast promo impact
            [0.015, 0.04, 0.01, 0.03, 0.02], # Resort Credit impact
            [0.008, 0.02, 0.005, 0.01, 0.01]  # Late Checkout impact
        ])


        # Default baseline data with more realistic hotel values in SAR
        BASELINE_DATA = {
            "occupancy": 75,
            "revenue": 450000,  # 450K SAR monthly revenue
            "costs": 270000,    # 270K SAR monthly costs
            "profit": 180000,   # 180K SAR monthly profit
            "revpar": 450000 / (75 * 100), # Assuming 100 rooms
            "goppar": 180000 / (75 * 100),
            "total_revenue": 450000,
            "total_cost": 270000,
            "net_profit": 180000
        }

        # Set initial occupancy to baseline
        if hasattr(self, 'occ_slider'):
            self.occ_slider.setValue(BASELINE_DATA["occupancy"])

        # Run initial calculation
        self._on_control_changed()

    @Slot()
    def _on_control_changed(self):
        """Handle any control value change and update UI labels."""
        # Debounce to avoid recalculating too frequently
        if not hasattr(self, "_last_update"):
            self._last_update = 0
        import time
        current_time = time.time()
        if current_time - self._last_update < 0.1:  # 100ms debounce
            return
        self._last_update = current_time

        # Update room rate labels
        for room_type, (slider, label) in self.room_sliders.items():
            value = slider.value()
            label.setText(f"{value:+d} SAR")

        # Update occupancy label
        self.occ_label.setText(f"{self.occ_slider.value()}%")

        # Update spinbox prefixes
        for dept, spinbox in self.staff_spinboxes.items():
            spinbox.setPrefix("+" if spinbox.value() >= 0 else "")


        # Calculate impact and update KPIs
        if ELASTICITY_MATRIX is not None and BASELINE_DATA is not None:
            try:
                scenario = self._get_current_scenario()
                results = self._local_calc(scenario)
                if results:
                    self._update_results_display(results)
                else:
                    print("Warning: _local_calc returned None")
                    # Set default values
                    for widget in self.kpi_widgets.values():
                        widget.setText("--")
            except Exception as e:
                print(f"Error in _on_control_changed: {e}")
                import traceback
                traceback.print_exc()
                # Set error values
                for widget in self.kpi_widgets.values():
                    widget.setText("ERROR")
        else:
            print("Warning: ELASTICITY_MATRIX or BASELINE_DATA is None")

    def _get_current_scenario(self):
        """Get current values from all controls"""
        # Room rates
        rates = {}
        for room_type, (slider, _) in self.room_sliders.items():
            rates[room_type] = slider.value()

        # Occupancy
        occupancy = self.occ_slider.value()

        # Staffing
        staffing = {}
        for dept, spinbox in self.staff_spinboxes.items():
            staffing[dept] = spinbox.value()

        # Promotions
        promotions = []
        for promo_id, checkbox in self.promo_checkboxes.items():
            if checkbox.isChecked():
                promotions.append(promo_id)

        return {
            "rates": rates,  # Changed from "room_rates"
            "occupancy": occupancy,  # Changed from "target_occupancy"
            "staffing": staffing,
            "promotions": promotions,
            "date_range": (date.today(), date.today() + timedelta(days=29))
        }

    def _local_calc(self, scenario):
        """Perform local calculation using simplified logic"""
        if ELASTICITY_MATRIX is None or BASELINE_DATA is None:
            return None

        try:
            # Get baseline values
            baseline_revenue = BASELINE_DATA["total_revenue"]
            baseline_cost = BASELINE_DATA["total_cost"]
            baseline_profit = BASELINE_DATA["net_profit"]
            baseline_occupancy = BASELINE_DATA["occupancy"]
            baseline_revpar = BASELINE_DATA["revpar"]
            baseline_goppar = BASELINE_DATA["goppar"]

            # Simplified waterfall data calculation with realistic values

            # Room rate impact (simplified calculation)
            total_rate_change = sum(scenario["rates"].values())
            rate_impact_factor = total_rate_change * 0.8  # 80% of rate changes flow to profit

            # Occupancy impact
            occupancy_change_pct = scenario.get("occupancy", 75) - baseline_occupancy
            occupancy_impact_factor = occupancy_change_pct * baseline_profit * 0.02  # 2% profit change per 1% occupancy

            # Promotion impact (cost of promotions)
            promo_count = len(scenario.get("promotions", []))
            promo_impact_factor = -promo_count * 2000  # Each promo costs ~2000 SAR

            # Staffing impact
            staff_changes = scenario.get("staffing", {})
            total_staff_change = staff_changes.get("housekeeping", 0) + staff_changes.get("fnb", 0)
            staff_impact_factor = total_staff_change * -3000  # Each FTE costs 3000 SAR per month

            # Calculate total impacts
            total_revenue_impact = rate_impact_factor * 1.2 + occupancy_impact_factor * 0.5  # Revenue benefits more from rates and occupancy
            total_cost_impact = -staff_impact_factor + (promo_count * 500)  # Staff costs and promo costs
            total_profit_impact = rate_impact_factor + occupancy_impact_factor + promo_impact_factor + staff_impact_factor

            # Calculate final values
            final_revenue = baseline_revenue + total_revenue_impact
            final_cost = baseline_cost + total_cost_impact
            final_profit = baseline_profit + total_profit_impact

            # Calculate RevPAR and GOPPAR changes (simplified)
            revenue_change_pct = total_revenue_impact / baseline_revenue if baseline_revenue > 0 else 0
            final_revpar = baseline_revpar * (1 + revenue_change_pct)
            final_goppar = baseline_goppar * (1 + (total_profit_impact / baseline_profit) if baseline_profit > 0 else 0)

            # Calculate percentage changes
            deltas_percent = {
                "revpar": ((final_revpar - baseline_revpar) / baseline_revpar * 100) if baseline_revpar > 0 else 0,
                "goppar": ((final_goppar - baseline_goppar) / baseline_goppar * 100) if baseline_goppar > 0 else 0,
                "revenue": (total_revenue_impact / baseline_revenue * 100) if baseline_revenue > 0 else 0,
                "cost": (total_cost_impact / baseline_cost * 100) if baseline_cost > 0 else 0,
                "profit": (total_profit_impact / baseline_profit * 100) if baseline_profit > 0 else 0
            }

            # Waterfall data
            waterfall_data = {
                "baseline": float(baseline_profit),
                "price": float(rate_impact_factor),
                "occupancy": float(occupancy_impact_factor),
                "promo": float(promo_impact_factor),
                "staff": float(staff_impact_factor),
                "final": float(final_profit)
            }

            result = {
                "deltas": {
                    "revpar": final_revpar - baseline_revpar,
                    "goppar": final_goppar - baseline_goppar,
                    "revenue": total_revenue_impact,
                    "cost": total_cost_impact,
                    "profit": total_profit_impact
                },
                "deltas_percent": deltas_percent,
                "waterfall": waterfall_data
            }

            return result

        except Exception as e:
            print(f"Error in local calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _update_results_display(self, results):
        """Update the UI with calculation results"""
        if results is None:
            # Clear displays or show error
            for widget in self.kpi_widgets.values():
                widget.setText("--")
            # Clear waterfall chart
            self.waterfall_chart = create_plotly_widget(go.Figure()) # Empty figure
            return

        # Update KPI tiles
        metrics = {
            "revpar": "RevPAR",
            "goppar": "GOPPAR",
            "revenue": "Revenue",
            "cost": "Cost",
            "profit": "Profit"
        }

        for metric_key, display_name in metrics.items():
            widget = self.kpi_widgets[metric_key.lower()]
            delta_percent = results["deltas_percent"].get(metric_key, 0.0)

            # Format text with colors
            color = self._get_impact_color(delta_percent, metric_key == "cost")
            formatted_value = f"{delta_percent:+.1f}%"

            widget.setText(formatted_value)
            widget.setStyleSheet(f"font-size: 18pt; font-weight: bold; color: {color}; padding: 4px 0 2px 0;") # Adjusted font size and padding

        # Update waterfall chart
        self._update_waterfall_chart(results["waterfall"])

    def _get_impact_color(self, value, invert=False):
        """Get color based on impact (green=good, red=bad)"""
        if abs(value) < 0.1:  # Near zero, slightly increased threshold for "neutral"
            return "#6c757d"  # Muted Gray

        if invert:
            value = -value  # For costs, negative is good

        if value >= 5: # Strong positive impact
            return "#28a745"  # Green
        elif value >= 1: # Moderate positive impact
            return "#20c997" # Teal-ish green
        elif value <= -5: # Strong negative impact
            return "#dc3545"  # Red
        elif value <= -1: # Moderate negative impact
            return "#fd7e14" # Orange-red
        else:
            return "#ffc107"  # Yellow/Amber for small positive/negative

    def _update_waterfall_chart(self, data):
        """Update the waterfall chart with new data"""
        fig = go.Figure(go.Waterfall(
            name="P&L Impact",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Baseline P&L", "Price Impact", "Occupancy Impact", "Promotions Cost/Benefit", "Staffing Cost/Benefit", "New P&L"],
            y=[
                data["baseline"],
                data["price"],
                data["occupancy"],
                data["promo"],
                data["staff"],
                data["final"] # Use the calculated final value explicitly
            ],
            text=[
                f"{format_currency(data['baseline'])}",
                f"{format_currency(data['price'], with_sign=True)}",
                f"{format_currency(data['occupancy'], with_sign=True)}",
                f"{format_currency(data['promo'], with_sign=True)}",
                f"{format_currency(data['staff'], with_sign=True)}",
                f"{format_currency(data['final'])}"
            ],
            textposition="outside",
            connector={"line": {"color": "#6c757d", "dash": "dot"}}, # Muted connector
            decreasing={"marker": {"color": "#dc3545", "line": {"color": "#dc3545", "width": 1}}}, # Red
            increasing={"marker": {"color": "#28a745", "line": {"color": "#28a745", "width": 1}}}, # Green
            totals={"marker": {"color": "#007bff", "line": {"color": "#007bff", "width": 1}}} # Blue for totals
        ))

        fig.update_layout(
            title_text="<b>P&L Impact Breakdown (SAR)</b>",
            title_font_size=16,
            showlegend=False,
            height=350,
            margin=dict(t=60, l=40, r=40, b=50),
            plot_bgcolor='#f8f9fa', # Light plot background
            paper_bgcolor='#ffffff', # White paper background
            font=dict(family="Arial, sans-serif", size=10, color="#343a40"),
            xaxis_title="Impact Categories",
            yaxis_title="Amount (SAR)"
        )

        # Update the widget
        new_chart = create_plotly_widget(fig)

        # Replace the old waterfall chart widget with the new one
        try:
            # Get the parent widget and its layout
            parent_widget = self.waterfall_chart.parent()
            if parent_widget and hasattr(parent_widget, 'layout') and parent_widget.layout():
                layout = parent_widget.layout()
                # Find the index of the current widget and replace it
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget() == self.waterfall_chart:
                        layout.removeWidget(self.waterfall_chart)
                        self.waterfall_chart.deleteLater()
                        layout.insertWidget(i, new_chart)
                        break
                else:
                    # If not found in layout, just add the new widget
                    layout.addWidget(new_chart)
                    self.waterfall_chart.deleteLater()
            else:
                # Fallback: just delete the old widget and update reference
                self.waterfall_chart.deleteLater()
        except Exception as e:
            print(f"Error replacing waterfall chart: {e}")
            # Fallback: just update the reference
            if hasattr(self, 'waterfall_chart'):
                self.waterfall_chart.deleteLater()

        self.waterfall_chart = new_chart
        self.waterfall_chart.setMinimumHeight(350) # Maintain size
        self.waterfall_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


    @Slot(dict)
    def _handle_fallback(self, scenario):
        """Handle server-side calculation for longer periods"""
        # For now, just use local calculation as fallback
        self._fetch_server_calculation(scenario)

    def _fetch_server_calculation(self, scenario):
        """Simplified calculation without server dependency"""
        try:
            # Use local calculation as fallback
            print("Using local calculation fallback")
            result = self._local_calc(scenario) # Use the main local_calc
            self._update_results_display(result)
        except Exception as e:
            print(f"Error in fallback calculation: {e}")

    # The _calculate_local_scenario and _transform_server_response methods are likely remnants
    # or for different contexts. Given _local_calc is the main one now, these might be redundant.
    # Keeping them for completeness but they are not called in the updated flow.
    def _calculate_local_scenario(self, scenario):
        """Local scenario calculation - *Deprecated in favor of _local_calc*"""
        # This function's logic is simpler and doesn't use the elasticity matrix fully.
        # It's kept for original structure but not used in current `_fetch_server_calculation` after my change.
        room_rates = scenario.get("room_rates", {})
        target_occupancy = scenario.get("target_occupancy", 75)

        # Calculate basic metrics
        # Placeholder calculation if BASELINE_DATA is not fully populated or for a very simple model
        base_revenue_per_room_type = {
            "Standard": 10000, "Deluxe": 15000, "Suite": 20000, "Presidential": 30000
        }
        total_revenue = 0
        for room_type, rate_change in room_rates.items():
            # Assume rate_change is SAR amount. Base revenue adjusted by a simple factor.
            # This is a very rough estimate.
            total_revenue += (base_revenue_per_room_type.get(room_type, 0) + rate_change * 100) * (target_occupancy / 100.0)

        total_costs = total_revenue * 0.6  # Assume 60% cost ratio
        profit = total_revenue - total_costs

        # Mock waterfall data based on this simple calculation
        baseline_profit_mock = BASELINE_DATA.get("net_profit", 40000) if BASELINE_DATA else 40000
        baseline_revenue_mock = BASELINE_DATA.get("total_revenue", 100000) if BASELINE_DATA else 100000
        baseline_costs_mock = BASELINE_DATA.get("total_cost", 60000) if BASELINE_DATA else 60000

        # Simplified profit changes for waterfall
        mock_price_impact = (total_revenue - baseline_revenue_mock) * 0.5
        mock_occupancy_impact = (total_revenue - baseline_revenue_mock) * 0.5
        mock_staff_impact = 0 # No staff logic in this simple calc
        mock_promo_impact = 0 # No promo logic in this simple calc

        return {
            "revenue": total_revenue,
            "costs": total_costs,
            "profit": profit,
            "occupancy": target_occupancy,
            "deltas": { # Simplified deltas for consistency
                "revpar": (total_revenue / (target_occupancy/100) / 30) - (baseline_revenue_mock / (BASELINE_DATA["occupancy"]/100) / 30) if (target_occupancy and BASELINE_DATA and BASELINE_DATA["occupancy"]) else 0,
                "goppar": (profit / (target_occupancy/100) / 30) - (baseline_profit_mock / (BASELINE_DATA["occupancy"]/100) / 30) if (target_occupancy and BASELINE_DATA and BASELINE_DATA["occupancy"]) else 0,
                "revenue": total_revenue - baseline_revenue_mock,
                "cost": total_costs - baseline_costs_mock,
                "profit": profit - baseline_profit_mock
            },
            "deltas_percent": {
                "revpar": ((total_revenue / (target_occupancy/100)) / baseline_revenue_mock - 1) * 100 if baseline_revenue_mock and target_occupancy else 0,
                "goppar": (profit / baseline_profit_mock - 1) * 100 if baseline_profit_mock else 0,
                "revenue": (total_revenue / baseline_revenue_mock - 1) * 100 if baseline_revenue_mock else 0,
                "cost": (total_costs / baseline_costs_mock - 1) * 100 if baseline_costs_mock else 0,
                "profit": (profit / baseline_profit_mock - 1) * 100 if baseline_profit_mock else 0
            },
            "waterfall": {
                "baseline": baseline_profit_mock,
                "price": mock_price_impact,
                "occupancy": mock_occupancy_impact,
                "promo": mock_promo_impact,
                "staff": mock_staff_impact,
                "final": profit
            }
        }

    def _transform_server_response(self, server_result):
        """Transform server response to match local calculation format - *Not currently used*"""
        # This method is for when a server actually provides a response.
        # Since we're using a local fallback for now, it's not strictly necessary.
        return {
            "deltas": {
                "revpar": server_result.get("revpar_delta", 0.0),
                "goppar": server_result.get("goppar_delta", 0.0),
                "revenue": server_result.get("revenue_delta", 0.0),
                "cost": server_result.get("cost_delta", 0.0),
                "profit": server_result.get("profit_delta", 0.0)
            },
            "deltas_percent": {
                "revpar": server_result.get("revpar_delta_percent", 0.0),
                "goppar": server_result.get("goppar_delta_percent", 0.0),
                "revenue": server_result.get("revenue_delta_percent", 0.0),
                "cost": server_result.get("cost_delta_percent", 0.0),
                "profit": server_result.get("profit_delta_percent", 0.0)
            },
            "waterfall": {
                "baseline": server_result.get("waterfall_data", {}).get("baseline", 0.0),
                "price": server_result.get("waterfall_data", {}).get("price_impact", 0.0),
                "occupancy": server_result.get("waterfall_data", {}).get("occupancy_impact", 0.0),
                "promo": server_result.get("waterfall_data", {}).get("promo_impact", 0.0),
                "staff": server_result.get("waterfall_data", {}).get("staff_impact", 0.0),
                "final": server_result.get("waterfall_data", {}).get("final", 0.0)
            }
        }

@data_required
def display():
    """Display function for the What-If Turbo view"""
    try:
        return WhatIfPanel()
    except Exception as e:
        print(f"Error creating WhatIfPanel: {e}")
        import traceback
        traceback.print_exc()

        # Return a simple error widget as fallback
        from PySide6.QtWidgets import QLabel, QVBoxLayout
        error_widget = QWidget()
        layout = QVBoxLayout(error_widget)
        error_label = QLabel(f"Error loading What-If Analysis: {str(e)}")
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
        return error_widget
