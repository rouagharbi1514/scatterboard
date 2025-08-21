# views/what_if.py
"""
What-If Analysis View
=====================

Interactive scenario analysis tool for hotel revenue management with
real-time financial impact visualization.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QCheckBox, QGroupBox, QPushButton, QFrame, QSizePolicy,
    QScrollArea, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPalette

from views.utils import create_plotly_widget, data_required, format_currency

# Matrix cache
ELASTICITY_MATRIX: np.ndarray = None
BASELINE_DATA: Dict = None
ROOM_TYPES: List[str] = []


def _collapsible(text: str) -> QWidget:
    """Create a collapsible explanation panel (UI only)."""
    container = QWidget()
    container.setStyleSheet("""
        QWidget {
            background-color: #FFFFFF;
            border: 1px solid #E6EAF1;
            border-radius: 12px;
            margin: 6px 0;
        }
    """)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(8)

    toggle_btn = QPushButton("📋 Show explanation")
    toggle_btn.setCursor(Qt.PointingHandCursor)
    toggle_btn.setStyleSheet("""
        QPushButton {
            background: #F2F6FF;
            color: #0F172A;
            padding: 8px 12px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 12px;
            border: 1px solid #D8E3F5;
            text-align: left;
        }
        QPushButton:hover { background: #EAF0FF; }
        QPushButton:pressed { background: #E0E8FF; }
    """)

    explanation = QLabel(text)
    explanation.setWordWrap(True)
    explanation.setStyleSheet("""
        QLabel {
            background: #FAFCFF;
            color: #334155;
            padding: 12px;
            border-radius: 10px;
            font-size: 12px;
            line-height: 1.55;
            border: 1px solid #E6EAF1;
        }
    """)
    explanation.setVisible(False)

    layout.addWidget(toggle_btn)
    layout.addWidget(explanation)

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

        # Modern background (soft gradients) — UI only
        self.setObjectName("whatIfRoot")
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#F6F8FC"))
        self.setPalette(palette)

        # Global, modern QSS
        self.setStyleSheet("""
            #whatIfRoot {
                background:
                    radial-gradient(560px 380px at 3% 5%, #EEF4FF 0%, transparent 60%),
                    radial-gradient(620px 420px at 100% 98%, #F8FBFF 0%, transparent 60%),
                    #F6F8FC;
                font-family: "Inter","Segoe UI", Arial;
                color: #0F172A;
            }

            /* Title */
            #titleLabel {
                font-size: 22px;
                font-weight: 800;
                letter-spacing: .2px;
                color: #0F172A;
            }

            /* Accent divider */
            #titleAccent {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2563EB, stop:1 #5B7CFF);
                border-radius: 2px;
            }

            /* Cards / groups */
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E6EAF1;
                border-radius: 14px;
                margin-top: 12px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 2px 8px;
                margin-left: 8px;
                color: #1E293B;
                background: #EFF4FF;
                border: 1px solid #DFE8FF;
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
            }

            /* Labels */
            .muted { color: #475569; font-size: 12px; }

            /* Sliders */
            QSlider::groove:horizontal {
                height: 6px; border-radius: 4px;
                background: #E7EEF9;
                margin: 8px 0;
            }
            QSlider::handle:horizontal {
                background: #2563EB;
                width: 18px; height: 18px;
                border: 2px solid #FFFFFF;
                border-radius: 9px;
                margin: -7px 0;
            }
            QSlider::sub-page:horizontal { background: #93C5FD; border-radius: 4px; }

            /* SpinBox */
            QSpinBox {
                background: #FFFFFF; color: #0F172A;
                border: 1px solid #D8E3F5; border-radius: 10px;
                padding: 5px 6px; min-width: 72px; height: 30px; font-weight: 700;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 18px; border: 1px solid #D8E3F5; border-radius: 6px;
                background: #EFF4FF;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #EAF0FF; }

            /* CheckBox */
            QCheckBox { font-size: 12px; color: #0F172A; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #CFDAEC; border-radius: 4px;
                background: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background: #2563EB;
                image: none;
            }

            /* ScrollArea */
            QScrollArea {
                border: 1px solid #E6EAF1;
                background-color: transparent;
                border-radius: 12px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px; margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #CBD8EE; border-radius: 6px; min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #B8C9EA; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        # Main layout for the widget
        main_wrapper_layout = QVBoxLayout(self)
        main_wrapper_layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Content widget for the scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Main content layout
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title + accent
        title = QLabel("What-If Turbo Simulation")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        accent = QFrame()
        accent.setObjectName("titleAccent")
        accent.setFixedHeight(3)
        main_layout.addWidget(accent)

        # Top section: Controls and KPIs
        top_section_layout = QHBoxLayout()
        top_section_layout.setSpacing(12)

        # Controls area (card)
        controls_group = QGroupBox("Scenario Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(8)
        self._apply_card_shadow(controls_group)

        # Room Rate Sliders
        rate_group = QGroupBox("Room Rate Adjustments (SAR)")
        rate_layout = QVBoxLayout(rate_group)
        rate_layout.setSpacing(6)

        self.room_sliders: Dict[str, Tuple[QSlider, QLabel]] = {}
        for room_type in ["Standard", "Deluxe", "Suite", "Presidential"]:
            slider_layout = QHBoxLayout()
            label = QLabel(f"{room_type}:")
            label.setStyleSheet("font-size: 12px; font-weight: 600; min-width: 90px; color: #0F172A;")
            slider_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-50, 50)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.setFixedWidth(220)
            slider.valueChanged.connect(self._on_control_changed)

            value_label = QLabel("0 SAR")
            value_label.setStyleSheet("font-size: 12px; font-weight: 800; min-width: 60px; color: #111827;")

            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)

            self.room_sliders[room_type] = (slider, value_label)
            rate_layout.addLayout(slider_layout)

        controls_layout.addWidget(rate_group)

        # Occupancy slider
        occ_group = QGroupBox("Target Occupancy (%)")
        occ_layout = QHBoxLayout(occ_group)
        occ_layout.setSpacing(10)

        self.occ_slider = QSlider(Qt.Horizontal)
        self.occ_slider.setRange(50, 100)
        self.occ_slider.setValue(80)
        self.occ_slider.setTickPosition(QSlider.TicksBelow)
        self.occ_slider.setTickInterval(5)
        self.occ_slider.setFixedWidth(220)
        self.occ_slider.valueChanged.connect(self._on_control_changed)

        self.occ_label = QLabel("80%")
        self.occ_label.setStyleSheet("font-size: 12px; font-weight: 800; min-width: 50px; color: #111827;")

        occ_layout.addWidget(self.occ_slider)
        occ_layout.addWidget(self.occ_label)

        controls_layout.addWidget(occ_group)

        # Staffing controls
        staff_group = QGroupBox("Staffing Levels (FTE)")
        staff_layout = QVBoxLayout(staff_group)
        staff_layout.setSpacing(6)

        self.staff_spinboxes: Dict[str, QSpinBox] = {}
        for dept in ["Housekeeping", "F&B"]:
            staff_row = QHBoxLayout()
            label = QLabel(f"{dept}:")
            label.setStyleSheet("font-size: 12px; font-weight: 600; min-width: 90px; color: #0F172A;")
            staff_row.addWidget(label)

            spinbox = QSpinBox()
            spinbox.setRange(-5, 5)
            spinbox.setValue(0)
            spinbox.setPrefix("+" if spinbox.value() >= 0 else "")
            spinbox.setSuffix(" FTE")
            spinbox.valueChanged.connect(lambda v, sb=spinbox: sb.setPrefix("+" if v >= 0 else ""))
            spinbox.valueChanged.connect(self._on_control_changed)

            staff_row.addWidget(spinbox)
            self.staff_spinboxes[dept.lower()] = spinbox
            staff_layout.addLayout(staff_row)

        controls_layout.addWidget(staff_group)

        # Promotions
        promo_group = QGroupBox("Promotional Bundles")
        promo_layout = QVBoxLayout(promo_group)
        promo_layout.setSpacing(4)

        self.promo_checkboxes: Dict[str, QCheckBox] = {}
        for promo_id, promo_name in [
            ("spa_discount", "10% Spa Discount"),
            ("breakfast", "2-for-1 Breakfast"),
            ("resort_credit", "$50 Resort Credit"),
            ("late_checkout", "Late Checkout (2pm)")
        ]:
            checkbox = QCheckBox(promo_name)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self._on_control_changed)
            self.promo_checkboxes[promo_id] = checkbox
            promo_layout.addWidget(checkbox)

        controls_layout.addWidget(promo_group)
        controls_layout.addStretch(1)

        top_section_layout.addWidget(controls_group)

        # Results area (card)
        results_group = QGroupBox("Financial Impact (30-day Outlook)")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)
        self._apply_card_shadow(results_group)

        # KPI tiles
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(10)

        self.kpi_widgets: Dict[str, QLabel] = {}
        kpi_metrics = ["RevPAR", "GOPPAR", "Revenue", "Cost", "Profit"]

        for metric in kpi_metrics:
            kpi_frame = QFrame()
            kpi_frame.setFrameShape(QFrame.StyledPanel)
            kpi_frame.setLineWidth(1)
            kpi_frame.setFixedSize(128, 76)
            kpi_frame.setStyleSheet("""
                QFrame {
                    background: #FFFFFF;
                    border: 1px solid #E6EAF1;
                    border-radius: 14px;
                    padding: 6px;
                }
            """)
            self._apply_card_shadow(kpi_frame, blur=22, y=6, alpha=36)

            kpi_layout_internal = QVBoxLayout(kpi_frame)
            kpi_layout_internal.setContentsMargins(6, 6, 6, 6)
            kpi_layout_internal.setSpacing(2)

            title_label = QLabel(metric)
            title_label.setStyleSheet("font-size: 10px; font-weight: 700; color: #475569;")
            title_label.setAlignment(Qt.AlignCenter)

            value_label = QLabel("--")
            value_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #0F172A; padding: 2px 0;")
            value_label.setAlignment(Qt.AlignCenter)

            kpi_layout_internal.addWidget(title_label)
            kpi_layout_internal.addWidget(value_label)

            self.kpi_widgets[metric.lower()] = value_label
            kpi_layout.addWidget(kpi_frame)

        results_layout.addLayout(kpi_layout)
        results_layout.addSpacing(6)

        # Waterfall chart placeholder
        self.waterfall_chart = QWidget()
        self.waterfall_chart.setMinimumHeight(280)
        self.waterfall_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        results_layout.addWidget(self.waterfall_chart)

        top_section_layout.addWidget(results_group)
        top_section_layout.setStretch(0, 1)
        top_section_layout.setStretch(1, 2)

        main_layout.addLayout(top_section_layout)

        # Explanation
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
        explanation_widget.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(explanation_widget)

        # Bottom hint
        bottom_hint = QLabel("📊 Scroll up/down to see all controls and results")
        bottom_hint.setAlignment(Qt.AlignCenter)
        bottom_hint.setStyleSheet("""
            QLabel {
                background: #EFF6FF;
                color: #1E40AF;
                padding: 8px 10px;
                border-radius: 10px;
                font-size: 11px;
                border: 1px solid #DFE8FF;
            }
        """)
        main_layout.addWidget(bottom_hint)
        main_layout.addSpacing(14)

        # Add scroll area to main wrapper
        main_wrapper_layout.addWidget(scroll_area)

        # Connect fallback signal
        self.fallback_needed.connect(self._handle_fallback)

        # Load matrix (synchronous)
        self._load_elasticity_matrix()

    # --- UI helper (ombre douce) ---
    def _apply_card_shadow(self, widget: QWidget, blur: int = 24, y: int = 8, alpha: int = 42):
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(blur)
        effect.setOffset(0, y)
        effect.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(effect)

    def _load_elasticity_matrix(self) -> None:
        """Load elasticity matrix with fallback to default values"""
        global ELASTICITY_MATRIX, BASELINE_DATA, ROOM_TYPES

        ROOM_TYPES = ["Standard", "Deluxe", "Suite", "Presidential"]

        # base_rate, occupancy_elasticity, revenue_impact, cost_impact
        ELASTICITY_MATRIX = np.array([
            [400, 0.8, 1.0, 0.5],   # Standard
            [600, 0.7, 1.2, 0.6],   # Deluxe
            [900, 0.6, 1.5, 0.7],   # Suite
            [1500, 0.5, 2.0, 0.8],  # Presidential
        ])

        BASELINE_DATA = {
            "occupancy": 75,
            "revenue": 450000,
            "costs": 270000,
            "profit": 180000,
            "revpar": 450000 / (75 * 100),
            "goppar": 180000 / (75 * 100),
            "total_revenue": 450000,
            "total_cost": 270000,
            "net_profit": 180000,
            "room_counts": {
                "Standard": 50,
                "Deluxe": 30,
                "Suite": 15,
                "Presidential": 5
            }
        }

        if hasattr(self, 'occ_slider'):
            self.occ_slider.setValue(BASELINE_DATA["occupancy"])

        self._on_control_changed()

    @Slot()
    def _on_control_changed(self) -> None:
        """Handle any control value change and update UI labels."""
        if not hasattr(self, "_last_update"):
            self._last_update = 0
        import time
        current_time = time.time()
        if current_time - self._last_update < 0.1:
            return
        self._last_update = current_time

        for room_type, (slider, label) in self.room_sliders.items():
            value = slider.value()
            label.setText(f"{value:+d} SAR")

        self.occ_label.setText(f"{self.occ_slider.value()}%")

        for _, spinbox in self.staff_spinboxes.items():
            spinbox.setPrefix("+" if spinbox.value() >= 0 else "")

        if ELASTICITY_MATRIX is not None and BASELINE_DATA is not None:
            try:
                scenario = self._get_current_scenario()
                results = self._local_calc(scenario)
                if results:
                    self._update_results_display(results)
                else:
                    for widget in self.kpi_widgets.values():
                        widget.setText("--")
            except Exception as e:
                print(f"Error in _on_control_changed: {e}")
                import traceback
                traceback.print_exc()
                for widget in self.kpi_widgets.values():
                    widget.setText("ERROR")
        else:
            print("Warning: ELASTICITY_MATRIX or BASELINE_DATA is None")

    def _get_current_scenario(self) -> Dict:
        """Get current values from all controls"""
        rates = {room_type: slider.value() for room_type, (slider, _) in self.room_sliders.items()}
        occupancy = self.occ_slider.value()
        staffing = {dept: sb.value() for dept, sb in self.staff_spinboxes.items()}
        promotions = [pid for pid, cb in self.promo_checkboxes.items() if cb.isChecked()]

        return {
            "rates": rates,
            "occupancy": occupancy,
            "staffing": staffing,
            "promotions": promotions,
            "date_range": (date.today(), date.today() + timedelta(days=29))
        }

    def _local_calc(self, scenario: Dict) -> Dict:
        """Perform realistic financial impact calculation (unchanged logic)"""
        baseline_revenue = BASELINE_DATA["total_revenue"]
        baseline_cost = BASELINE_DATA["total_cost"]
        baseline_profit = BASELINE_DATA["net_profit"]
        baseline_occupancy = BASELINE_DATA["occupancy"]
        baseline_revpar = BASELINE_DATA["revpar"]
        baseline_goppar = BASELINE_DATA["goppar"]
        room_counts = BASELINE_DATA["room_counts"]
        total_rooms = sum(room_counts.values())

        revenue_impact = 0
        occupancy_change = scenario["occupancy"] - baseline_occupancy

        for i, room_type in enumerate(ROOM_TYPES):
            base_rate = ELASTICITY_MATRIX[i][0]
            elasticity = ELASTICITY_MATRIX[i][1]
            rate_change = scenario["rates"].get(room_type, 0)

            demand_factor = 1 - (elasticity * (rate_change / base_rate))
            room_revenue = (base_rate + rate_change) * room_counts[room_type] * 30
            revenue_impact += room_revenue * demand_factor - (base_rate * room_counts[room_type] * 30)

        occupancy_impact = occupancy_change * 0.01 * baseline_revenue

        promo_count = len(scenario["promotions"])
        promo_cost = promo_count * 2000
        promo_revenue_boost = 0.05 * baseline_revenue if promo_count > 0 else 0
        net_promo_impact = promo_revenue_boost - promo_cost

        staff_changes = scenario["staffing"]
        hk_change = staff_changes.get("housekeeping", 0)
        fb_change = staff_changes.get("f&b", 0)

        hk_cost = 2500 * hk_change * (1 - 0.02 * max(0, hk_change))
        fb_cost = 3000 * fb_change * (1 - 0.015 * max(0, fb_change))
        staff_cost_impact = hk_cost + fb_cost

        staff_revenue_impact = 0.01 * baseline_revenue * (hk_change + fb_change)

        total_revenue_impact = revenue_impact + occupancy_impact + net_promo_impact + staff_revenue_impact
        total_cost_impact = staff_cost_impact - (0.3 * total_revenue_impact)

        final_revenue = baseline_revenue + total_revenue_impact
        final_cost = baseline_cost + total_cost_impact
        final_profit = final_revenue - final_cost

        final_revpar = final_revenue / (total_rooms * 30)
        final_goppar = final_profit / (total_rooms * 30)

        deltas_percent = {
            "revpar": ((final_revpar - baseline_revpar) / baseline_revpar * 100) if baseline_revpar > 0 else 0,
            "goppar": ((final_goppar - baseline_goppar) / baseline_goppar * 100) if baseline_goppar > 0 else 0,
            "revenue": (total_revenue_impact / baseline_revenue * 100) if baseline_revenue > 0 else 0,
            "cost": (total_cost_impact / baseline_cost * 100) if baseline_cost > 0 else 0,
            "profit": ((final_profit - baseline_profit) / baseline_profit * 100) if baseline_profit > 0 else 0
        }

        waterfall_data = {
            "baseline": float(baseline_profit),
            "price": float(revenue_impact),
            "occupancy": float(occupancy_impact),
            "promo": float(net_promo_impact),
            "staff": float(staff_revenue_impact - staff_cost_impact),
            "final": float(final_profit)
        }

        return {
            "deltas_percent": deltas_percent,
            "waterfall": waterfall_data
        }

    def _update_results_display(self, results: Dict) -> None:
        """Update the UI with calculation results (UI only)"""
        if results is None:
            for widget in self.kpi_widgets.values():
                widget.setText("--")
            self.waterfall_chart = create_plotly_widget(go.Figure())
            return

        metrics = {
            "revpar": "RevPAR",
            "goppar": "GOPPAR",
            "revenue": "Revenue",
            "cost": "Cost",
            "profit": "Profit"
        }

        for metric_key, _ in metrics.items():
            widget = self.kpi_widgets[metric_key.lower()]
            delta_percent = results["deltas_percent"].get(metric_key, 0.0)

            color = self._get_impact_color(delta_percent, metric_key == "cost")
            formatted_value = f"{delta_percent:+.1f}%"

            widget.setText(formatted_value)
            widget.setStyleSheet(f"font-size: 18px; font-weight: 800; color: {color}; padding: 2px 0;")

        self._update_waterfall_chart(results["waterfall"])

    def _get_impact_color(self, value: float, invert: bool = False) -> str:
        """Get color based on impact (green=good, red=bad)"""
        if abs(value) < 0.1:
            return "#64748B"  # Muted gray
        if invert:
            value = -value
        if value >= 5:
            return "#16A34A"
        if value >= 1:
            return "#22C55E"
        if value <= -5:
            return "#DC2626"
        if value <= -1:
            return "#F97316"
        return "#EAB308"

    def _update_waterfall_chart(self, data: Dict) -> None:
        """Update the waterfall chart with new data"""
        fig = go.Figure(go.Waterfall(
            name="P&L Impact",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Baseline", "Price Impact", "Occupancy Impact", "Promotions", "Staffing", "Final"],
            y=[data["baseline"], data["price"], data["occupancy"], data["promo"], data["staff"], data["final"]],
            text=[
                f"{format_currency(data['baseline'])}",
                f"{format_currency(data['price'], with_sign=True)}",
                f"{format_currency(data['occupancy'], with_sign=True)}",
                f"{format_currency(data['promo'], with_sign=True)}",
                f"{format_currency(data['staff'], with_sign=True)}",
                f"{format_currency(data['final'])}"
            ],
            textposition="outside",
            connector={"line": {"color": "#94A3B8", "dash": "dot"}},
            decreasing={"marker": {"color": "#EF4444", "line": {"color": "#DC2626", "width": 1}}},
            increasing={"marker": {"color": "#22C55E", "line": {"color": "#16A34A", "width": 1}}},
            totals={"marker": {"color": "#2563EB", "line": {"color": "#1D4ED8", "width": 1}}}
        ))

        fig.update_layout(
            title_text="<b>Profit & Loss Impact (30 Days)</b>",
            title_font_size=16,
            showlegend=False,
            height=360,
            margin=dict(t=60, l=40, r=40, b=50),
            plot_bgcolor='#F8FAFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Arial, sans-serif", size=11, color="#0F172A"),
            xaxis_title="Impact Categories",
            yaxis_title="Amount (SAR)",
            hovermode="x unified"
        )
        fig.add_hline(y=data["baseline"], line_dash="dot", line_color="#94A3B8")

        new_chart = create_plotly_widget(fig)

        try:
            parent_widget = self.waterfall_chart.parent()
            if parent_widget and hasattr(parent_widget, 'layout') and parent_widget.layout():
                layout = parent_widget.layout()
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget() == self.waterfall_chart:
                        layout.removeWidget(self.waterfall_chart)
                        self.waterfall_chart.deleteLater()
                        layout.insertWidget(i, new_chart)
                        break
                else:
                    layout.addWidget(new_chart)
                    self.waterfall_chart.deleteLater()
            else:
                self.waterfall_chart.deleteLater()
        except Exception as e:
            print(f"Error replacing waterfall chart: {e}")
            if hasattr(self, 'waterfall_chart'):
                self.waterfall_chart.deleteLater()

        self.waterfall_chart = new_chart
        self.waterfall_chart.setMinimumHeight(360)
        self.waterfall_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @Slot(dict)
    def _handle_fallback(self, scenario: Dict) -> None:
        """Handle server-side calculation for longer periods"""
        self._fetch_server_calculation(scenario)

    def _fetch_server_calculation(self, scenario: Dict) -> None:
        """Simplified calculation without server dependency"""
        try:
            result = self._local_calc(scenario)
            self._update_results_display(result)
        except Exception as e:
            print(f"Error in fallback calculation: {e}")


@data_required
def display() -> QWidget:
    """Display function for the What-If Turbo view"""
    try:
        return WhatIfPanel()
    except Exception as e:
        print(f"Error creating WhatIfPanel: {e}")
        import traceback
        traceback.print_exc()

        # Fallback error widget (UI only)
        error_widget = QWidget()
        layout = QVBoxLayout(error_widget)
        error_label = QLabel(f"Error loading What-If Analysis: {str(e)}")
        error_label.setStyleSheet("""
            QLabel {
                color: #DC2626;
                font-size: 14pt;
                padding: 20px;
                background-color: #FFF2F2;
                border: 1px solid #FECACA;
                border-radius: 12px;
            }
        """)
        layout.addWidget(error_label)
        return error_widget
