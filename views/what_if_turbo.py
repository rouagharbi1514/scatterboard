# flake8: noqa
# views/what_if_turbo.py
"""
What-If Turbo Panel
===================

Dockable side-panel for real-time scenario analysis with interactive controls
and Plotly waterfall charts. Provides instant feedback on KPI changes.
"""

from __future__ import annotations
import json
import traceback
import numpy as np
from typing import Dict, Tuple, List
from io import StringIO
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QProgressBar,
)
# Try to import QWebEngineView, fallback to QLabel if not available
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    WEBENGINE_AVAILABLE = True
except ImportError:
    # Fallback for environments without QWebEngine
    QWebEngineView = None
    QWebEngineSettings = None
    WEBENGINE_AVAILABLE = False

from data.helpers import get_df
from views.utils import data_required, format_currency
try:
    from connectors.local_server_connector import get_baseline_data
    from connectors.oracle_cloud_connector import fetch_baseline_data
except ImportError:
    def get_baseline_data() -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_baseline_data() -> pd.DataFrame:
        return pd.DataFrame()


class WhatIfTurboPanel(QWidget):
    """
    What-If Turbo analysis panel with real-time scenario calculations.

    Features:
    - Interactive controls for room rate, occupancy, staffing
    - Real-time KPI updates
    - Plotly waterfall charts
    - Export capabilities (JSON, Excel, PDF)
    - Offline-first data loading with caching
    """

    # Signal emitted when scenario changes
    scenario_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What-If Turbo Analysis")
        self.setMinimumWidth(400)

        # Data attributes
        self.baseline_data = pd.DataFrame()
        self.baseline_kpis: Dict[str, float] = {}
        self.current_scenario: Dict[str, float] = {}
        self.room_types: List[str] = ["Standard", "Deluxe", "Suite", "Presidential"]
        self.room_counts: Dict[str, int] = {}

        # Initialize chart_view to None
        self.chart_view = None

        try:
            # UI setup
            self._setup_ui()
            self._load_baseline_data()
            self._connect_signals()

            # Update timer to prevent excessive calculations
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self._calculate_scenario)
            self.update_timer.setSingleShot(True)

            # Initial calculation
            self._update_inputs()
        except Exception as e:
            print(f"Error initializing What-If Turbo panel: {e}")
            import traceback
            traceback.print_exc()
            # Create minimal fallback UI
            self._setup_minimal_ui()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("What-If Turbo Analysis")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            padding: 10px;
            background: #ecf0f1;
            border-radius: 5px;
        """)
        layout.addWidget(title)

        # Input controls
        layout.addWidget(self._create_input_controls())

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # KPI display
        layout.addWidget(self._create_kpi_display())

        # Waterfall chart
        layout.addWidget(self._create_chart_section())

        # Action buttons
        layout.addWidget(self._create_action_buttons())

        layout.addStretch()

    def _create_input_controls(self) -> QGroupBox:
        """Create input control section."""
        group = QGroupBox("Scenario Inputs")
        layout = QGridLayout(group)

        # Room Rate Slider
        layout.addWidget(QLabel("Room Rate ($):"), 0, 0)
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 500)
        self.rate_slider.setValue(150)
        self.rate_label = QLabel("$150")
        self.rate_label.setMinimumWidth(60)

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

        # Promotions
        layout.addWidget(QLabel("Promotions:"), 4, 0)
        self.promotion_combo = QComboBox()
        self.promotion_combo.addItems([
            "None",
            "10% Discount",
            "Weekend Special",
            "Corporate Rate"
        ])
        layout.addWidget(self.promotion_combo, 4, 1)

        # Active promotion checkbox
        self.promotion_active = QCheckBox("Apply Promotion")
        layout.addWidget(self.promotion_active, 5, 0, 1, 2)

        return group

    def _create_kpi_display(self) -> QGroupBox:
        """Create KPI display section."""
        group = QGroupBox("Key Performance Indicators")
        layout = QGridLayout(group)

        # KPI labels
        self.revpar_label = QLabel("RevPAR: $0.00")
        self.goppar_label = QLabel("GOPPAR: $0.00")
        self.profit_delta_label = QLabel("Profit Δ: $0.00")
        self.occupancy_rooms_label = QLabel("Occupied Rooms: 0")

        # Style KPI labels
        kpi_style = """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #3498db;
                border-radius: 5px;
                background: #ecf0f1;
                color: black;
            }
        """

        for label in [self.revpar_label, self.goppar_label,
                      self.profit_delta_label, self.occupancy_rooms_label]:
            label.setStyleSheet(kpi_style)
            label.setMinimumWidth(180)

        layout.addWidget(self.revpar_label, 0, 0)
        layout.addWidget(self.goppar_label, 0, 1)
        layout.addWidget(self.profit_delta_label, 1, 0)
        layout.addWidget(self.occupancy_rooms_label, 1, 1)

        return group

    def _create_chart_section(self) -> QGroupBox:
        """Create waterfall chart section."""
        group = QGroupBox("Profit Impact Analysis")
        layout = QVBoxLayout(group)

        # Chart view - fallback to QLabel if QWebEngineView is not available
        if WEBENGINE_AVAILABLE and QWebEngineView:
            self.chart_view = QWebEngineView()
            self.chart_view.setMinimumHeight(300)
            self.chart_view.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )

            # Enable JavaScript and remote content
            settings = self.chart_view.settings()
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
                True
            )
        else:
            # Fallback to QLabel when QWebEngineView is not available
            self.chart_view = QLabel("Chart functionality requires QWebEngineView")
            self.chart_view.setMinimumHeight(300)
            self.chart_view.setAlignment(Qt.AlignCenter)
            self.chart_view.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    border: 2px dashed #dee2e6;
                    border-radius: 5px;
                    color: #6c757d;
                    font-size: 14px;
                }
            """)

        layout.addWidget(self.chart_view)
        return group

    def _create_action_buttons(self) -> QGroupBox:
        """Create action buttons section."""
        group = QGroupBox("Actions")
        layout = QHBoxLayout(group)

        # Save JSON button
        self.save_json_btn = QPushButton("Save JSON")
        self.save_json_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #229954; }
        """)

        # Export Excel button
        self.export_excel_btn = QPushButton("Export Excel")
        self.export_excel_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2471a3; }
        """)

        # Export PDF button
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #cb4335; }
        """)

        layout.addWidget(self.save_json_btn)
        layout.addWidget(self.export_excel_btn)
        layout.addWidget(self.export_pdf_btn)

        return group

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        try:
            # Input controls
            if hasattr(self, 'rate_slider'):
                self.rate_slider.valueChanged.connect(self._on_rate_changed)
            if hasattr(self, 'occupancy_spin'):
                self.occupancy_spin.valueChanged.connect(self._trigger_update)
            if hasattr(self, 'housekeeping_spin'):
                self.housekeeping_spin.valueChanged.connect(self._trigger_update)
            if hasattr(self, 'fb_spin'):
                self.fb_spin.valueChanged.connect(self._trigger_update)
            if hasattr(self, 'promotion_combo'):
                self.promotion_combo.currentTextChanged.connect(self._trigger_update)
            if hasattr(self, 'promotion_active'):
                self.promotion_active.toggled.connect(self._trigger_update)

            # Action buttons
            if hasattr(self, 'save_json_btn'):
                self.save_json_btn.clicked.connect(self._save_json)
            if hasattr(self, 'export_excel_btn'):
                self.export_excel_btn.clicked.connect(self._export_excel)
            if hasattr(self, 'export_pdf_btn'):
                self.export_pdf_btn.clicked.connect(self._export_pdf)
        except Exception as e:
            print(f"Error connecting signals: {e}")

    def _load_baseline_data(self) -> None:
        """Load baseline data with offline-first approach."""
        try:
            # Try to get current data first
            try:
                self.baseline_data = get_df()
                if not self.baseline_data.empty:
                    print("✓ Loaded baseline data from current DataFrame")
                    self._extract_room_counts()
                    return
            except Exception as e:
                print(f"Warning: Could not load current DataFrame: {e}")

            # Try Oracle Cloud
            try:
                self.baseline_data = fetch_baseline_data()
                if not self.baseline_data.empty:
                    print("✓ Loaded baseline data from Oracle Cloud")
                    self._extract_room_counts()
                    self._cache_baseline_data()
                    return
            except Exception as e:
                print(f"Warning: Could not load from Oracle Cloud: {e}")

            # Fallback to local server
            try:
                self.baseline_data = get_baseline_data()
                if not self.baseline_data.empty:
                    print("✓ Loaded baseline data from local server")
                    self._extract_room_counts()
                    return
            except Exception as e:
                print(f"Warning: Could not load from local server: {e}")

            # Create default data if all else fails
            print("Creating default baseline data")
            self.baseline_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'revenue': np.random.uniform(1000, 2000, 30),
                'occupancy': np.random.uniform(0.6, 0.9, 30),
                'rate': np.random.uniform(120, 180, 30),
                'room_type': np.random.choice(self.room_types, 30)
            })
            self._extract_room_counts()

        except Exception as e:
            print(f"Error loading baseline data: {e}")
            # Create minimal fallback data
            self.baseline_data = pd.DataFrame({
                'date': [pd.Timestamp.now()],
                'revenue': [1500.0],
                'occupancy': [0.75],
                'rate': [150.0],
                'room_type': ['Standard']
            })
            self.room_counts = {rt: 1 for rt in self.room_types}

    def _extract_room_counts(self) -> None:
        """Extract room counts from baseline data."""
        try:
            if 'room_type' in self.baseline_data.columns:
                room_counts = self.baseline_data['room_type'].value_counts()
                self.room_counts = room_counts.to_dict()
            else:
                # Default room counts
                self.room_counts = {
                    "Standard": 50,
                    "Deluxe": 30,
                    "Suite": 15,
                    "Presidential": 5
                }
        except Exception as e:
            print(f"Error extracting room counts: {e}")
            self.room_counts = {
                "Standard": 50,
                "Deluxe": 30,
                "Suite": 15,
                "Presidential": 5
            }

    def _cache_baseline_data(self) -> None:
        """Cache baseline data (Redis if available, in-memory fallback)."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.set('baseline_data', self.baseline_data.to_json())
        except (ImportError, Exception):
            # In-memory caching fallback
            pass

    def _on_rate_changed(self, value: int) -> None:
        """Handle room rate slider change."""
        try:
            if hasattr(self, 'rate_label'):
                self.rate_label.setText(f"${value}")
            self._trigger_update()
        except Exception as e:
            print(f"Error handling rate change: {e}")

    def _trigger_update(self) -> None:
        """Trigger scenario update with debouncing."""
        try:
            if hasattr(self, 'update_timer'):
                self.update_timer.start(300)  # 300ms delay
        except Exception as e:
            print(f"Error triggering update: {e}")

    def _update_inputs(self) -> None:
        """Update current scenario inputs."""
        try:
            self.current_scenario = {
                'room_rate': getattr(self.rate_slider, 'value', lambda: 150)(),
                'occupancy_pct': getattr(self.occupancy_spin, 'value', lambda: 75.0)() if hasattr(self, 'occupancy_spin') else 75.0,
                'housekeeping_staff': getattr(self.housekeeping_spin, 'value', lambda: 10)() if hasattr(self, 'housekeeping_spin') else 10,
                'fb_staff': getattr(self.fb_spin, 'value', lambda: 8)() if hasattr(self, 'fb_spin') else 8,
                'promotion_active': getattr(self.promotion_active, 'isChecked', lambda: False)() if hasattr(self, 'promotion_active') else False,
            }
        except Exception as e:
            print(f"Error updating inputs: {e}")
            # Fallback to default values
            self.current_scenario = {
                'room_rate': 150.0,
                'occupancy_pct': 75.0,
                'housekeeping_staff': 10,
                'fb_staff': 8,
                'promotion_active': False,
            }

    def _calculate_scenario(self) -> None:
        """Calculate scenario KPIs and update display."""
        try:
            self._update_inputs()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)

            # Calculate baseline KPIs if not already done
            if not self.baseline_kpis:
                baseline_inputs = {
                    'room_rate': 150.0,
                    'occupancy_pct': 75.0,
                    'housekeeping_staff': 10,
                    'fb_staff': 8,
                    'promotion_active': False,
                }
                try:
                    self.baseline_kpis = self._simulate_30day_pl(
                        baseline_inputs
                    )
                except Exception as e:
                    print(f"Error calculating baseline KPIs: {e}")
                    # Create default baseline KPIs
                    self.baseline_kpis = {
                        'revpar': 112.5,
                        'goppar': 67.5,
                        'profit_delta': 0.0,
                        'occupancy_rooms': 75.0,
                        'total_revenue': 3375.0,
                        'total_profit': 2025.0,
                    }
            self.progress_bar.setValue(30)

            # Calculate scenario KPIs
            try:
                scenario_kpis = self._simulate_30day_pl(
                    self.current_scenario
                )
            except Exception as e:
                print(f"Error calculating scenario KPIs: {e}")
                # Create default scenario KPIs based on inputs
                rate = self.current_scenario.get('room_rate', 150)
                occ = self.current_scenario.get('occupancy_pct', 75) / 100
                revpar = rate * occ
                scenario_kpis = {
                    'revpar': revpar,
                    'goppar': revpar * 0.6,  # Assume 60% flow-through
                    'profit_delta': (revpar - 112.5) * 0.6 * 30,  # 30 days
                    'occupancy_rooms': occ * 100,  # Assume 100 rooms
                    'total_revenue': revpar * 30 * 100,
                    'total_profit': revpar * 0.6 * 30 * 100,
                }
            self.progress_bar.setValue(70)

            # Update KPI labels
            self._update_kpi_display(scenario_kpis)

            # Update waterfall chart
            self._update_waterfall_chart(scenario_kpis)

            # Emit signal
            self.scenario_changed.emit(scenario_kpis)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)

        except Exception as e:
            print(f"Error calculating scenario: {e}")
            traceback.print_exc()
            # Show error in UI
            self.revpar_label.setText("RevPAR: Error")
            self.goppar_label.setText("GOPPAR: Error")
            self.profit_delta_label.setText("Profit Δ: Error")
            self.occupancy_rooms_label.setText("Occupied Rooms: Error")
            self.progress_bar.setVisible(False)

    def _simulate_30day_pl(self, inputs: Dict) -> Dict:
        """Simulate 30-day P&L with realistic financial model."""
        # Base values
        base_rate = inputs['room_rate']
        occupancy = inputs['occupancy_pct'] / 100
        housekeeping = inputs['housekeeping_staff']
        fb_staff = inputs['fb_staff']
        promotion = inputs['promotion_active']
        
        # Calculate demand elasticity (price sensitivity)
        price_change = (base_rate - 150) / 150  # % change from baseline
        demand_factor = max(0.7, 1 - 0.8 * abs(price_change))  # Elasticity factor
        
        # Adjust occupancy based on price and promotion
        adj_occupancy = occupancy * demand_factor
        if promotion:
            adj_occupancy *= 1.15  # 15% boost from promotion
        
        # Revenue calculation with leakage
        total_rooms = sum(self.room_counts.values())
        occupied_rooms = adj_occupancy * total_rooms
        revenue = occupied_rooms * base_rate * 30  # 30 days
        
        # Apply leakage (revenue not captured)
        leakage = 0.05 * revenue  # 5% leakage
        net_revenue = revenue - leakage
        
        # Cost calculations
        # Fixed costs (30% of revenue)
        fixed_costs = 0.3 * net_revenue
        
        # Variable costs
        # Staff costs with diminishing returns
        hk_cost = 2500 * housekeeping * (1 - 0.02 * max(0, housekeeping - 10))
        fb_cost = 3000 * fb_staff * (1 - 0.015 * max(0, fb_staff - 8))
        
        # Promotion cost
        promo_cost = 0.1 * net_revenue if promotion else 0
        
        total_costs = fixed_costs + hk_cost + fb_cost + promo_cost
        
        # Profit calculation
        profit = net_revenue - total_costs
        
        # Key metrics
        revpar = net_revenue / (total_rooms * 30)
        goppar = profit / (total_rooms * 30)
        
        return {
            'revpar': revpar,
            'goppar': goppar,
            'profit_delta': profit - self.baseline_kpis.get('total_profit', 0),
            'occupancy_rooms': occupied_rooms,
            'total_revenue': net_revenue,
            'total_profit': profit,
            'total_costs': total_costs
        }

    def _update_kpi_display(self, kpis: Dict[str, float]) -> None:
        """Update KPI display labels."""
        try:
            if hasattr(self, 'revpar_label'):
                self.revpar_label.setText(f"RevPAR: ${kpis.get('revpar', 0):.2f}")
            if hasattr(self, 'goppar_label'):
                self.goppar_label.setText(f"GOPPAR: ${kpis.get('goppar', 0):.2f}")

            profit_delta = kpis.get('profit_delta', 0)
            delta_color = "#28a745" if profit_delta >= 0 else "#e74c3c"
            delta_sign = "+" if profit_delta >= 0 else ""

            if hasattr(self, 'profit_delta_label'):
                self.profit_delta_label.setText(
                    f"Profit Δ: {delta_sign}${profit_delta:,.2f}"
                )
                self.profit_delta_label.setStyleSheet(f"""
                    QLabel {{
                        font-size: 14px;
                        font-weight: bold;
                        padding: 8px;
                        border: 2px solid {delta_color};
                        border-radius: 5px;
                        background: #ecf0f1;
                        color: black;
                    }}
                """)

            if hasattr(self, 'occupancy_rooms_label'):
                self.occupancy_rooms_label.setText(
                    f"Occupied Rooms: {kpis.get('occupancy_rooms', 0):.1f}"
                )
        except Exception as e:
            print(f"Error updating KPI display: {e}")

    def _update_waterfall_chart(self, scenario_kpis: Dict[str, float]) -> None:
        """Update waterfall chart with profit analysis."""
        try:
            if not self.baseline_kpis:
                return

            # Calculate waterfall data
            baseline = self.baseline_kpis['total_profit']
            revenue_impact = scenario_kpis['total_revenue'] - self.baseline_kpis['total_revenue']
            cost_impact = -(scenario_kpis['total_costs'] - self.baseline_kpis.get('total_costs', 0))
            final = scenario_kpis['total_profit']
            
            # Create categories and values
            categories = ["Baseline", "Revenue Impact", "Cost Impact", "Final"]
            values = [baseline, revenue_impact, cost_impact, final]
            measures = ["absolute", "relative", "relative", "total"]
            
            # Create colors
            colors = ["#3498db", 
                      "#2ecc71" if revenue_impact >= 0 else "#e74c3c",
                      "#2ecc71" if cost_impact >= 0 else "#e74c3c",
                      "#9b59b6"]

            # Create Plotly waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Profit Analysis",
                orientation="v",
                measure=measures,
                x=categories,
                textposition="outside",
                text=[f"${v:,.0f}" for v in values],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#9b59b6"}},
            ))

            # Add annotations
            fig.add_annotation(
                x=1, y=final,
                text=f"Final Profit: ${final:,.0f}",
                showarrow=True,
                arrowhead=4,
                ax=0,
                ay=-40
            )

            # Update layout
            fig.update_layout(
                title="Profit Impact Waterfall",
                showlegend=False,
                template="plotly_white",
                height=350,
                margin=dict(l=0, r=0, t=50, b=20),
                xaxis_title="Category",
                yaxis_title="Amount ($)",
                hovermode="x unified"
            )

            # Display chart based on available widget type
            if WEBENGINE_AVAILABLE and hasattr(self.chart_view, 'setHtml'):
                # Convert to HTML and display in QWebEngineView
                html_string = StringIO()
                fig.write_html(html_string, include_plotlyjs="cdn", full_html=False)
                self.chart_view.setHtml(html_string.getvalue())
            else:
                # Fallback: display summary in QLabel
                profit_delta = scenario_kpis.get('profit_delta', 0)
                delta_sign = "+" if profit_delta >= 0 else ""
                summary_text = f"Profit Impact: {delta_sign}${profit_delta:,.2f}"
                self.chart_view.setText(summary_text)

        except Exception as e:
            print(f"Error updating waterfall chart: {e}")
            # Show error message in chart view
            if hasattr(self.chart_view, 'setHtml'):
                error_html = f"""
                <html><body>
                <div style="text-align: center; padding: 50px;">
                    <h3>Chart Error</h3>
                    <p>Unable to generate waterfall chart: {str(e)}</p>
                </div>
                </body></html>
                """
                self.chart_view.setHtml(error_html)
            else:
                self.chart_view.setText(f"Chart Error: {str(e)}")

    def _save_json(self) -> None:
        """Save scenario to JSON file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Scenario",
                f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
                "JSON Files (*.json)"
            )

            if filename:
                scenario_data = {
                    'inputs': self.current_scenario,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'baseline_kpis': self.baseline_kpis,
                    'scenario_kpis': self._simulate_30day_pl(
                        self.current_scenario
                    ),
                }

                with open(filename, 'w') as f:
                    json.dump(scenario_data, f, indent=2)

                QMessageBox.information(
                    self, "Success", f"Scenario saved to {filename}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save scenario: {str(e)}"
            )

    def _export_excel(self) -> None:
        """Export scenario to Excel file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export to Excel",
                f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "Excel Files (*.xlsx)"
            )

            if filename:
                scenario_kpis = self._simulate_30day_pl(
                    self.current_scenario
                )

                # Create DataFrame for export
                data = {
                    'Metric': list(scenario_kpis.keys()),
                    'Value': list(scenario_kpis.values()),
                }
                df = pd.DataFrame(data)

                # Export to Excel
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Scenario KPIs', index=False)

                    inputs_df = pd.DataFrame([self.current_scenario])
                    inputs_df.to_excel(
                        writer, sheet_name='Inputs', index=False
                    )

                QMessageBox.information(
                    self, "Success", f"Scenario exported to {filename}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to export to Excel: {str(e)}"
            )

    def _export_pdf(self) -> None:
        """Export chart to PDF file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export to PDF",
                f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                "PDF Files (*.pdf)"
            )

            if filename:
                if WEBENGINE_AVAILABLE and hasattr(self.chart_view, 'page'):
                    # Use QWebEngineView's printToPdf functionality
                    self.chart_view.page().printToPdf(filename)
                    QMessageBox.information(
                        self, "Success", f"Chart exported to {filename}"
                    )
                else:
                    # Fallback: inform user that PDF export requires QWebEngineView
                    QMessageBox.information(
                        self, "PDF Export",
                        "PDF export requires QWebEngineView. Please use Excel export instead."
                    )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to export to PDF: {str(e)}"
            )

    def _setup_minimal_ui(self) -> None:
        """Setup minimal UI when full UI fails."""
        layout = QVBoxLayout(self)

        title = QLabel("What-If Turbo Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        error_label = QLabel("UI initialization failed. Basic mode active.")
        error_label.setStyleSheet("color: orange; padding: 10px;")
        layout.addWidget(error_label)

        # Basic controls
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 500)
        self.rate_slider.setValue(150)
        layout.addWidget(QLabel("Room Rate:"))
        layout.addWidget(self.rate_slider)

        # Basic KPI display
        self.revpar_label = QLabel("RevPAR: $0.00")
        self.goppar_label = QLabel("GOPPAR: $0.00")
        layout.addWidget(self.revpar_label)
        layout.addWidget(self.goppar_label)

        layout.addStretch()


@data_required
def display() -> QWidget:
    """Create and return What-If Turbo panel widget."""
    try:
        return WhatIfTurboPanel()
    except Exception as e:
        print(f"Error creating What-If Turbo panel: {e}")
        import traceback
        traceback.print_exc()

        # Return a simple error widget as fallback
        from PySide6.QtWidgets import QLabel
        error_widget = QLabel(f"Error loading What-If Turbo: {str(e)}")
        error_widget.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-size: 14pt;
                padding: 20px;
                background-color: #2A232A;
                border-radius: 10px;
            }
        """)
        return error_widget