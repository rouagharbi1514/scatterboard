# views/housekeeping.py
"""
Housekeeping Analysis
─────────────────────
This view has been corrected to use the specific column names from your dataset.
It provides a comprehensive analysis of housekeeping operations, including
staffing, efficiency, room-specific metrics, and laundry performance.

Tabs
––––
• Dashboard: Visualizes staff count vs. rooms serviced and tracks the key
             rooms-per-staff efficiency metric.
• Efficiency: Analyzes housekeeping expenses and cost per occupied room (CPOR).
• Room Analysis: Shows the number of rooms serviced and revenue generated,
                 broken down by room type.
• Laundry: A detailed KPI dashboard for laundry revenue, profit, margins,
           and efficiency metrics.
"""

from __future__ import annotations

import traceback
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDateEdit,
    QComboBox,
    QTabWidget,
    QSizePolicy,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings

from data.helpers import get_df
from views.utils import data_required

# ──────────────────────────────────────────────────────────────
# Reusable Helper Widgets & Functions
# ──────────────────────────────────────────────────────────────


def _plotly_view(fig) -> QWebEngineView:
    """Returns a Qt web-view widget with an embedded Plotly figure."""
    from io import StringIO

    html = StringIO()
    # Write the figure to an HTML string, using the CDN for the Plotly.js library
    fig.write_html(html, include_plotlyjs="cdn", full_html=False)

    # Create a QWebEngineView to display the HTML
    view = QWebEngineView()
    view.setHtml(html.getvalue())
    # Allow the view to access remote URLs (for the Plotly.js CDN)
    view.settings().setAttribute(
        QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
    )
    view.setMinimumHeight(450)
    view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return view


def _perf_label(p: float) -> str:
    """Classifies performance as 'strong', 'moderate', or 'weak' based on a percentage value."""
    if abs(p) >= 30:
        return "strong"
    elif abs(p) >= 15:
        return "moderate"
    else:
        return "weak"


def _collapsible(text: str) -> QWidget:
    """Creates a collapsible widget with a button to show/hide explanation text."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 4, 0, 8)

    btn = QPushButton("Show explanation")
    btn.setFixedWidth(150)
    btn.setStyleSheet(
        "QPushButton{background:#4a86e8;color:#fff;padding:5px 10px;border-radius:4px}"
        "QPushButton:hover{background:#3a76d8}"
    )

    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    lbl.setStyleSheet(
        "background:rgba(25,45,90,0.92);color:white;padding:12px;border-radius:5px;"
    )

    def _toggle():
        """Toggles the visibility of the explanation label."""
        is_visible = not lbl.isVisible()
        lbl.setVisible(is_visible)
        btn.setText("Hide explanation" if is_visible else "Show explanation")

    btn.clicked.connect(_toggle)
    lay.addWidget(btn, alignment=Qt.AlignLeft)
    lay.addWidget(lbl)
    return box

# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

@data_required
def display() -> QWidget:
    """
    Main function to build and display the housekeeping analysis dashboard.
    It loads data, sets up UI controls, and orchestrates the rendering of charts.
    """
    try:
        # --- 1. Data Loading and Validation ---
        df = get_df()
        if df is None or df.empty:
            raise ValueError("Dataset is empty or could not be loaded.")
        df = df.copy()

        # Define the mapping of required columns to the names used in your dataset
        # This makes the script adaptable to your specific data schema.
        COLUMN_MAP = {
            "Date": "Date",
            "RoomsServiced": "OccupiedRooms",
            "StaffCount": "HousekeepingStaffCount",
            "Expenses": "HousekeepingExpenses",
            "HoursSpent": "HousekeepingHoursSpent",
            "LaundryRevenue": "LaundryRevenue",
            "LaundryExpenses": "LaundryExpenses",
            "LaundryItems": "LaundryItemsProcessed",
            "ADR": "ADR",
            "RoomTypesOccupied": {
                "Single": "SingleRoomsOccupied",
                "Double": "DoubleRoomsOccupied",
                "Family": "FamilyRoomsOccupied",
                "Royal": "RoyalRoomsOccupied",
            },
            "RoomTypesRevenue": {
                "Single": "SingleRoomRevenue",
                "Double": "DoubleRoomRevenue",
                "Family": "FamilyRoomRevenue",
                "Royal": "RoyalRoomRevenue",
            },
        }

        # Validate that all essential columns exist in the DataFrame
        required_cols = [
            COLUMN_MAP["Date"],
            COLUMN_MAP["RoomsServiced"],
            COLUMN_MAP["StaffCount"],
            COLUMN_MAP["Expenses"],
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing essential columns: {', '.join(missing_cols)}")

        # Ensure Date column is in datetime format
        date_col = COLUMN_MAP["Date"]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().any():
            raise ValueError(f"Could not parse all dates in '{date_col}' column.")

        # --- 2. UI Setup ---
        root = QWidget()
        root.setLayout(QVBoxLayout())

        title = QLabel("Housekeeping Analysis")
        title.setStyleSheet("font-size:18pt;font-weight:bold;margin-bottom:6px;")
        root.layout().addWidget(title)

        # Date pickers default to the full range of the data
        d_min, d_max = df[date_col].min().date(), df[date_col].max().date()
        start_pick = QDateEdit(QDate(d_min.year, d_min.month, d_min.day))
        end_pick = QDateEdit(QDate(d_max.year, d_max.month, d_max.day))
        for p in (start_pick, end_pick):
            p.setCalendarPopup(True)

        # Controls for filtering and aggregation
        period_sel = QComboBox()
        period_sel.addItems(["Daily", "Weekly", "Monthly"])
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(80)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Date range:"))
        filter_layout.addWidget(start_pick)
        filter_layout.addWidget(QLabel("to"))
        filter_layout.addWidget(end_pick)
        filter_layout.addSpacing(16)
        filter_layout.addWidget(QLabel("Period:"))
        filter_layout.addWidget(period_sel)
        filter_layout.addWidget(apply_btn)
        filter_layout.addStretch()
        root.layout().addLayout(filter_layout)

        tabs = QTabWidget()
        root.layout().addWidget(tabs)

        # --- 3. Render Logic ---
        def _render():
            """
            This function is called when the 'Apply' button is clicked.
            It filters the data, performs calculations, and generates the charts.
            """
            try:
                # Filter data based on date picker values
                begin = pd.Timestamp(start_pick.date().toPython())
                finish = pd.Timestamp(end_pick.date().toPython())
                sub_df = df[(df[date_col] >= begin) & (df[date_col] <= finish)].copy()

                tabs.clear()
                if sub_df.empty:
                    warn = QLabel("No data in selected date range")
                    warn.setAlignment(Qt.AlignCenter)
                    tabs.addTab(warn, "Info")
                    return

                # Determine aggregation frequency
                period = period_sel.currentText()
                freq = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[period]
                grouper = pd.Grouper(key=COLUMN_MAP["Date"], freq=freq)

                # --- Pre-calculate essential metrics ---
                sub_df["RoomsPerStaff"] = sub_df[COLUMN_MAP["RoomsServiced"]] / sub_df[COLUMN_MAP["StaffCount"]]
                sub_df["CostPerRoom"] = sub_df[COLUMN_MAP["Expenses"]] / sub_df[COLUMN_MAP["RoomsServiced"]]
                # Calculate cleaning time in minutes per room
                if COLUMN_MAP['HoursSpent'] in sub_df.columns:
                    sub_df["CleaningTime"] = (sub_df[COLUMN_MAP['HoursSpent']] * 60) / sub_df[COLUMN_MAP["RoomsServiced"]]


                # --- Tab 1: Dashboard ---
                g = sub_df.groupby(grouper).agg(
                    Staff=(COLUMN_MAP["StaffCount"], "mean"),
                    Rooms=(COLUMN_MAP["RoomsServiced"], "sum"),
                    RPS=("RoomsPerStaff", "mean")
                ).reset_index()

                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                fig1.add_trace(go.Bar(x=g[date_col], y=g["Staff"], name="Staff", marker_color="#3b82f6"))
                fig1.add_trace(go.Scatter(x=g[date_col], y=g["Rooms"], mode="lines+markers", name="Rooms Serviced", marker_color="#ef4444"), secondary_y=True)
                fig1.add_trace(go.Scatter(x=g[date_col], y=g["RPS"], mode="lines+markers", name="Rooms/Staff", line=dict(dash="dash", color="#10b981")), secondary_y=True)
                fig1.update_layout(title="Staffing vs. Rooms Serviced", template="plotly_white", hovermode="x unified")
                fig1.update_yaxes(title_text="Avg Staff Count", secondary_y=False)
                fig1.update_yaxes(title_text="Total Rooms & Ratio", secondary_y=True)

                dash_tab = QWidget()
                dash_tab.setLayout(QVBoxLayout())
                dash_tab.layout().addWidget(_plotly_view(fig1))
                avg_rps = sub_df["RoomsPerStaff"].mean()
                # Assuming a target of 14 rooms per staff
                diff_rps = (avg_rps / 14 - 1) * 100
                dash_tab.layout().addWidget(
                    _collapsible(
                        f"Average rooms per staff is <b>{avg_rps:.1f}</b>, which is "
                        f"<b>{abs(diff_rps):.1f}%</b> {'above' if diff_rps > 0 else 'below'} the target of 14. "
                        f"This indicates <b>{_perf_label(diff_rps)}</b> staff utilisation."
                    )
                )
                tabs.addTab(dash_tab, "Dashboard")

                # --- Tab 2: Efficiency Analysis ---
                g2 = sub_df.groupby(grouper).agg(
                    Cost=(COLUMN_MAP["Expenses"], "sum"),
                    CPR=("CostPerRoom", "mean")
                ).reset_index()

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Bar(x=g2[date_col], y=g2["Cost"], name="Total Cost", marker_color="#9333ea"))
                fig2.add_trace(go.Scatter(x=g2[date_col], y=g2["CPR"], mode="lines+markers", name="Cost/Room", marker_color="#f97316"), secondary_y=True)
                fig2.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Target CPOR: $30", secondary_y=True)
                fig2.update_layout(title="Cost Efficiency Analysis", template="plotly_white", hovermode="x unified")
                fig2.update_yaxes(title_text="Total Housekeeping Cost", secondary_y=False)
                fig2.update_yaxes(title_text="Cost per Occupied Room (CPOR)", secondary_y=True)

                eff_tab = QWidget()
                eff_tab.setLayout(QVBoxLayout())
                eff_tab.layout().addWidget(_plotly_view(fig2))
                avg_cpr = sub_df["CostPerRoom"].mean()
                diff_cpr = (avg_cpr / 30 - 1) * 100
                eff_tab.layout().addWidget(
                    _collapsible(
                        f"Average Cost Per Room (CPOR) is <b>${avg_cpr:.2f}</b>. This is "
                        f"<b>{abs(diff_cpr):.1f}%</b> {'over' if diff_cpr > 0 else 'under'} the $30 target, "
                        f"representing <b>{_perf_label(diff_cpr)}</b> cost control."
                    )
                )
                tabs.addTab(eff_tab, "Efficiency Analysis")

                # --- Tab 3: Room Analysis ---
                room_occ_cols = list(COLUMN_MAP["RoomTypesOccupied"].values())
                room_rev_cols = list(COLUMN_MAP["RoomTypesRevenue"].values())

                if all(c in sub_df.columns for c in room_occ_cols + room_rev_cols):
                    room_data = []
                    for rt_name, occ_col in COLUMN_MAP["RoomTypesOccupied"].items():
                        rev_col = COLUMN_MAP["RoomTypesRevenue"][rt_name]
                        room_data.append({
                            "RoomType": rt_name,
                            "RoomsServiced": sub_df[occ_col].sum(),
                            "TotalRevenue": sub_df[rev_col].sum()
                        })
                    ra_df = pd.DataFrame(room_data)
                    ra_df = ra_df[ra_df["RoomsServiced"] > 0]
                    ra_df["RevenuePerRoom"] = ra_df["TotalRevenue"] / ra_df["RoomsServiced"]

                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(go.Bar(x=ra_df["RoomType"], y=ra_df["RoomsServiced"], name="Rooms Serviced", marker_color="#2dd4bf"))
                    fig3.add_trace(go.Scatter(x=ra_df["RoomType"], y=ra_df["RevenuePerRoom"], mode="lines+markers", name="Revenue/Room", marker_color="#facc15"), secondary_y=True)
                    fig3.update_layout(
                        title="Performance by Room Type", template="plotly_white",
                        yaxis=dict(title="Total Rooms Serviced"),
                        yaxis2=dict(title="Revenue per Room Serviced ($)"),
                        xaxis_title="Room Type"
                    )

                    room_tab = QWidget()
                    room_tab.setLayout(QVBoxLayout())
                    room_tab.layout().addWidget(_plotly_view(fig3))

                    if not ra_df.empty:
                        most_serviced = ra_df.loc[ra_df["RoomsServiced"].idxmax()]
                        most_profitable_room = ra_df.loc[ra_df["RevenuePerRoom"].idxmax()]
                        room_tab.layout().addWidget(
                            _collapsible(
                                f"<b>{most_serviced['RoomType']}</b> rooms were the most frequently serviced ({most_serviced['RoomsServiced']:,} rooms).<br>"
                                f"<b>{most_profitable_room['RoomType']}</b> rooms were the most profitable per room, generating <b>${most_profitable_room['RevenuePerRoom']:,.2f}</b> on average."
                            )
                        )
                    tabs.addTab(room_tab, "Room Analysis")


                # --- Tab 4: Laundry ---
                laundry_cols = [
                    COLUMN_MAP["LaundryRevenue"], COLUMN_MAP["LaundryExpenses"],
                    COLUMN_MAP["LaundryItems"], COLUMN_MAP["RoomsServiced"], COLUMN_MAP["ADR"]
                ]
                laundry_tab = QWidget()
                laundry_tab.setLayout(QVBoxLayout())
                if not all(c in sub_df.columns for c in laundry_cols):
                    warn = QLabel("Required laundry data not available.")
                    warn.setAlignment(Qt.AlignCenter)
                    laundry_tab.layout().addWidget(warn)
                else:
                    # Calculate laundry KPIs
                    sub_df["LaundryProfit"] = sub_df[COLUMN_MAP["LaundryRevenue"]] - sub_df[COLUMN_MAP["LaundryExpenses"]]

                    # Aggregate totals for KPI calculations
                    total_laundry_rev = sub_df[COLUMN_MAP["LaundryRevenue"]].sum()
                    total_laundry_prof = sub_df["LaundryProfit"].sum()
                    total_rooms_occ = sub_df[COLUMN_MAP["RoomsServiced"]].sum()
                    total_laundry_items = sub_df[COLUMN_MAP["LaundryItems"]].sum()
                    mean_adr = sub_df[COLUMN_MAP["ADR"]].mean()

                    kpi_margin = (total_laundry_prof / total_laundry_rev * 100) if total_laundry_rev else 0
                    kpi_cpor = (total_laundry_rev / total_rooms_occ) if total_rooms_occ else 0
                    kpi_uplift = (kpi_cpor / mean_adr * 100) if mean_adr else 0
                    kpi_ipr = (total_laundry_items / total_rooms_occ) if total_rooms_occ else 0

                    # KPI strip
                    kpi_box = QHBoxLayout()
                    kpi_values = [
                        ("Total Revenue", f"${total_laundry_rev:,.0f}"),
                        ("Total Profit", f"${total_laundry_prof:,.0f}"),
                        ("Profit Margin", f"{kpi_margin:.1f}%"),
                        ("Cost Per Room", f"${kpi_cpor:.2f}"),
                        ("RevPAR Uplift", f"{kpi_uplift:.1f}%"),
                        ("Items / Room", f"{kpi_ipr:.1f}"),
                    ]
                    for title, val in kpi_values:
                        card = QWidget()
                        cl = QVBoxLayout(card)
                        ttl_label = QLabel(title)
                        ttl_label.setStyleSheet("color:#888;font-size:9pt")
                        val_label = QLabel(val)
                        val_label.setStyleSheet("font-size:13pt;font-weight:bold")
                        cl.addWidget(ttl_label)
                        cl.addWidget(val_label)
                        cl.setContentsMargins(8, 4, 8, 4)
                        card.setStyleSheet("background:rgba(74,134,232,0.12);border-radius:6px")
                        kpi_box.addWidget(card)
                    laundry_tab.layout().addLayout(kpi_box)

                    # Time series chart
                    lg = sub_df.groupby(grouper).agg(
                        Rev=(COLUMN_MAP["LaundryRevenue"], "sum"),
                        Prof=("LaundryProfit", "sum")
                    ).reset_index()
                    lg["Margin"] = lg["Prof"] / lg["Rev"] * 100

                    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig4.add_trace(go.Bar(x=lg[date_col], y=lg["Rev"], name="Revenue", marker_color="#3b82f6"))
                    fig4.add_trace(go.Scatter(x=lg[date_col], y=lg["Margin"], name="Margin %", mode="lines+markers", marker_color="#f97316"), secondary_y=True)
                    fig4.update_layout(title="Laundry Revenue & Profit Margin Trend", template="plotly_white", hovermode="x unified")
                    fig4.update_yaxes(title_text="Total Revenue", secondary_y=False)
                    fig4.update_yaxes(title_text="Profit Margin %", secondary_y=True)
                    laundry_tab.layout().addWidget(_plotly_view(fig4))

                    laundry_tab.layout().addWidget(
                        _collapsible(
                            f"Laundry operations generated <b>${total_laundry_rev:,.0f}</b> in revenue with a "
                            f"<b>{kpi_margin:.1f}%</b> profit margin, indicating <b>{_perf_label(kpi_margin)}</b> performance. "
                            f"This added an average of <b>${kpi_cpor:.2f}</b> to each occupied room, uplifting RevPAR by <b>{kpi_uplift:.1f}%</b>."
                        )
                    )
                tabs.addTab(laundry_tab, "Laundry")

            except Exception as e:
                # Gracefully handle any errors during rendering
                traceback.print_exc()
                tabs.clear()
                err_label = QLabel(f"An error occurred during rendering:\n{e}")
                err_label.setStyleSheet("color:red; font-size:12pt;")
                err_label.setAlignment(Qt.AlignCenter)
                tabs.addTab(err_label, "Error")

        # --- 4. Initial Setup and Connections ---
        apply_btn.clicked.connect(_render)
        _render()  # Initial render on load
        return root

    except Exception as e:
        # Catch errors during the initial setup
        traceback.print_exc()
        fail_widget = QWidget()
        fail_layout = QVBoxLayout(fail_widget)
        fail_label = QLabel(f"Failed to initialize Housekeeping view:\n{e}")
        fail_label.setAlignment(Qt.AlignCenter)
        fail_label.setStyleSheet("color:red; font-size:14pt;")
        fail_layout.addWidget(fail_label)
        return fail_widget
