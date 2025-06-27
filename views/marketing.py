# views/marketing.py
"""
Marketing Analysis View
=======================

Shows marketing campaign performance, channel metrics, and ROI analysis using
interactive Plotly charts for a modern and consistent look.
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
    QGridLayout,  # Add this import
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
from views.utils import data_required, kpi_tile

# ──────────────────────────────────────────────────────────────
# Reusable Helper Widgets & Functions
# ──────────────────────────────────────────────────────────────


def _plotly_view(fig) -> QWebEngineView:
    """Returns a Qt web-view widget with an embedded Plotly figure."""
    from io import StringIO

    html = StringIO()
    fig.write_html(html, include_plotlyjs="cdn", full_html=False)

    view = QWebEngineView()
    view.setHtml(html.getvalue())
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
    Main function to build and display the marketing analysis dashboard.
    """
    try:
        # --- 1. Data Loading and Validation ---
        df = get_df()
        if df is None or df.empty:
            raise ValueError("Dataset is empty or could not be loaded.")
        df = df.copy()

        # Define the mapping of required columns to the names used in your dataset
        COLUMN_MAP = {
            "Date": "Date",
            "Revenue": "TotalRevenue",
            "Cost": "MarketingSpend",
            "Campaign": "MarketingCampaignType",
            "Channel": "MarketingChannel",
            "BookingID": "GuestID", # Using GuestID as a proxy for a unique booking
        }

        # Validate that all essential columns exist in the DataFrame
        required_cols = list(COLUMN_MAP.values())
        # 'BookingID' is used for counting bookings, so it's essential
        required_cols.remove("GuestID") # Remove GuestID as it's a value, not a key
        required_cols.append(COLUMN_MAP["BookingID"])

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing essential columns: {', '.join(missing_cols)}")

        # Ensure Date column is in datetime format
        date_col = COLUMN_MAP["Date"]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().any():
            # Drop rows where date could not be parsed
            df.dropna(subset=[date_col], inplace=True)

        # Create a 'Bookings' column for counting
        df['Bookings'] = 1

        # --- 2. UI Setup ---
        root = QWidget()
        root.setLayout(QVBoxLayout())

        title = QLabel("Marketing Analysis")
        title.setStyleSheet("font-size:18pt;font-weight:bold;margin-bottom:6px;")
        root.layout().addWidget(title)

        d_min, d_max = df[date_col].min().date(), df[date_col].max().date()
        start_pick = QDateEdit(QDate(d_min.year, d_min.month, d_min.day))
        end_pick = QDateEdit(QDate(d_max.year, d_max.month, d_max.day))
        for p in (start_pick, end_pick):
            p.setCalendarPopup(True)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(80)

        campaign_combo = QComboBox()
        campaign_combo.addItem("All Campaigns")
        if COLUMN_MAP["Campaign"] in df.columns:
            campaign_combo.addItems(df[COLUMN_MAP["Campaign"]].unique())

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Date Range:"))
        filter_layout.addWidget(start_pick)
        filter_layout.addWidget(QLabel("to"))
        filter_layout.addWidget(end_pick)
        filter_layout.addSpacing(16)
        filter_layout.addWidget(QLabel("Campaign:"))
        filter_layout.addWidget(campaign_combo)
        filter_layout.addWidget(apply_btn)
        filter_layout.addStretch()
        root.layout().addLayout(filter_layout)

        kpi_grid = QGridLayout()
        root.layout().addLayout(kpi_grid)

        tabs = QTabWidget()
        root.layout().addWidget(tabs)

        # --- 3. Render Logic ---
        def _render():
            try:
                begin = pd.Timestamp(start_pick.date().toPython())
                finish = pd.Timestamp(end_pick.date().toPython())
                campaign_filter = campaign_combo.currentText()

                sub_df = df[(df[date_col] >= begin) & (df[date_col] <= finish)]
                if campaign_filter != "All Campaigns":
                    sub_df = sub_df[sub_df[COLUMN_MAP["Campaign"]] == campaign_filter]
                sub_df = sub_df.copy()

                # Clear previous content
                while kpi_grid.count():
                    item = kpi_grid.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                tabs.clear()

                if sub_df.empty:
                    warn = QLabel("No data in selected range/filter")
                    warn.setAlignment(Qt.AlignCenter)
                    tabs.addTab(warn, "Info")
                    return

                # --- Tab 1: Channel Performance ---
                channel_tab = QWidget()
                channel_tab.setLayout(QVBoxLayout())

                channel_metrics = sub_df.groupby(COLUMN_MAP["Channel"]).agg(
                    Bookings=('Bookings', 'sum'),
                    Cost=(COLUMN_MAP["Cost"], 'sum'),
                    Revenue=(COLUMN_MAP["Revenue"], 'sum')
                ).reset_index()
                channel_metrics['ROAS'] = channel_metrics['Revenue'] / channel_metrics['Cost']

                fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Bookings vs Cost", "Return on Ad Spend (ROAS)"))
                fig1.add_trace(go.Bar(name='Bookings', x=channel_metrics[COLUMN_MAP["Channel"]], y=channel_metrics['Bookings'], marker_color='blue'), row=1, col=1)
                fig1.add_trace(go.Bar(name='Cost', x=channel_metrics[COLUMN_MAP["Channel"]], y=channel_metrics['Cost'], marker_color='red'), row=1, col=1)

                roas_sorted = channel_metrics.sort_values('ROAS', ascending=True)
                fig1.add_trace(go.Bar(name='ROAS', x=roas_sorted['ROAS'], y=roas_sorted[COLUMN_MAP["Channel"]], orientation='h'), row=1, col=2)

                fig1.update_layout(title_text="Channel Performance Analysis", barmode='group', template="plotly_white", height=500)
                channel_tab.layout().addWidget(_plotly_view(fig1))
                tabs.addTab(channel_tab, "Channel Performance")

                # --- Tab 2: Campaign Performance ---
                campaign_tab = QWidget()
                campaign_tab.setLayout(QVBoxLayout())

                campaign_metrics = sub_df.groupby(COLUMN_MAP["Campaign"]).agg(
                    Bookings=('Bookings', 'sum'),
                    Cost=(COLUMN_MAP["Cost"], 'sum'),
                    Revenue=(COLUMN_MAP["Revenue"], 'sum')
                ).reset_index()
                campaign_metrics['ROAS'] = campaign_metrics['Revenue'] / campaign_metrics['Cost']

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Bar(name='Revenue', x=campaign_metrics[COLUMN_MAP["Campaign"]], y=campaign_metrics['Revenue'], marker_color='green'), secondary_y=False)
                fig2.add_trace(go.Bar(name='Cost', x=campaign_metrics[COLUMN_MAP["Campaign"]], y=campaign_metrics['Cost'], marker_color='orange'), secondary_y=False)
                fig2.add_trace(
                    go.Scatter(
                        name='ROAS',
                        x=campaign_metrics[COLUMN_MAP["Campaign"]],
                        y=campaign_metrics['ROAS'],
                        mode='lines+markers',
                        marker_color='purple'
                    ),
                    secondary_y=True
                )

                fig2.update_layout(title_text="Campaign Performance (Revenue, Cost, ROAS)", barmode='group', template="plotly_white")
                fig2.update_yaxes(title_text="<b>Primary</b> Amount ($)", secondary_y=False)
                fig2.update_yaxes(title_text="<b>Secondary</b> ROAS (x)", secondary_y=True)
                campaign_tab.layout().addWidget(_plotly_view(fig2))
                tabs.addTab(campaign_tab, "Campaign Performance")

                # --- Tab 3: Trends ---
                trends_tab = QWidget()
                trends_tab.setLayout(QVBoxLayout())

                trend_df = sub_df.groupby(date_col).agg(
                    Bookings=('Bookings', 'sum'),
                    Revenue=(COLUMN_MAP["Revenue"], 'sum'),
                    Cost=(COLUMN_MAP["Cost"], 'sum')
                ).reset_index()
                trend_df['ROAS'] = trend_df['Revenue'] / trend_df['Cost']

                fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                fig3.add_trace(go.Scatter(name='Revenue', x=trend_df[date_col], y=trend_df['Revenue'], mode='lines', line=dict(color='blue')), secondary_y=False)
                fig3.add_trace(go.Scatter(name='Cost', x=trend_df[date_col], y=trend_df['Cost'], mode='lines', line=dict(color='red')), secondary_y=False)
                fig3.add_trace(go.Scatter(name='ROAS', x=trend_df[date_col], y=trend_df['ROAS'], mode='lines', line=dict(color='green', dash='dot')), secondary_y=True)

                fig3.update_layout(title_text="Daily Trends: Revenue, Cost, and ROAS", template="plotly_white")
                fig3.update_yaxes(title_text="Amount ($)", secondary_y=False)
                fig3.update_yaxes(title_text="ROAS (x)", secondary_y=True)
                trends_tab.layout().addWidget(_plotly_view(fig3))
                tabs.addTab(trends_tab, "Trends")

            except Exception as e:
                traceback.print_exc()
                tabs.clear()
                err_label = QLabel(f"An error occurred during rendering:\n{e}")
                err_label.setStyleSheet("color:red; font-size:12pt;")
                err_label.setAlignment(Qt.AlignCenter)
                tabs.addTab(err_label, "Error")

        # --- 4. Initial Setup and Connections ---
        apply_btn.clicked.connect(_render)
        campaign_combo.currentIndexChanged.connect(_render)
        _render()
        return root

    except Exception as e:
        traceback.print_exc()
        fail_widget = QWidget()
        fail_layout = QVBoxLayout(fail_widget)
        fail_label = QLabel(f"Failed to initialize Marketing view:\n{e}")
        fail_label.setAlignment(Qt.AlignCenter)
        fail_label.setStyleSheet("color:red; font-size:14pt;")
        fail_layout.addWidget(fail_label)
        return fail_widget
