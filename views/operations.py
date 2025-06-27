# views/operations.py – with interactive Plotly charts
# ----------------------------------------------------
# • Uses Plotly for interactive charts
# • Provides operational efficiency analysis
# • Includes correlation analysis and KPI tracking

from __future__ import annotations

import traceback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDateEdit,
    QPushButton,
    QGridLayout,
    QComboBox,
    QTabWidget,
    QTableView,
    QSizePolicy,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate

from data.helpers import get_df
from views.utils import data_required, kpi_tile, create_plotly_widget


def _classify_performance(value: float) -> str:
    """Classify performance based on value percentage."""
    if value >= 70:
        return "strong"
    elif value >= 50:
        return "moderate"
    else:
        return "weak"


def _collapsible(text: str) -> QWidget:
    """Create a collapsible explanation panel."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 5, 0, 5)

    # Toggle button
    toggle_btn = QPushButton("Show explanation")
    toggle_btn.setStyleSheet("""
        QPushButton {
            background-color: #4a86e8;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            max-width: 150px;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
    """)

    # Explanation label with dark blue background
    explanation = QLabel(text)
    explanation.setWordWrap(True)
    explanation.setStyleSheet("""
        background-color: rgba(25, 45, 90, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-size: 11pt;
        line-height: 1.4;
    """)
    explanation.setVisible(False)

    # Add widgets to layout
    layout.addWidget(toggle_btn)
    layout.addWidget(explanation)

    # Connect toggle button
    def toggle_explanation():
        is_visible = explanation.isVisible()
        explanation.setVisible(not is_visible)
        toggle_btn.setText("Hide explanation" if not is_visible else "Show explanation")

    toggle_btn.clicked.connect(toggle_explanation)

    return container


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





class PandasModel(QAbstractTableModel):
    """A model to interface between a Qt view and pandas dataframe"""
    def __init__(self, dataframe):
        super().__init__()
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return str(self._dataframe.iat[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._dataframe.columns[section])
        return str(self._dataframe.index[section])


# ─────────────────────────────────────────────────────────────
# FOOD & BEVERAGE DASHBOARD (with correlation analysis)
# ─────────────────────────────────────────────────────────────
@data_required
def display_fb() -> QWidget:
    """Display Food & Beverage operations analytics dashboard."""

    # Generate sample data function
    def generate_sample_data():
        # Generate more sample data to provide a better date range
        date_range = pd.date_range(start='2024-01-01', end=pd.Timestamp.today(), freq="D")
        outlets = ["Main Restaurant", "Lobby Bar", "Pool Bar", "Room Service"]

        data = []
        np.random.seed(42)  # For reproducible results

        for date in date_range:
            is_weekend = date.dayofweek >= 5
            weekend_factor = 1.4 if is_weekend else 1.0

            for outlet in outlets:
                # Base values per outlet
                if outlet == "Main Restaurant":
                    base_guests = 110
                    base_check = 45
                    food_ratio = 0.7
                elif outlet == "Lobby Bar":
                    base_guests = 60
                    base_check = 28
                    food_ratio = 0.3
                elif outlet == "Pool Bar":
                    base_guests = 75
                    base_check = 32
                    food_ratio = 0.4
                else:  # Room Service
                    base_guests = 28
                    base_check = 55
                    food_ratio = 0.6

                # Randomize
                guests = int(base_guests * weekend_factor * np.random.normal(1, 0.15))
                avg_check = base_check * np.random.normal(1, 0.1)
                total_revenue = guests * avg_check

                # Split revenue
                food_rev = total_revenue * food_ratio * np.random.uniform(0.9, 1.1)
                bev_rev = total_revenue - food_rev

                data.append({
                    "date": date,
                    "outlet": outlet,
                    "guests": guests,
                    "avg_check": avg_check,
                    "food_revenue": food_rev,
                    "beverage_revenue": bev_rev,
                    "total_revenue": total_revenue
                })

        return pd.DataFrame(data)

    # Create root widget
    root = QWidget()
    root_layout = QVBoxLayout()
    root.setLayout(root_layout)

    # Add header
    header = QLabel("Food & Beverage Analysis")
    header.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
    root_layout.addWidget(header)

    # Create sample data
    sample_data = generate_sample_data()

    # Get actual date range from data
    min_date = sample_data["date"].min().date()
    max_date = sample_data["date"].max().date()

    # Add date filters
    filter_layout = QHBoxLayout()
    filter_layout.addWidget(QLabel("Date Range:"))

    # Date pickers
    start_date = QDateEdit()
    end_date = QDateEdit()
    start_date.setCalendarPopup(True)
    end_date.setCalendarPopup(True)

    # Set date range limits
    start_date.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
    start_date.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))
    end_date.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
    end_date.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))

    # Set default dates - start from data start, end at data end
    start_date.setDate(QDate(min_date.year, min_date.month, min_date.day))
    end_date.setDate(QDate(max_date.year, max_date.month, max_date.day))

    filter_layout.addWidget(start_date)
    filter_layout.addWidget(QLabel("to"))
    filter_layout.addWidget(end_date)

    # Apply button
    apply_btn = QPushButton("Apply")
    apply_btn.setStyleSheet("background-color: #3b82f6; color: white; padding: 5px 15px; border-radius: 4px;")
    filter_layout.addWidget(apply_btn)
    filter_layout.addStretch()

    root_layout.addLayout(filter_layout)

    # KPI grid for summary metrics
    kpi_grid = QGridLayout()
    kpi_grid.setSpacing(10)
    root_layout.addLayout(kpi_grid)

    # Add tabs for different analyses
    tabs = QTabWidget()
    tabs.setStyleSheet("""
        QTabWidget::pane { border: 0; }
        QTabBar::tab {
            background: rgba(40, 40, 40, 0.7);
            color: white;
            padding: 8px 16px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            border: 1px solid #555555;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: rgba(0, 0, 0, 0.8);
            border-bottom: 2px solid #4a86e8;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected {
            background: rgba(60, 60, 60, 0.7);
        }
    """)
    root_layout.addWidget(tabs)

    # Function to filter data
    def get_filtered_data():
        start = pd.Timestamp(start_date.date().toPython())
        end = pd.Timestamp(end_date.date().toPython())
        return sample_data[(sample_data["date"] >= start) & (sample_data["date"] <= end)]

    # Add correlation analysis tab
    def create_correlation_tab(df):
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)

        # Select numeric columns for correlation
        numeric_df = df[['guests', 'avg_check', 'food_revenue', 'beverage_revenue', 'total_revenue']]
        corr_matrix = numeric_df.corr()

        # Create Plotly correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title="Correlation"
            )
        ))

        fig.update_layout(
            title="F&B Metrics Correlation Matrix",
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            template="plotly_white",
            height=500,
            width=800,
            xaxis=dict(tickangle=45)
        )

        # Add chart to tab
        corr_layout.addWidget(create_plotly_widget(fig))

        # Calculate KPI - highest absolute correlation excluding self-correlations
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        max_corr = corr_matrix.abs().where(mask).max().max() * 100
        performance = _classify_performance(max_corr)

        # Find which metrics have this correlation
        max_corr_idx = np.where(np.abs(corr_matrix.values) == max_corr/100)
        if len(max_corr_idx[0]) >= 2:
            metric1 = corr_matrix.index[max_corr_idx[0][0]]
            metric2 = corr_matrix.columns[max_corr_idx[1][0]]
            corr_direction = "positive" if corr_matrix.loc[metric1, metric2] > 0 else "negative"
        else:
            metric1, metric2 = "metrics", "metrics"
            corr_direction = "strong"

        # Create explanation
        explanation = (
            f"The strongest relationship is between {metric1} and {metric2} at {max_corr:.1f}%, "
            f"a {performance} {corr_direction} correlation. The heatmap reveals how different "
            f"F&B metrics relate to each other, helping identify key operational drivers."
        )

        # Add collapsible explanation
        corr_layout.addWidget(_collapsible(explanation))

        return corr_tab

    # Basic render function
    def render():
        # Clear existing widgets
        for i in reversed(range(kpi_grid.count())):
            kpi_grid.itemAt(i).widget().setParent(None)
        tabs.clear()

        # Get filtered data
        df = get_filtered_data()

        # Calculate KPIs
        total_revenue = df["total_revenue"].sum()
        avg_check = df["avg_check"].mean()
        total_guests = df["guests"].sum()

        # Display KPIs
        kpis = [
            ("Food/Beverage", f"{df['food_revenue'].sum()/df['beverage_revenue'].sum():.1f}"),
        ]
        for i, (label, value) in enumerate(kpis):
            kpi_grid.addWidget(kpi_tile(label, value), i//4, i%4)

        # Create revenue tab
        revenue_tab = QWidget()
        revenue_layout = QVBoxLayout(revenue_tab)

        # Get revenue by outlet
        outlet_revenue = df.groupby("outlet")["total_revenue"].sum()

        # Create Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=outlet_revenue.index,
                y=outlet_revenue.values,
                marker_color='#3b82f6',
                text=[f'${val:,.0f}' for val in outlet_revenue.values],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Revenue by Outlet",
            xaxis_title="Outlet",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            height=500,
            width=800,
            yaxis=dict(tickformat="$,.0f"),
            showlegend=False
        )

        # Add chart to tab
        revenue_layout.addWidget(create_plotly_widget(fig))

        # Calculate KPI - share of top-grossing outlet
        top_outlet_revenue = outlet_revenue.max()
        top_outlet_name = outlet_revenue.idxmax()
        top_outlet_share = (top_outlet_revenue / total_revenue) * 100
        performance = _classify_performance(top_outlet_share)

        # Create explanation
        outlet_explanation = (
            f"The {top_outlet_name} generates {top_outlet_share:.1f}% of total revenue, a {performance} "
            f"concentration. This chart compares revenue across outlets, highlighting which "
            f"food and beverage venues contribute most to the bottom line."
        )

        # Add collapsible explanation
        revenue_layout.addWidget(_collapsible(outlet_explanation))

        tabs.addTab(revenue_tab, "Revenue Analysis")

        # Create trends tab
        trends_tab = QWidget()
        trends_layout = QVBoxLayout(trends_tab)

        # Get daily totals
        daily = df.groupby("date")[["food_revenue", "beverage_revenue"]].sum()

        # Create Plotly line chart
        fig2 = go.Figure()

        # Add food revenue line
        fig2.add_trace(go.Scatter(
            x=daily.index,
            y=daily["food_revenue"],
            mode='lines+markers',
            name='Food',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=6)
        ))

        # Add beverage revenue line
        fig2.add_trace(go.Scatter(
            x=daily.index,
            y=daily["beverage_revenue"],
            mode='lines+markers',
            name='Beverage',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=6)
        ))

        fig2.update_layout(
            title="Daily Revenue Breakdown",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            height=500,
            width=800,
            yaxis=dict(tickformat="$,.0f"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add chart to tab
        trends_layout.addWidget(create_plotly_widget(fig2))

        # Calculate KPI - 30-day revenue growth (if we have enough data)
        total_daily = daily["food_revenue"] + daily["beverage_revenue"]

        # Default explanation if not enough data
        trend_explanation = (
            f"This chart shows the daily breakdown between food and beverage revenue, "
            f"revealing seasonal patterns and the relative contribution of each category."
        )

        # Calculate 30-day growth if we have enough data
        if len(total_daily) >= 15:
            # Compare first half to second half as an approximation
            half = len(total_daily) // 2
            first_half = total_daily.iloc[:half].sum()
            second_half = total_daily.iloc[half:].sum()

            if first_half > 0:
                growth_pct = ((second_half / first_half) - 1) * 100
                trend = "growth" if growth_pct > 0 else "decline"
                performance = _classify_performance(abs(growth_pct))

                trend_explanation = (
                    f"Revenue shows {abs(growth_pct):.1f}% {trend} in the recent period, a {performance} "
                    f"trend. The chart illustrates daily food and beverage revenue patterns, "
                    f"helping identify seasonal fluctuations and category performance."
                )

        # Add collapsible explanation
        trends_layout.addWidget(_collapsible(trend_explanation))

        tabs.addTab(trends_tab, "Revenue Trends")

        # Add correlation tab
        tabs.addTab(create_correlation_tab(df), "Correlation Analysis")

    # Connect button and do initial render
    apply_btn.clicked.connect(render)
    render()

    return root


# Helper function to get efficiency data
def get_efficiency_data():
    """Get operational efficiency data or return None if not available."""
    try:
        return get_df('efficiency')  # Attempt to get efficiency data
    except Exception:
        return None

# Helper function to get custom chart data
def get_custom_charts():
    """Get custom chart data or return None if not available."""
    try:
        return get_df('custom_charts')  # Attempt to get custom chart data
    except Exception:
        return None

# ── Display Efficiency (added implementation) ─────────────────
@data_required
def display_efficiency() -> QWidget:
    """Display operational efficiency metrics."""
    root = QWidget()
    layout = QVBoxLayout()
    root.setLayout(layout)

    # Add header
    header = QLabel("Operational Efficiency")
    header.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
    layout.addWidget(header)

    # Get some sample data
    df = get_efficiency_data()
    if df is None or df.empty:
        # Create sample data if no data is available
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
        np.random.seed(42)

        data = {
            "date": dates,
            "staff_utilization": np.random.uniform(0.7, 0.95, 30),
            "avg_response_time": np.random.uniform(5, 20, 30),
            "costs_per_room": np.random.uniform(20, 35, 30),
            "guest_satisfaction": np.random.uniform(3.5, 4.8, 30)
        }
        df = pd.DataFrame(data)

    # Create efficiency metrics visualization with Plotly
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # Add staff utilization line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['staff_utilization'] * 100,
            mode='lines+markers',
            name='Staff Utilization %',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )

    # Add response time line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['avg_response_time'],
            mode='lines+markers',
            name='Avg Response Time (min)',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="Utilization (%)", secondary_y=False)
    fig.update_yaxes(title_text="Response Time (minutes)", secondary_y=True)

    fig.update_layout(
        title="Operational Efficiency Metrics",
        template="plotly_white",
        height=500,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add chart to layout
    layout.addWidget(create_plotly_widget(fig))

    # Calculate KPI - average staff utilization
    avg_utilization = df["staff_utilization"].mean() * 100
    performance = _classify_performance(avg_utilization)

    # Create explanation
    efficiency_explanation = (
        f"Average staff utilization is {avg_utilization:.1f}%, a {performance} efficiency level. "
        f"The chart tracks utilization against response times, showing how staffing efficiency "
        f"affects service speed and overall operational performance."
    )

    # Add collapsible explanation
    layout.addWidget(_collapsible(efficiency_explanation))

    # Add table with detailed metrics
    table = QTableView()
    model = PandasModel(df)
    table.setModel(model)
    layout.addWidget(table)

    return root


# ── Custom Charts ────────────────────────────────────────────
@data_required
def display_custom_charts() -> QWidget:
    """Display custom charts and analytics."""
    root = QWidget()
    layout = QVBoxLayout()
    root.setLayout(layout)

    # Add header
    header = QLabel("Custom Charts")
    header.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
    layout.addWidget(header)

    # Get chart data
    chart_data = get_custom_charts()
    if chart_data is None or len(chart_data) == 0:
        # Sample data if no data is available
        chart_data = [
            {
                "title": "Revenue vs. Occupancy",
                "type": "scatter",
                "x": np.random.uniform(50, 95, 30),
                "y": np.random.uniform(10000, 50000, 30),
                "x_label": "Occupancy %",
                "y_label": "Daily Revenue ($)"
            },
            {
                "title": "Market Segment Mix",
                "type": "pie",
                "values": [38, 27, 15, 10, 10],
                "labels": ["Corporate", "Leisure", "Groups", "OTA", "Other"]
            }
        ]

    # Create tabs for each chart
    chart_tabs = QTabWidget()

    for i, chart in enumerate(chart_data):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # Chart-specific KPIs and explanations
        explanation = ""

        if chart["type"] == "scatter":
            # Create Plotly scatter plot
            fig = go.Figure()

            # Add scatter points
            fig.add_trace(go.Scatter(
                x=chart["x"],
                y=chart["y"],
                mode='markers',
                marker=dict(
                    color='#3b82f6',
                    size=8,
                    opacity=0.7
                ),
                name='Data Points'
            ))

            # Add trend line
            z = np.polyfit(chart["x"], chart["y"], 1)
            p = np.poly1d(z)
            trend_x = np.linspace(min(chart["x"]), max(chart["x"]), 100)
            trend_y = p(trend_x)

            fig.add_trace(go.Scatter(
                x=trend_x,
                y=trend_y,
                mode='lines',
                line=dict(color='#ef4444', dash='dash', width=2),
                name='Trend Line'
            ))

            fig.update_layout(
                title=chart["title"],
                xaxis_title=chart["x_label"],
                yaxis_title=chart["y_label"],
                template="plotly_white",
                height=500,
                width=800
            )

            # Calculate KPI - correlation coefficient
            corr = np.corrcoef(chart["x"], chart["y"])[0, 1]
            corr_pct = abs(corr) * 100
            performance = _classify_performance(corr_pct)
            corr_direction = "positive" if corr > 0 else "negative"

            explanation = (
                f"The correlation between {chart['x_label']} and {chart['y_label']} is {corr:.2f}, "
                f"a {performance} {corr_direction} relationship. The scatter plot reveals how these metrics "
                f"influence each other, with the trend line showing the overall pattern."
            )

        elif chart["type"] == "pie":
            # Create Plotly pie chart
            fig = go.Figure(data=[go.Pie(
                labels=chart["labels"],
                values=chart["values"],
                marker_colors=['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'],
                textinfo='label+percent',
                textposition='auto'
            )])

            fig.update_layout(
                title=chart["title"],
                template="plotly_white",
                height=500,
                width=800
            )

            # Calculate KPI - largest slice share
            largest_slice = max(chart["values"])
            total = sum(chart["values"])
            largest_share = (largest_slice / total) * 100
            largest_label = chart["labels"][chart["values"].index(largest_slice)]
            performance = _classify_performance(largest_share)

            explanation = (
                f"The '{largest_label}' segment accounts for {largest_share:.1f}% of the total, "
                f"a {performance} concentration. The pie chart shows the distribution across segments, "
                f"highlighting which areas have the most significant impact."
            )

        # Add chart to tab
        tab_layout.addWidget(create_plotly_widget(fig))

        # Add collapsible explanation
        tab_layout.addWidget(_collapsible(explanation))

        chart_tabs.addTab(tab, chart["title"])

    layout.addWidget(chart_tabs)

    return root
