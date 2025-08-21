# flake8: noqa
# views/operations.py – Soft UI (pastel) + interactive Plotly
# -----------------------------------------------------------
# • Palette pastel unifiée (même design que les autres vues)
# • Toolbar date-range compacte + bouton Apply
# • Cartes (cards) pour héberger les graphiques
# • Panneaux "Show insight" repliables et homogènes

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDateEdit, QPushButton,
    QGridLayout, QTabWidget, QTableView, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QDate

from data.helpers import get_df
from views.utils import data_required, kpi_tile, create_plotly_widget


# ─────────────────────────────────────────────────────────
# Palette douce + helpers
# ─────────────────────────────────────────────────────────
GRID_COLOR = "#eef2f7"
FONT_COLOR = "#0f172a"

SOFT_BLUE  = "#64b5f6"
SOFT_RED   = "#ef9a9a"
SOFT_TEAL  = "#80cbc4"
SOFT_MINT  = "#a7f3d0"
SOFT_LILAC = "#c4b5fd"
SOFT_AMBER = "#fde68a"
SOFT_GREEN = "#9ae6b4"
SOFT_CORAL = "#f8b4b4"
SOFT_CYAN  = "#9cd3d3"
SOFT_GREY  = "#cfd8e3"


def _apply_soft_layout(fig: go.Figure, title: str, height: int = 500):
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font=dict(color=FONT_COLOR, size=11),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)


def _card_wrap(widget: QWidget) -> QFrame:
    """Encapsule un widget dans une 'carte' douce."""
    card = QFrame()
    card.setObjectName("card")
    card.setStyleSheet("""
        QFrame#card {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 12px;
        }
    """)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.addWidget(widget)
    return card


def _collapsible(text: str) -> QWidget:
    """Panneau repliable pour les insights (design doux)."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 6, 0, 0)

    toggle_btn = QPushButton("Show insight")
    toggle_btn.setCursor(Qt.PointingHandCursor)
    toggle_btn.setStyleSheet("""
        QPushButton {
            background-color: #e9efff;
            color: #1d4ed8;
            padding: 6px 10px;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #dbe6ff;
            max-width: 140px;
        }
        QPushButton:hover { background-color: #e3eaff; }
        QPushButton:pressed { background-color: #d8e2ff; }
    """)

    explanation = QLabel(text)
    explanation.setWordWrap(True)
    explanation.setStyleSheet("""
        QLabel {
            background-color: #f6f8fc;
            color: #334155;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid #e5e9f2;
            font-size: 11pt;
            line-height: 1.45;
        }
    """)
    explanation.setVisible(False)

    layout.addWidget(toggle_btn)
    layout.addWidget(explanation)

    def toggle_explanation():
        is_visible = explanation.isVisible()
        explanation.setVisible(not is_visible)
        toggle_btn.setText("Hide insight" if not is_visible else "Show insight")

    toggle_btn.clicked.connect(toggle_explanation)
    return container


def _classify_performance(value: float) -> str:
    """Classify performance based on percentage value."""
    if value >= 70:
        return "strong"
    elif value >= 50:
        return "moderate"
    else:
        return "weak"


# ─────────────────────────────────────────────────────────
# Table model
# ─────────────────────────────────────────────────────────
class PandasModel(QAbstractTableModel):
    """Model Qt pour DataFrame pandas (read-only)."""
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


# ─────────────────────────────────────────────────────────
# FOOD & BEVERAGE DASHBOARD
# ─────────────────────────────────────────────────────────
@data_required
def display_fb() -> QWidget:
    """Food & Beverage operations analytics dashboard (soft UI)."""

    # ---- sample data (fallback demo) ----
    def generate_sample_data():
        date_range = pd.date_range(start='2024-01-01', end=pd.Timestamp.today(), freq="D")
        outlets = ["Main Restaurant", "Lobby Bar", "Pool Bar", "Room Service"]
        data = []
        np.random.seed(42)
        for date in date_range:
            is_weekend = date.dayofweek >= 5
            weekend_factor = 1.4 if is_weekend else 1.0
            for outlet in outlets:
                if outlet == "Main Restaurant":
                    base_guests, base_check, food_ratio = 110, 45, 0.7
                elif outlet == "Lobby Bar":
                    base_guests, base_check, food_ratio = 60, 28, 0.3
                elif outlet == "Pool Bar":
                    base_guests, base_check, food_ratio = 75, 32, 0.4
                else:
                    base_guests, base_check, food_ratio = 28, 55, 0.6
                guests = int(base_guests * weekend_factor * np.random.normal(1, 0.15))
                avg_check = base_check * np.random.normal(1, 0.1)
                total_revenue = guests * avg_check
                food_rev = total_revenue * food_ratio * np.random.uniform(0.9, 1.1)
                bev_rev = total_revenue - food_rev
                data.append({
                    "date": date, "outlet": outlet, "guests": guests, "avg_check": avg_check,
                    "food_revenue": food_rev, "beverage_revenue": bev_rev, "total_revenue": total_revenue
                })
        return pd.DataFrame(data)

    sample_data = generate_sample_data()

    # ---- root + QSS ----
    root = QWidget()
    root.setObjectName("OpsRoot")
    root.setLayout(QVBoxLayout())
    root.layout().setContentsMargins(12, 12, 12, 12)
    root.layout().setSpacing(10)

    root.setStyleSheet("""
        QWidget#OpsRoot {
            background: #fbfcfe;
            color: #0f172a;
            font-size: 13px;
        }
        QLabel#header {
            color: #0f172a;
            font-size: 20px;
            font-weight: 800;
            padding: 12px 16px;
            border-radius: 14px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f4f7ff, stop:1 #ffffff);
            border: 1px solid #e9edf7;
        }
        QFrame#toolbar {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 12px;
        }
        QFrame#toolbar QLabel { color: #334155; }
        QDateEdit {
            background: #ffffff;
            border: 1px solid #dfe6f3;
            border-radius: 8px;
            padding: 6px 8px;
            min-width: 130px;
            selection-background-color: #e0eaff;
        }
        QPushButton#applyBtn {
            background: #e9efff;
            color: #1d4ed8;
            border: 1px solid #dbe6ff;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton#applyBtn:hover  { background: #e3eaff; }
        QPushButton#applyBtn:pressed{ background: #d8e2ff; }

        QTabWidget::pane {
            border: 1px solid #e5e9f2;
            border-radius: 12px;
            background: #ffffff;
            padding: 4px;
        }
        QTabBar::tab {
            background: #f6f8fc;
            border: 1px solid #e5e9f2;
            padding: 7px 14px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            color: #334155;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #0f172a;
            border-bottom-color: transparent;
        }
        QTabBar::tab:hover { background: #eef2f9; }
    """)

    # ---- header ----
    header = QLabel("Food & Beverage Analysis")
    header.setObjectName("header")
    root.layout().addWidget(header)

    # ---- toolbar ----
    toolbar = QFrame()
    toolbar.setObjectName("toolbar")
    tl = QHBoxLayout(toolbar)
    tl.setContentsMargins(12, 10, 12, 10)
    tl.setSpacing(8)

    # date pickers
    min_date = sample_data["date"].min().date()
    max_date = sample_data["date"].max().date()

    start_date = QDateEdit(QDate(min_date.year, min_date.month, min_date.day))
    end_date   = QDateEdit(QDate(max_date.year, max_date.month, max_date.day))
    for p in (start_date, end_date):
        p.setCalendarPopup(True)

    tl.addWidget(QLabel("Date Range:"))
    tl.addWidget(start_date)
    tl.addWidget(QLabel(" to "))
    tl.addWidget(end_date)
    tl.addStretch()

    apply_btn = QPushButton("Apply")
    apply_btn.setObjectName("applyBtn")
    tl.addWidget(apply_btn)

    root.layout().addWidget(toolbar)

    # ---- KPI grid ----
    kpi_grid = QGridLayout()
    kpi_grid.setSpacing(10)
    root.layout().addLayout(kpi_grid)

    # ---- tabs ----
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    # helpers
    def get_filtered_data():
        start = pd.Timestamp(start_date.date().toPython())
        end = pd.Timestamp(end_date.date().toPython())
        return sample_data[(sample_data["date"] >= start) & (sample_data["date"] <= end)].copy()

    def create_correlation_tab(df: pd.DataFrame) -> QWidget:
        corr_tab = QWidget()
        lay = QVBoxLayout(corr_tab)

        numeric_df = df[['guests', 'avg_check', 'food_revenue', 'beverage_revenue', 'total_revenue']]
        corr_matrix = numeric_df.corr()

        # heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
            colorscale=[[0, SOFT_RED], [0.5, "#ffffff"], [1, SOFT_TEAL]],
            zmin=-1, zmax=1, zmid=0, hoverongaps=False, showscale=True,
            text=corr_matrix.round(2).values, texttemplate="%{text}", textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        _apply_soft_layout(fig, "F&B Metrics Correlation Matrix", height=520)
        lay.addWidget(_card_wrap(create_plotly_widget(fig)))

        # KPI + insight
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        max_corr = corr_matrix.abs().where(mask).max().max() * 100
        performance = _classify_performance(max_corr)
        # find pair
        where_max = np.where(np.abs(corr_matrix.values) == max_corr/100)
        if len(where_max[0]) >= 2:
            m1 = corr_matrix.index[where_max[0][0]]
            m2 = corr_matrix.columns[where_max[1][0]]
            direction = "positive" if corr_matrix.loc[m1, m2] > 0 else "negative"
        else:
            m1, m2, direction = "—", "—", "strong"

        expl = (
            f"Strongest relationship: **{m1} ↔ {m2}** at **{max_corr:.1f}%**, "
            f"a **{performance}** {direction} correlation. Use it to target key operational drivers."
        )
        lay.addWidget(_collapsible(expl))
        return corr_tab

    # render
    def render():
        # clear KPI grid
        for i in reversed(range(kpi_grid.count())):
            w = kpi_grid.itemAt(i).widget()
            if w:
                w.setParent(None)
        tabs.clear()

        df = get_filtered_data()
        if df.empty:
            info = QWidget()
            l = QVBoxLayout(info)
            msg = QLabel("No data for selected date range.")
            msg.setStyleSheet("font-size: 14px; color: #8b5e34;")
            l.addWidget(msg, 0, Qt.AlignCenter)
            tabs.addTab(info, "Info")
            return

        # KPIs
        total_revenue = float(df["total_revenue"].sum())
        avg_check = float(df["avg_check"].mean())
        total_guests = int(df["guests"].sum())
        fb_ratio = (df["food_revenue"].sum() / df["beverage_revenue"].sum()) if df["beverage_revenue"].sum() else 0

        kpis = [
            ("Total Revenue", f"${total_revenue:,.0f}"),
            ("Avg Check", f"${avg_check:,.2f}"),
            ("Guests", f"{total_guests:,}"),
            ("Food/Beverage", f"{fb_ratio:.1f}"),
        ]
        for i, (label, value) in enumerate(kpis):
            kpi_grid.addWidget(kpi_tile(label, value), i // 4, i % 4)

        # ─ Revenue Analysis tab
        revenue_tab = QWidget()
        rl = QVBoxLayout(revenue_tab)

        outlet_revenue = df.groupby("outlet")["total_revenue"].sum().sort_values(ascending=False)
        fig1 = go.Figure([go.Bar(
            x=outlet_revenue.index, y=outlet_revenue.values,
            marker_color=SOFT_BLUE,
            text=[f'${v:,.0f}' for v in outlet_revenue.values], textposition='outside'
        )])
        fig1.update_yaxes(tickformat="$,.0f")
        _apply_soft_layout(fig1, "Revenue by Outlet", height=520)
        rl.addWidget(_card_wrap(create_plotly_widget(fig1)))

        top_outlet_name = outlet_revenue.idxmax()
        top_outlet_share = (outlet_revenue.max() / total_revenue * 100) if total_revenue else 0
        perf = _classify_performance(top_outlet_share)
        outlet_expl = (
            f"**{top_outlet_name}** contributes **{top_outlet_share:.1f}%** of total revenue "
            f"(**{perf}** concentration). Compare outlets to spot focus areas."
        )
        rl.addWidget(_collapsible(outlet_expl))
        tabs.addTab(revenue_tab, "Revenue Analysis")

        # ─ Trends tab
        trends_tab = QWidget()
        tl = QVBoxLayout(trends_tab)

        daily = df.groupby("date")[["food_revenue", "beverage_revenue"]].sum().reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily["date"], y=daily["food_revenue"],
            mode="lines+markers", name="Food",
            line=dict(color=SOFT_AMBER, width=3),
            marker=dict(size=7, line=dict(color="white", width=1))
        ))
        fig2.add_trace(go.Scatter(
            x=daily["date"], y=daily["beverage_revenue"],
            mode="lines+markers", name="Beverage",
            line=dict(color=SOFT_BLUE, width=3),
            marker=dict(size=7, line=dict(color="white", width=1))
        ))
        fig2.update_yaxes(tickformat="$,.0f")
        _apply_soft_layout(fig2, "Daily Revenue Breakdown", height=520)
        tl.addWidget(_card_wrap(create_plotly_widget(fig2)))

        total_daily = daily["food_revenue"] + daily["beverage_revenue"]
        trend_expl = (
            "Daily split between **Food** and **Beverage** reveals seasonality and pacing."
        )
        if len(total_daily) >= 15:
            half = len(total_daily) // 2
            first, second = total_daily.iloc[:half].sum(), total_daily.iloc[half:].sum()
            if first > 0:
                growth_pct = (second / first - 1) * 100
                trend = "growth" if growth_pct > 0 else "decline"
                performance = _classify_performance(abs(growth_pct))
                trend_expl = (
                    f"Recent period shows **{abs(growth_pct):.1f}% {trend}** (**{performance}**). "
                    "Use to align staffing and procurement."
                )
        tl.addWidget(_collapsible(trend_expl))
        tabs.addTab(trends_tab, "Revenue Trends")

        # ─ Correlation tab
        tabs.addTab(create_correlation_tab(df), "Correlation Analysis")

    apply_btn.clicked.connect(render)
    render()

    return root


# ─────────────────────────────────────────────────────────
# DATA PROVIDERS (optionnels)
# ─────────────────────────────────────────────────────────
def get_efficiency_data():
    """Get operational efficiency data or return None if not available."""
    try:
        return get_df('efficiency')
    except Exception:
        return None


def get_custom_charts():
    """Get custom chart data or return None if not available."""
    try:
        return get_df('custom_charts')
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# OPERATIONAL EFFICIENCY
# ─────────────────────────────────────────────────────────
@data_required
def display_efficiency() -> QWidget:
    """Operational efficiency metrics (soft UI)."""
    root = QWidget()
    root.setObjectName("OpsRoot")
    root.setLayout(QVBoxLayout())
    root.layout().setContentsMargins(12, 12, 12, 12)
    root.layout().setSpacing(10)

    # Reuse same QSS as display_fb
    root.setStyleSheet("""
        QWidget#OpsRoot { background: #fbfcfe; color: #0f172a; font-size: 13px; }
        QLabel#header {
            color: #0f172a; font-size: 20px; font-weight: 800;
            padding: 12px 16px; border-radius: 14px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f4f7ff, stop:1 #ffffff);
            border: 1px solid #e9edf7;
        }
    """)

    header = QLabel("Operational Efficiency")
    header.setObjectName("header")
    root.layout().addWidget(header)

    # data
    df = get_efficiency_data()
    if df is None or df.empty:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "date": dates,
            "staff_utilization": np.random.uniform(0.7, 0.95, 30),
            "avg_response_time": np.random.uniform(5, 20, 30),
            "costs_per_room": np.random.uniform(20, 35, 30),
            "guest_satisfaction": np.random.uniform(3.5, 4.8, 30)
        })

    # figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['staff_utilization'] * 100,
        mode='lines+markers', name='Staff Utilization (%)',
        line=dict(color=SOFT_TEAL, width=3),
        marker=dict(size=7, line=dict(color="white", width=1))
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['avg_response_time'],
        mode='lines+markers', name='Avg Response Time (min)',
        line=dict(color=SOFT_RED, width=3),
        marker=dict(size=7, line=dict(color="white", width=1))
    ), secondary_y=True)

    fig.update_yaxes(title_text="Utilization (%)", secondary_y=False)
    fig.update_yaxes(title_text="Response Time (minutes)", secondary_y=True)
    _apply_soft_layout(fig, "Operational Efficiency Metrics", height=520)

    root.layout().addWidget(_card_wrap(create_plotly_widget(fig)))

    # insight
    avg_util = df["staff_utilization"].mean() * 100
    perf = _classify_performance(avg_util)
    expl = (
        f"Average staff utilization **{avg_util:.1f}%** (**{perf}**). "
        "Track utilization vs response time to balance service speed & staffing."
    )
    root.layout().addWidget(_collapsible(expl))

    # table
    table = QTableView()
    table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    table.setModel(PandasModel(df))
    root.layout().addWidget(_card_wrap(table))

    return root


# ─────────────────────────────────────────────────────────
# CUSTOM CHARTS
# ─────────────────────────────────────────────────────────
@data_required
def display_custom_charts() -> QWidget:
    """Custom charts & analytics (soft UI)."""
    root = QWidget()
    root.setObjectName("OpsRoot")
    root.setLayout(QVBoxLayout())
    root.layout().setContentsMargins(12, 12, 12, 12)
    root.layout().setSpacing(10)

    root.setStyleSheet("""
        QWidget#OpsRoot { background: #fbfcfe; color: #0f172a; font-size: 13px; }
        QLabel#header {
            color: #0f172a; font-size: 20px; font-weight: 800;
            padding: 12px 16px; border-radius: 14px;
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f4f7ff, stop:1 #ffffff);
            border: 1px solid #e9edf7;
        }
        QTabWidget::pane {
            border: 1px solid #e5e9f2; border-radius: 12px; background: #ffffff; padding: 4px;
        }
        QTabBar::tab {
            background: #f6f8fc; border: 1px solid #e5e9f2; padding: 7px 14px; margin-right: 6px;
            border-top-left-radius: 10px; border-top-right-radius: 10px; color: #334155;
        }
        QTabBar::tab:selected { background: #ffffff; color: #0f172a; border-bottom-color: transparent; }
        QTabBar::tab:hover { background: #eef2f9; }
    """)

    header = QLabel("Custom Charts")
    header.setObjectName("header")
    root.layout().addWidget(header)

    chart_data = get_custom_charts()
    if not chart_data:
        chart_data = [
            {
                "title": "Revenue vs. Occupancy",
                "type": "scatter",
                "x": np.random.uniform(50, 95, 30),
                "y": np.random.uniform(10000, 50000, 30),
                "x_label": "Occupancy (%)",
                "y_label": "Daily Revenue ($)"
            },
            {
                "title": "Market Segment Mix",
                "type": "pie",
                "values": [38, 27, 15, 10, 10],
                "labels": ["Corporate", "Leisure", "Groups", "OTA", "Other"]
            }
        ]

    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    for chart in chart_data:
        tab = QWidget()
        lay = QVBoxLayout(tab)

        if chart["type"] == "scatter":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart["x"], y=chart["y"], mode='markers',
                marker=dict(color=SOFT_BLUE, size=8, opacity=0.75, line=dict(color="white", width=1)),
                name='Data Points'
            ))
            # trend line
            z = np.polyfit(chart["x"], chart["y"], 1)
            p = np.poly1d(z)
            tx = np.linspace(min(chart["x"]), max(chart["x"]), 100)
            ty = p(tx)
            fig.add_trace(go.Scatter(
                x=tx, y=ty, mode='lines', line=dict(color=SOFT_RED, dash='dash', width=2),
                name='Trend Line'
            ))
            fig.update_xaxes(title=chart["x_label"])
            fig.update_yaxes(title=chart["y_label"], tickformat="$,.0f" if "$" in chart["y_label"] else None)
            _apply_soft_layout(fig, chart["title"], height=520)

            # insight
            corr = np.corrcoef(chart["x"], chart["y"])[0, 1]
            corr_pct = abs(corr) * 100
            perf = _classify_performance(corr_pct)
            direction = "positive" if corr > 0 else "negative"
            expl = (
                f"Correlation between **{chart['x_label']}** and **{chart['y_label']}** is **{corr:.2f}** "
                f"(**{perf}**, {direction}). Trend line shows overall pattern."
            )

            lay.addWidget(_card_wrap(create_plotly_widget(fig)))
            lay.addWidget(_collapsible(expl))

        elif chart["type"] == "pie":
            fig = go.Figure([go.Pie(
                labels=chart["labels"], values=chart["values"], hole=0.45,
                marker_colors=[SOFT_BLUE, SOFT_AMBER, SOFT_TEAL, SOFT_RED, SOFT_LILAC],
                textinfo='label+percent', textposition='inside'
            )])
            _apply_soft_layout(fig, chart["title"], height=520)

            largest_slice = max(chart["values"])
            total = sum(chart["values"])
            largest_share = (largest_slice / total * 100) if total else 0
            largest_label = chart["labels"][chart["values"].index(largest_slice)]
            perf = _classify_performance(largest_share)
            expl = (
                f"'{largest_label}' accounts for **{largest_share:.1f}%** (**{perf}** concentration). "
                "Use mix to prioritize campaigns."
            )

            lay.addWidget(_card_wrap(create_plotly_widget(fig)))
            lay.addWidget(_collapsible(expl))

        tabs.addTab(tab, chart["title"])

    return root

