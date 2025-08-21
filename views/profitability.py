# flake8: noqa
# views/profitability.py
"""
Room-level Profit Analysis (modern UI)
=====================================

* Two entry points wired in ROUTES:
      • display_room_profit()               – macro profit view
      • display_room_type_profitability()   – room-type deep dive
* Identical date-range picker pattern used in Seasonality / Room Cost
* Charts auto-hide when required columns are missing
* Design: soft pastel theme, cards, rounded tabs, collapsible insights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDateEdit,
    QPushButton, QTabWidget, QFrame
)

from data.helpers import get_df
from views.utils import data_required, create_plotly_widget


# ────────────────────────────────────────────────────────
# Palette douce + helpers Plotly
# ────────────────────────────────────────────────────────
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

def _apply_soft_layout(fig: go.Figure, title: str, height: int = 500, width: int | None = None):
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template="plotly_white",
        font=dict(color=FONT_COLOR, size=11),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)


# ────────────────────────────────────────────────────────
# reusable helpers
# ────────────────────────────────────────────────────────
def _mask(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def _add_profit_cols(df: pd.DataFrame) -> None:
    if "Profit" not in df.columns and {"RoomRevenue", "TotalRoomCost"}.issubset(df.columns):
        df["Profit"] = df["RoomRevenue"] - df["TotalRoomCost"]
    if "ProfitPerRoom" not in df.columns and {"Profit", "OccupiedRooms"}.issubset(df.columns):
        df["ProfitPerRoom"] = df["Profit"] / df["OccupiedRooms"].replace(0, np.nan)


def _classify_performance(value: float) -> str:
    if value >= 30:
        return "strong"
    elif value >= 15:
        return "moderate"
    else:
        return "weak"


def _collapsible(text: str) -> QWidget:
    """Collapsible insight panel (same soft style as other views)."""
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


def _maybe(tab: QTabWidget, title: str, func, df: pd.DataFrame, cols: set[str]):
    """Add a tab with chart and explanation if required columns exist, wrapped in a card."""
    if cols.issubset(df.columns) and not df.empty:
        fig, explanation = func(df)

        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(8, 8, 8, 8)

        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background: #ffffff;
                border: 1px solid #e5e9f2;
                border-radius: 12px;
            }
        """)
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 10, 10, 10)
        v.addWidget(create_plotly_widget(fig))
        v.addWidget(_collapsible(explanation))

        tab_layout.addWidget(card)
        tab.addTab(tab_widget, title)


# ────────────────────────────────────────────────────────
# chart builders – macro profit view
# ────────────────────────────────────────────────────────
def _trend_rev_cost_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby(df["Date"].dt.month)[["RoomRevenue", "TotalRoomCost"]].sum()
    g["Profit"] = g["RoomRevenue"] - g["TotalRoomCost"]

    total_revenue = g["RoomRevenue"].sum()
    total_profit = g["Profit"].sum()
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
    performance = _classify_performance(profit_margin)

    profit_growth = ((g["Profit"].iloc[-1] - g["Profit"].iloc[0]) / g["Profit"].iloc[0]) * 100 \
        if len(g) > 1 and g["Profit"].iloc[0] != 0 else 0
    trend = "increasing" if profit_growth > 0 else "decreasing"

    explanation = (
        f"Overall profit margin is {profit_margin:.1f}%, which represents **{performance}** performance. "
        f"Profit shows a {trend} trend of {abs(profit_growth):.1f}% over the period. "
        f"A widening gap between the green (revenue) and red (cost) areas indicates expanding profit margin."
    )

    fig = go.Figure()
    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in g.index]

    fig.add_trace(go.Scatter(
        x=month_names, y=g["RoomRevenue"],
        fill='tozeroy', mode='none', name='Revenue',
        fillcolor='rgba(154,230,180,0.35)', line=dict(color=SOFT_GREEN)
    ))
    fig.add_trace(go.Scatter(
        x=month_names, y=g["TotalRoomCost"],
        fill='tozeroy', mode='none', name='Cost',
        fillcolor='rgba(239,154,154,0.35)', line=dict(color=SOFT_RED)
    ))
    fig.add_trace(go.Scatter(
        x=month_names, y=g["Profit"],
        mode='lines+markers', name='Profit',
        line=dict(color=SOFT_BLUE, width=3),
        marker=dict(size=7, line=dict(color="white", width=1))
    ))

    _apply_soft_layout(fig, "Room Revenue • Cost • Profit")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Amount ($)", tickformat="$,.0f")
    return fig, explanation


def _profit_by_source(df: pd.DataFrame) -> tuple[go.Figure, str]:
    pivot = df.groupby([df["Date"].dt.month, "BookingSource"])["Profit"].sum().unstack(fill_value=0)

    total_by_source = pivot.sum()
    top_source = total_by_source.idxmax() if not total_by_source.empty else "N/A"
    top_share = (total_by_source.max() / total_by_source.sum()) * 100 if total_by_source.sum() > 0 else 0
    performance = _classify_performance(top_share)

    source_count = len(total_by_source)
    top_3_share = total_by_source.nlargest(min(3, source_count)).sum() / total_by_source.sum() * 100 \
        if total_by_source.sum() > 0 else 0

    explanation = (
        f"'{top_source}' is the top profit source with {top_share:.1f}% of total profit, "
        f"representing **{performance}** channel concentration. "
        f"Top 3 sources generate {top_3_share:.1f}% of profit."
    )

    fig = go.Figure()
    palette = [SOFT_BLUE, SOFT_RED, SOFT_TEAL, "#f6d28f", SOFT_LILAC, SOFT_MINT, SOFT_GREY, "#ffd3b6"]

    for i, source in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            x=[pd.Timestamp(2023, m, 1).strftime("%b") for m in pivot.index],
            y=pivot[source], name=source, marker_color=palette[i % len(palette)],
            text=[f'${v:,.0f}' if v > 0 else '' for v in pivot[source]], textposition='inside'
        ))

    _apply_soft_layout(fig, "Profit by Booking Source")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Profit ($)", tickformat="$,.0f")
    fig.update_layout(barmode='stack')
    return fig, explanation


def _fixed_vs_variable_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    if {"FixedProfit", "VariableProfit"}.issubset(df.columns):
        g = df.groupby(df["Date"].dt.month)[["FixedProfit", "VariableProfit"]].sum()
        fixed_profit = g["FixedProfit"].sum()
        variable_profit = g["VariableProfit"].sum()
        fixed_col, var_col = "FixedProfit", "VariableProfit"
    elif {"FixedRoomCost", "VariableRoomCost"}.issubset(df.columns):
        cost = df.groupby(df["Date"].dt.month)[["FixedRoomCost", "VariableRoomCost"]].sum()
        rev = df.groupby(df["Date"].dt.month)["RoomRevenue"].sum()
        g = pd.DataFrame({"Fixed": rev - cost["VariableRoomCost"], "Variable": rev - cost["FixedRoomCost"]})
        fixed_profit = g["Fixed"].sum(); variable_profit = g["Variable"].sum()
        fixed_col, var_col = "Fixed", "Variable"
    else:
        fig = go.Figure()
        fig.add_annotation(text="No fixed/variable profit data", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=14, color=FONT_COLOR))
        _apply_soft_layout(fig, "Fixed vs Variable Profit")
        return fig, "Fixed/variable profit breakdown data is not available in this dataset."

    total_profit = fixed_profit + variable_profit
    variable_share = (variable_profit / total_profit) * 100 if total_profit > 0 else 0
    performance = _classify_performance(variable_share)

    explanation = (
        f"Variable profit accounts for {variable_share:.1f}% of total profit (**{performance}** flexibility)."
    )

    fig = go.Figure()
    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in g.index]

    fig.add_trace(go.Scatter(
        x=month_names, y=g[fixed_col],
        fill='tozeroy', mode='none', name=fixed_col,
        fillcolor='rgba(207,216,227,0.45)', line=dict(color=SOFT_GREY)
    ))
    fig.add_trace(go.Scatter(
        x=month_names, y=g[fixed_col] + g[var_col],
        fill='tonexty', mode='none', name=var_col,
        fillcolor='rgba(154,230,180,0.35)', line=dict(color=SOFT_GREEN)
    ))

    _apply_soft_layout(fig, "Fixed vs Variable Profit")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Profit ($)", tickformat="$,.0f")
    return fig, explanation


def _weekday_weekend_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    grp = df.groupby([df["Date"].dt.month, df["Date"].dt.weekday < 5])["Profit"].sum().unstack()
    grp.columns = ["Weekend", "Weekday"]

    weekend_total = grp["Weekend"].sum()
    weekday_total = grp["Weekday"].sum()
    weekend_daily = weekend_total / 2 if weekend_total else 0
    weekday_daily = weekday_total / 5 if weekday_total else 0
    uplift = ((weekend_daily - weekday_daily) / weekday_daily) * 100 if weekday_daily > 0 else 0
    performance = _classify_performance(abs(uplift))
    trend = "premium" if uplift > 0 else "discount"

    explanation = (
        f"Weekend profit shows a {performance} {trend} of {abs(uplift):.1f}% per day vs weekdays."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp.index, y=grp["Weekend"], name="Weekend",
        marker_color=SOFT_CORAL, text=[f'${v:,.0f}' if v > 0 else '' for v in grp["Weekend"]],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=grp.index, y=grp["Weekday"], name="Weekday",
        marker_color=SOFT_CYAN, text=[f'${v:,.0f}' if v > 0 else '' for v in grp["Weekday"]],
        textposition='outside'
    ))

    _apply_soft_layout(fig, "Weekday vs Weekend Profit")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Profit ($)", tickformat="$,.0f")
    fig.update_layout(barmode='group')
    return fig, explanation


def _profit_heatmap(df: pd.DataFrame) -> tuple[go.Figure, str]:
    # Average ProfitPerRoom by weekday (ordered Mon..Sun)
    dfc = df.copy()
    dfc["weekday"] = dfc["Date"].dt.weekday
    g = dfc.groupby("weekday")["ProfitPerRoom"].agg(mean="mean", count="count").reindex(range(7)).fillna(0).reset_index()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g["weekday_name"] = g["weekday"].map(dict(enumerate(weekday_names)))

    avg_profit = g["mean"].mean()
    max_profit = g["mean"].max(); min_profit = g["mean"].min()
    max_day = g.loc[g["mean"].idxmax(), "weekday_name"]
    min_day = g.loc[g["mean"].idxmin(), "weekday_name"]
    max_uplift = ((max_profit - avg_profit) / avg_profit) * 100 if avg_profit > 0 else 0
    performance = _classify_performance(max_uplift)

    explanation = (
        f"**{max_day}** is the most profitable (avg **${max_profit:.2f}** per room, {max_uplift:.1f}% above avg). "
        f"**{min_day}** is the lowest (**${min_profit:.2f}**)."
    )

    fig = go.Figure()
    colors = [
        (SOFT_CORAL if val < avg_profit else (SOFT_TEAL if val < max_profit * 0.9 else SOFT_GREEN))
        for val in g["mean"]
    ]

    fig.add_trace(go.Bar(
        x=g["weekday_name"], y=g["mean"], marker_color=colors,
        text=[f'${v:.0f}' for v in g["mean"]], textposition='outside',
        hovertemplate='<b>%{x}</b><br>Average Profit: $%{y:.2f}<br>Data Points: %{customdata}<extra></extra>',
        customdata=g["count"]
    ))
    fig.add_hline(y=avg_profit, line_dash="dash", line_color=SOFT_AMBER,
                  annotation_text=f"Weekly Average: ${avg_profit:.2f}",
                  annotation_position="top right")

    _apply_soft_layout(fig, "Profit per Room by Day of Week")
    fig.update_xaxes(title="Day of Week")
    fig.update_yaxes(title="Average Profit per Room ($)")
    fig.update_layout(showlegend=False)
    return fig, explanation


# ────────────────────────────────────────────────────────
# chart builders – room-type deep dive
# ────────────────────────────────────────────────────────
def _room_margin_bar(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType")[["RoomRevenue", "TotalRoomCost"]].sum()
    g["Margin%"] = (g["RoomRevenue"] - g["TotalRoomCost"]) / g["RoomRevenue"]

    best_margin = g["Margin%"].max() * 100
    worst_margin = g["Margin%"].min() * 100
    margin_spread = best_margin - worst_margin
    best_type = g["Margin%"].idxmax()
    worst_type = g["Margin%"].idxmin()
    performance = _classify_performance(margin_spread)

    explanation = (
        f"Profit margin ranges from {worst_margin:.1f}% ({worst_type}) to {best_margin:.1f}% ({best_type}), "
        f"a **{performance}** spread of {margin_spread:.1f} pts."
    )

    sorted_margins = g["Margin%"].sort_values()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_margins.values * 100, y=sorted_margins.index, orientation='h',
        marker_color=SOFT_MINT, text=[f'{v*100:.1f}%' for v in sorted_margins.values], textposition='outside'
    ))
    _apply_soft_layout(fig, "Room-Type Profit Margin")
    fig.update_xaxes(title="Profit Margin (%)", tickformat=".1f", ticksuffix="%")
    fig.update_yaxes(title="Room Type")
    fig.update_layout(margin=dict(l=150))
    return fig, explanation


def _pareto_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType")["Profit"].sum().sort_values(ascending=False)
    pct = g.cumsum() / g.sum()

    room_count = len(g)
    top_20_cnt = max(1, int(room_count * 0.2))
    top_20_share = pct.iloc[top_20_cnt-1] * 100 if not pct.empty else 0
    performance = _classify_performance(top_20_share)
    pareto_strength = "close to Pareto principle" if abs(top_20_share - 80) < 10 else "deviates from Pareto"

    explanation = (
        f"Top 20% room types generate {top_20_share:.1f}% of profit (**{performance}**; {pareto_strength})."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=g.index, y=g.values, name="Profit", marker_color="#93c5fd",
        text=[f'${v:,.0f}' for v in g.values], textposition='outside'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pct.index, y=pct.values * 100, mode='lines+markers', name="Cumulative %",
        line=dict(color=SOFT_AMBER, width=3),
        marker=dict(size=7, line=dict(color="white", width=1))
    ), secondary_y=True)

    _apply_soft_layout(fig, "Profit Contribution Pareto")
    fig.update_xaxes(title="Room Type")
    fig.update_yaxes(title_text="Profit ($)", tickformat="$,.0f", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", tickformat=".0f", ticksuffix="%", secondary_y=True)
    return fig, explanation


def _adr_profit_bubble(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType").agg(
        ADR=("ADR", "mean"),
        Profit=("Profit", "sum"),
        OccRooms=("OccupiedRooms", "sum"),
    )
    g["ProfitPerRoom"] = g["Profit"] / g["OccRooms"].replace(0, np.nan)

    corr = g["ADR"].corr(g["ProfitPerRoom"])
    corr_pct = abs(corr) * 100
    performance = _classify_performance(corr_pct)
    relationship = "positive" if corr > 0 else "negative"
    implication = "higher rates tend to yield higher profits" if corr > 0 else "higher rates don't necessarily translate to higher profits"

    explanation = (
        f"Correlation between ADR and profit per room is {corr:.2f} (**{performance}**, {relationship}). "
        f"This means {implication}. Bubble size = occupied rooms."
    )

    max_rooms = g["OccRooms"].max() if g["OccRooms"].max() > 0 else 1
    sizes = (g["OccRooms"] / max_rooms) * 40 + 20

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["ADR"], y=g["ProfitPerRoom"], mode='markers+text',
        marker=dict(size=sizes, color=SOFT_BLUE, opacity=0.6, line=dict(width=2, color="#0369a1")),
        text=g.index, textposition="middle center", textfont=dict(size=10, color='white'),
        name="Room Types",
        hovertemplate="<b>%{text}</b><br>ADR: $%{x:,.0f}<br>Profit per Room: $%{y:,.0f}<br>Occupied Rooms: %{customdata:,.0f}<extra></extra>",
        customdata=g["OccRooms"]
    ))
    _apply_soft_layout(fig, "ADR vs Profit per Room")
    fig.update_xaxes(title="ADR ($)", tickformat="$,.0f")
    fig.update_yaxes(title="Profit per Room ($)", tickformat="$,.0f")
    return fig, explanation


def _upsell_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    if "UpsellRevenue" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="UpsellRevenue column missing", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=14, color=FONT_COLOR))
        _apply_soft_layout(fig, "Upsell Profit Lift")
        return fig, "UpsellRevenue data is not available in this dataset."

    g = df.groupby("RoomType")[["UpsellRevenue", "Profit"]].sum()
    lift = (g["UpsellRevenue"] / g["Profit"].replace(0, np.nan)).sort_values()

    avg_lift = (lift.mean() * 100) if not lift.empty else 0
    performance = _classify_performance(avg_lift)
    best_lift = (lift.max() * 100) if not lift.empty else 0
    best_type = lift.idxmax() if not lift.empty else "N/A"

    explanation = (
        f"On average, upsell adds {avg_lift:.1f}% to room profit (**{performance}**). "
        f"Best: {best_type} ({best_lift:.1f}%)."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lift.index, y=lift.values * 100, marker_color="#f6ad55",  # soft orange
        text=[f'{v*100:.1f}%' for v in lift.values], textposition='outside'
    ))
    _apply_soft_layout(fig, "Upsell Profit Lift")
    fig.update_xaxes(title="Room Type", tickangle=45)
    fig.update_yaxes(title="Upsell Revenue / Profit (%)", tickformat=".1f", ticksuffix="%")
    return fig, explanation


# ────────────────────────────────────────────────────────
# UI stub generator (modern UI)
# ────────────────────────────────────────────────────────
def _build_page(
    title: str,
    chart_spec: list[tuple[str, callable, set[str]]],
    base_df: pd.DataFrame,
) -> QWidget:
    root = QWidget()
    root.setObjectName("ProfitRoot")
    root.setLayout(QVBoxLayout())
    root.layout().setContentsMargins(12, 12, 12, 12)
    root.layout().setSpacing(10)

    # Global QSS (same design language)
    root.setStyleSheet("""
        QWidget#ProfitRoot {
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
        QDateEdit::drop-down { width: 18px; }
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

    # Header
    header = QLabel(title)
    header.setObjectName("header")
    root.layout().addWidget(header)

    # Toolbar (date range) as a card
    toolbar = QFrame()
    toolbar.setObjectName("toolbar")
    tl = QHBoxLayout(toolbar)
    tl.setContentsMargins(12, 10, 12, 10)
    tl.setSpacing(8)

    min_d, max_d = base_df["Date"].min().date(), base_df["Date"].max().date()
    start_picker = QDateEdit(QDate(min_d.year, min_d.month, min_d.day))
    end_picker   = QDateEdit(QDate(max_d.year, max_d.month, max_d.day))
    for p in (start_picker, end_picker):
        p.setCalendarPopup(True)

    tl.addWidget(QLabel("Date Range:"))
    tl.addWidget(start_picker)
    tl.addWidget(QLabel(" to "))
    tl.addWidget(end_picker)
    tl.addStretch()

    apply_btn = QPushButton("Apply")
    apply_btn.setObjectName("applyBtn")
    tl.addWidget(apply_btn)
    root.layout().addWidget(toolbar)

    # Tabs
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    def _render():
        start = pd.Timestamp(start_picker.date().toPython())
        end   = pd.Timestamp(end_picker.date().toPython())
        df = _mask(base_df, start, end)
        _add_profit_cols(df)

        tabs.clear()
        added = False
        for t, builder, req in chart_spec:
            _maybe(tabs, t, builder, df, req)
            if tabs.count() > 0:
                added = True

        if not added:
            info_card = QFrame()
            info_card.setObjectName("card")
            info_card.setStyleSheet("""
                QFrame#card {
                    background: #ffffff;
                    border: 1px solid #e5e9f2;
                    border-radius: 12px;
                }
            """)
            v = QVBoxLayout(info_card)
            v.setContentsMargins(16, 16, 16, 16)
            lbl = QLabel("No data / missing columns for selected range.")
            lbl.setStyleSheet("font-size: 14px; color: #8b5e34;")
            v.addWidget(lbl)
            wrap = QWidget()
            wlay = QVBoxLayout(wrap)
            wlay.addWidget(info_card)
            tabs.addTab(wrap, "Info")
            tabs.setTabEnabled(0, True)

    apply_btn.clicked.connect(_render)
    _render()
    return root


# ────────────────────────────────────────────────────────
# PUBLIC ENTRY – overall profit page
# ────────────────────────────────────────────────────────
@data_required
def display_room_profit() -> QWidget:
    base_df = get_df()
    charts = [
        ("Trend", _trend_rev_cost_profit, {"RoomRevenue", "TotalRoomCost", "Date"}),
        ("By Source", _profit_by_source, {"BookingSource", "Profit", "Date"}),
        ("Fixed vs Variable", _fixed_vs_variable_profit, {"RoomRevenue", "Date"}),  # alt paths handled
        ("Weekday/Weekend", _weekday_weekend_profit, {"Profit", "Date"}),
        ("Weekly Profit Analysis", _profit_heatmap, {"ProfitPerRoom", "Date"}),
    ]
    return _build_page("Room Profit Analysis", charts, base_df)


# ────────────────────────────────────────────────────────
# PUBLIC ENTRY – room-type deep dive
# ────────────────────────────────────────────────────────
@data_required
def display_room_type_profitability() -> QWidget:
    base_df = get_df()
    charts = [
        ("Margins by Type", _room_margin_bar, {"RoomRevenue", "TotalRoomCost", "RoomType"}),
        ("Profit Pareto", _pareto_profit, {"RoomType", "Profit"}),
        ("ADR vs Profit", _adr_profit_bubble, {"RoomType", "ADR", "Profit", "OccupiedRooms"}),
        ("Upsell Lift", _upsell_profit, {"RoomType", "UpsellRevenue", "Profit"}),
    ]
    return _build_page("Room-Type Profitability", charts, base_df)
