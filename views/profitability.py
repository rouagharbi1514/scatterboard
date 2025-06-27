# views/profitability.py
"""
Room-level Profit Analysis
==========================

* Two entry points wired in ROUTES:
      • display_room_profit()               – macro profit view
      • display_room_type_profitability()   – room-type deep dive
* Identical date-range picker pattern used in Seasonality / Room Cost
* Charts auto-hide when required columns are missing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDateEdit,
    QPushButton,
    QTabWidget,
    QSizePolicy,
)

from data.helpers import get_df  # cached DataFrame loader
from views.utils import data_required, create_plotly_widget  # decorator already in project


# ────────────────────────────────────────────────────────
# reusable helpers
# ────────────────────────────────────────────────────────
def _mask(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def _add_profit_cols(df: pd.DataFrame) -> None:
    if "Profit" not in df.columns and {"RoomRevenue", "TotalRoomCost"}.issubset(
        df.columns
    ):
        df["Profit"] = df["RoomRevenue"] - df["TotalRoomCost"]
    if "ProfitPerRoom" not in df.columns and {"Profit", "OccupiedRooms"}.issubset(
        df.columns
    ):
        df["ProfitPerRoom"] = df["Profit"] / df["OccupiedRooms"].replace(0, np.nan)


def _classify_performance(value: float) -> str:
    """Classify performance based on value percentage."""
    if value >= 30:
        return "strong"
    elif value >= 15:
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


def _maybe(tab: QTabWidget, title: str, func, df: pd.DataFrame, cols: set[str]):
    """Add a tab with chart and explanation if required columns exist."""
    if cols.issubset(df.columns) and not df.empty:
        fig, explanation = func(df)
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.addWidget(create_plotly_widget(fig))
        tab_layout.addWidget(_collapsible(explanation))
        tab.addTab(tab_widget, title)


# ────────────────────────────────────────────────────────
# chart builders – macro profit view
# ────────────────────────────────────────────────────────
def _trend_rev_cost_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby(df["Date"].dt.month)[["RoomRevenue", "TotalRoomCost"]].sum()
    g["Profit"] = g["RoomRevenue"] - g["TotalRoomCost"]

    # Calculate KPI - overall profit margin
    total_revenue = g["RoomRevenue"].sum()
    total_profit = g["Profit"].sum()
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
    performance = _classify_performance(profit_margin)

    # Calculate trend
    profit_growth = ((g["Profit"].iloc[-1] - g["Profit"].iloc[0]) / g["Profit"].iloc[0]) * 100 if len(g) > 1 and g["Profit"].iloc[0] != 0 else 0
    trend = "increasing" if profit_growth > 0 else "decreasing"

    explanation = (
        f"Overall profit margin is {profit_margin:.1f}%, which represents **{performance}** performance. "
        f"Profit shows a {trend} trend of {abs(profit_growth):.1f}% over the period. "
        f"A widening gap between the green (revenue) and red (cost) areas indicates expanding profit margin."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add revenue area
    fig.add_trace(go.Scatter(
        x=g.index,
        y=g["RoomRevenue"],
        fill='tozeroy',
        mode='none',
        name='Revenue',
        fillcolor='rgba(74, 222, 128, 0.3)',  # Green with transparency
        line=dict(color='rgba(74, 222, 128, 1)')
    ))

    # Add cost area
    fig.add_trace(go.Scatter(
        x=g.index,
        y=g["TotalRoomCost"],
        fill='tozeroy',
        mode='none',
        name='Cost',
        fillcolor='rgba(239, 68, 68, 0.3)',  # Red with transparency
        line=dict(color='rgba(239, 68, 68, 1)')
    ))

    # Add profit line
    fig.add_trace(go.Scatter(
        x=g.index,
        y=g["Profit"],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Room Revenue • Cost • Profit",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_white",
        height=500,
        width=800,
        yaxis=dict(tickformat="$,.0f"),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=[str(i) for i in range(1, 13)]
        )
    )

    return fig, explanation


def _profit_by_source(df: pd.DataFrame) -> tuple[go.Figure, str]:
    pivot = df.groupby([df["Date"].dt.month, "BookingSource"])["Profit"]
    pivot = pivot.sum().unstack(fill_value=0)

    # Calculate KPI - source with highest profit contribution
    total_by_source = pivot.sum()
    top_source = total_by_source.idxmax() if not total_by_source.empty else "N/A"
    top_share = (total_by_source.max() / total_by_source.sum()) * 100 if total_by_source.sum() > 0 else 0
    performance = _classify_performance(top_share)

    # Calculate diversity (concentration)
    source_count = len(total_by_source)
    top_3_share = total_by_source.nlargest(min(3, source_count)).sum() / total_by_source.sum() * 100 if total_by_source.sum() > 0 else 0

    explanation = (
        f"'{top_source}' is the top profit source with {top_share:.1f}% of total profit, "
        f"representing **{performance}** channel concentration. "
        f"The top 3 sources generate {top_3_share:.1f}% of profit. "
        f"High dependence on few sources may indicate opportunity for channel diversification."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Define colors for different booking sources
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Add stacked bars for each booking source
    for i, source in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[source],
            name=source,
            marker_color=colors[i % len(colors)],
            text=[f'${val:,.0f}' if val > 0 else '' for val in pivot[source]],
            textposition='inside'
        ))

    fig.update_layout(
        title="Profit by Booking Source",
        xaxis_title="Month",
        yaxis_title="Profit ($)",
        barmode='stack',
        template="plotly_white",
        height=500,
        width=800,
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    return fig, explanation


def _fixed_vs_variable_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    if {"FixedProfit", "VariableProfit"}.issubset(df.columns):
        g = df.groupby(df["Date"].dt.month)[["FixedProfit", "VariableProfit"]].sum()
        fixed_profit = g["FixedProfit"].sum()
        variable_profit = g["VariableProfit"].sum()
    else:
        # Heuristic: assume profit splits same share as costs if Cost columns
        # exist
        if {"FixedRoomCost", "VariableRoomCost"}.issubset(df.columns):
            cost = df.groupby(df["Date"].dt.month)[
                ["FixedRoomCost", "VariableRoomCost"]
            ].sum()
            rev = df.groupby(df["Date"].dt.month)["RoomRevenue"].sum()
            g = pd.DataFrame(
                {
                    "Fixed": rev - cost["VariableRoomCost"],
                    "Variable": rev - cost["FixedRoomCost"],
                }
            )
            fixed_profit = g["Fixed"].sum()
            variable_profit = g["Variable"].sum()
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No fixed/variable profit data",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(height=500)
            return fig, "Fixed/variable profit breakdown data is not available in this dataset."

    # Calculate KPI - variable profit share
    total_profit = fixed_profit + variable_profit
    variable_share = (variable_profit / total_profit) * 100 if total_profit > 0 else 0
    performance = _classify_performance(variable_share)

    explanation = (
        f"Variable profit accounts for {variable_share:.1f}% of total profit, which is a **{performance}** "
        f"level of operational flexibility. A higher share of variable profit typically indicates "
        f"better ability to adapt to changing demand and market conditions."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add fixed profit area
    fig.add_trace(go.Scatter(
        x=g.index,
        y=g.iloc[:, 0],  # Fixed profit
        fill='tozeroy',
        mode='none',
        name=g.columns[0],
        fillcolor='rgba(156, 163, 175, 0.3)',  # Gray with transparency
        line=dict(color='rgba(156, 163, 175, 1)')
    ))

    # Add variable profit area (stacked)
    fig.add_trace(go.Scatter(
        x=g.index,
        y=g.iloc[:, 0] + g.iloc[:, 1],  # Fixed + Variable for stacking
        fill='tonexty',
        mode='none',
        name=g.columns[1],
        fillcolor='rgba(74, 222, 128, 0.3)',  # Green with transparency
        line=dict(color='rgba(74, 222, 128, 1)')
    ))

    fig.update_layout(
        title="Fixed vs Variable Profit",
        xaxis_title="Month",
        yaxis_title="Profit ($)",
        template="plotly_white",
        height=500,
        width=800,
        yaxis=dict(tickformat="$,.0f"),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=[str(i) for i in range(1, 13)]
        )
    )

    return fig, explanation


def _weekday_weekend_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    grp = (
        df.groupby([df["Date"].dt.month, df["Date"].dt.weekday < 5])["Profit"]
        .sum()
        .unstack()
    )
    grp.columns = ["Weekend", "Weekday"]

    # Calculate KPI - weekend vs weekday profit uplift
    weekend_total = grp["Weekend"].sum()
    weekday_total = grp["Weekday"].sum()

    # Normalize by count of days (approx. 5/7 weekdays, 2/7 weekend)
    weekend_daily = weekend_total / 2 if weekend_total else 0
    weekday_daily = weekday_total / 5 if weekday_total else 0

    if weekday_daily > 0:
        uplift = ((weekend_daily - weekday_daily) / weekday_daily) * 100
    else:
        uplift = 0

    performance = _classify_performance(abs(uplift))
    trend = "premium" if uplift > 0 else "discount"

    explanation = (
        f"Weekend profit shows a {performance} {trend} of {abs(uplift):.1f}% per day "
        f"compared to weekdays. This indicates {'higher' if uplift > 0 else 'lower'} "
        f"revenue potential on weekends, which may suggest opportunities for "
        f"{'optimizing weekday rates' if uplift > 0 else 'weekend promotions'}."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add weekend bars
    fig.add_trace(go.Bar(
        x=grp.index,
        y=grp["Weekend"],
        name="Weekend",
        marker_color='#f97316',  # Orange
        text=[f'${val:,.0f}' if val > 0 else '' for val in grp["Weekend"]],
        textposition='outside'
    ))

    # Add weekday bars
    fig.add_trace(go.Bar(
        x=grp.index,
        y=grp["Weekday"],
        name="Weekday",
        marker_color='#3b82f6',  # Blue
        text=[f'${val:,.0f}' if val > 0 else '' for val in grp["Weekday"]],
        textposition='outside'
    ))

    fig.update_layout(
        title="Weekday vs Weekend Profit",
        xaxis_title="Month",
        yaxis_title="Profit ($)",
        template="plotly_white",
        height=500,
        width=800,
        yaxis=dict(tickformat="$,.0f"),
        barmode='group',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=[str(i) for i in range(1, 13)]
        )
    )

    return fig, explanation


def _profit_heatmap(df: pd.DataFrame) -> tuple[go.Figure, str]:
    heat = df.pivot_table(
        index=df["Date"].dt.weekday,
        columns=df["Date"].dt.month,
        values="ProfitPerRoom",
        aggfunc="mean",
    )

    # Calculate KPI - highest vs average profit cell
    avg_profit = heat.mean().mean()
    max_profit = heat.max().max()
    max_i, max_j = np.unravel_index(heat.values.argmax(), heat.shape)

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    max_day = weekday_names[max_i]
    max_month = max_j + 1  # Adjust for 1-based month

    if avg_profit > 0:
        max_uplift = ((max_profit - avg_profit) / avg_profit) * 100
    else:
        max_uplift = 0

    performance = _classify_performance(max_uplift)

    explanation = (
        f"The highest profit per room (${max_profit:.2f}) occurs on {max_day}s in month {max_month}, "
        f"which is {max_uplift:.1f}% above average (${avg_profit:.2f}) - a **{performance}** uplift. "
        f"Darker cells represent higher profit opportunities, suggesting potential for targeted pricing strategies."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=heat.values,
        x=[f"Month {i}" for i in heat.columns],
        y=weekday_names,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title="Profit per Room ($)"
        ),
        text=[[f"${val:.0f}" if not pd.isna(val) else "" for val in row] for row in heat.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Profit per Room Heatmap",
        xaxis_title="Month",
        yaxis_title="Day of Week",
        template="plotly_white",
        height=500,
        width=800,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")  # Monday at top
    )

    return fig, explanation


# ────────────────────────────────────────────────────────
# chart builders – room-type deep dive
# ────────────────────────────────────────────────────────
def _room_margin_bar(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType")[["RoomRevenue", "TotalRoomCost"]].sum()
    g["Margin%"] = (g["RoomRevenue"] - g["TotalRoomCost"]) / g["RoomRevenue"]

    # Calculate KPI - best vs worst margin spread
    best_margin = g["Margin%"].max() * 100
    worst_margin = g["Margin%"].min() * 100
    margin_spread = best_margin - worst_margin

    best_type = g["Margin%"].idxmax()
    worst_type = g["Margin%"].idxmin()

    performance = _classify_performance(margin_spread)

    explanation = (
        f"Profit margin ranges from {worst_margin:.1f}% ({worst_type}) to {best_margin:.1f}% ({best_type}), "
        f"a **{performance}** margin spread of {margin_spread:.1f} percentage points. "
        f"Room types with margins below 15% may require pricing review or cost optimization."
    )

    # Create Plotly figure
    sorted_margins = g["Margin%"].sort_values()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sorted_margins.values * 100,  # Convert to percentage
        y=sorted_margins.index,
        orientation='h',
        marker_color='#4ade80',  # Green
        text=[f'{val:.1f}%' for val in sorted_margins.values * 100],
        textposition='outside'
    ))

    fig.update_layout(
        title="Room-Type Profit Margin",
        xaxis_title="Profit Margin (%)",
        yaxis_title="Room Type",
        template="plotly_white",
        height=500,
        width=800,
        xaxis=dict(tickformat=".1f", ticksuffix="%"),
        margin=dict(l=150)  # Extra left margin for room type labels
    )

    return fig, explanation


def _pareto_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType")["Profit"].sum().sort_values(ascending=False)
    pct = g.cumsum() / g.sum()

    # Calculate KPI - top 20% room types' share of total profit
    room_count = len(g)
    top_20_pct_count = max(1, int(room_count * 0.2))
    top_20_pct_share = pct.iloc[top_20_pct_count-1] * 100 if not pct.empty else 0

    performance = _classify_performance(top_20_pct_share)
    pareto_strength = "close to Pareto principle" if abs(top_20_pct_share - 80) < 10 else "deviates from Pareto principle"

    explanation = (
        f"The top 20% of room types generate {top_20_pct_share:.1f}% of total profit, "
        f"a **{performance}** concentration that {pareto_strength}. "
        f"This suggests {'focusing investment on top performers' if top_20_pct_share > 70 else 'a more balanced profit distribution'}."
    )

    # Create Plotly figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add profit bars
    fig.add_trace(
        go.Bar(
            x=g.index,
            y=g.values,
            name="Profit",
            marker_color='#60a5fa',  # Blue
            text=[f'${val:,.0f}' for val in g.values],
            textposition='outside',
            yaxis='y'
        ),
        secondary_y=False,
    )

    # Add cumulative percentage line
    fig.add_trace(
        go.Scatter(
            x=pct.index,
            y=pct.values * 100,  # Convert to percentage
            mode='lines+markers',
            name="Cumulative %",
            line=dict(color='#facc15', width=3),  # Yellow
            marker=dict(size=8),
            yaxis='y2'
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Profit Contribution Pareto",
        template="plotly_white",
        height=500,
        width=800,
        xaxis=dict(title="Room Type"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Profit ($)", tickformat="$,.0f", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", tickformat=".0f", ticksuffix="%", secondary_y=True)

    return fig, explanation


def _adr_profit_bubble(df: pd.DataFrame) -> tuple[go.Figure, str]:
    g = df.groupby("RoomType").agg(
        ADR=("ADR", "mean"),
        ProfitPerRoom=("Profit", "sum"),
        OccRooms=("OccupiedRooms", "sum"),
    )
    g["ProfitPerRoom"] = g["ProfitPerRoom"] / g["OccRooms"].replace(0, np.nan)

    # Calculate KPI - correlation between ADR and profit per room
    corr = g["ADR"].corr(g["ProfitPerRoom"])
    corr_pct = abs(corr) * 100

    performance = _classify_performance(corr_pct)
    relationship = "positive" if corr > 0 else "negative"
    implication = "higher rates tend to yield higher profits" if corr > 0 else "higher rates don't necessarily translate to higher profits"

    explanation = (
        f"The correlation between ADR and profit is {corr:.2f}, a **{performance}** {relationship} relationship. "
        f"This means {implication}. The bubble size represents occupied room volume, "
        f"highlighting the most impactful room types."
    )

    # Create Plotly figure
    fig = go.Figure()

    # Normalize bubble sizes (between 20 and 60)
    max_rooms = g["OccRooms"].max()
    sizes = (g["OccRooms"] / max_rooms) * 40 + 20

    fig.add_trace(go.Scatter(
        x=g["ADR"],
        y=g["ProfitPerRoom"],
        mode='markers+text',
        marker=dict(
            size=sizes,
            color='#38bdf8',  # Light blue
            opacity=0.6,
            line=dict(width=2, color='#0369a1')  # Darker blue border
        ),
        text=g.index,  # Room type labels
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        name="Room Types",
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "ADR: $%{x:,.0f}<br>" +
            "Profit per Room: $%{y:,.0f}<br>" +
            "Occupied Rooms: %{customdata:,.0f}<br>" +
            "<extra></extra>"
        ),
        customdata=g["OccRooms"]
    ))

    fig.update_layout(
        title="ADR vs Profit per Room",
        xaxis_title="ADR ($)",
        yaxis_title="Profit per Room ($)",
        template="plotly_white",
        height=500,
        width=800,
        xaxis=dict(tickformat="$,.0f"),
        yaxis=dict(tickformat="$,.0f")
    )

    return fig, explanation


def _upsell_profit(df: pd.DataFrame) -> tuple[go.Figure, str]:
    if "UpsellRevenue" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="UpsellRevenue column missing",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=500)
        return fig, "UpsellRevenue data is not available in this dataset."

    g = df.groupby("RoomType")[["UpsellRevenue", "Profit"]].sum()
    lift = g["UpsellRevenue"] / g["Profit"].replace(0, np.nan)

    # Calculate KPI - average upsell $ per $ profit
    avg_lift = lift.mean() * 100 if not lift.empty else 0
    performance = _classify_performance(avg_lift)

    # Find best performing room type
    best_lift = lift.max() * 100 if not lift.empty else 0
    best_type = lift.idxmax() if not lift.empty else "N/A"

    explanation = (
        f"On average, upsell revenue adds {avg_lift:.1f}% to room profit, a **{performance}** contribution. "
        f"The '{best_type}' room type performs best with {best_lift:.1f}% upsell lift. "
        f"Higher bars suggest room types with better upselling opportunities."
    )

    # Create Plotly figure
    sorted_lift = lift.sort_values()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sorted_lift.index,
        y=sorted_lift.values * 100,  # Convert to percentage
        marker_color='#f97316',  # Orange
        text=[f'{val:.1f}%' for val in sorted_lift.values * 100],
        textposition='outside'
    ))

    fig.update_layout(
        title="Upsell Profit Lift",
        xaxis_title="Room Type",
        yaxis_title="Upsell Revenue / Profit (%)",
        template="plotly_white",
        height=500,
        width=800,
        yaxis=dict(tickformat=".1f", ticksuffix="%"),
        xaxis=dict(tickangle=45)
    )

    return fig, explanation


# ────────────────────────────────────────────────────────
# UI stub generator
# ────────────────────────────────────────────────────────
def _build_page(
    title: str,
    chart_spec: list[tuple[str, callable, set[str]]],
    base_df: pd.DataFrame,
) -> QWidget:
    root = QWidget()
    root.setLayout(QVBoxLayout())
    lbl = QLabel(title)
    lbl.setStyleSheet("font-size:18pt;font-weight:bold;")
    root.layout().addWidget(lbl)

    # date pickers
    min_d, max_d = base_df["Date"].min().date(), base_df["Date"].max().date()
    start_picker = QDateEdit(QDate(min_d.year, min_d.month, min_d.day))
    end_picker = QDateEdit(QDate(max_d.year, max_d.month, max_d.day))
    for p in (start_picker, end_picker):
        p.setCalendarPopup(True)

    row = QHBoxLayout()
    row.addWidget(QLabel("Date Range:"))
    row.addWidget(start_picker)
    row.addWidget(QLabel(" to "))
    row.addWidget(end_picker)
    apply_btn = QPushButton("Apply")
    row.addWidget(apply_btn)
    row.addStretch()
    root.layout().addLayout(row)

    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    def _render():
        start = pd.Timestamp(start_picker.date().toPython())
        end = pd.Timestamp(end_picker.date().toPython())
        df = _mask(base_df, start, end)
        _add_profit_cols(df)
        tabs.clear()
        for t, builder, req in chart_spec:
            _maybe(tabs, t, builder, df, req)

        if tabs.count() == 0:
            msg = QLabel("No data / missing columns for selected range.")
            msg.setStyleSheet("font-size:14pt;color:#f97316;")
            root.layout().addWidget(msg)

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
        (
            "Fixed vs Variable",
            _fixed_vs_variable_profit,
            {"RoomRevenue", "Date"},
        ),  # helper handles alt paths
        ("Weekday/Weekend", _weekday_weekend_profit, {"Profit", "Date"}),
        ("Profit Heatmap", _profit_heatmap, {"ProfitPerRoom", "Date"}),
    ]
    return _build_page("Room Profit Analysis", charts, base_df)


# ────────────────────────────────────────────────────────
# PUBLIC ENTRY – room-type deep dive
# ────────────────────────────────────────────────────────
@data_required
def display_room_type_profitability() -> QWidget:
    base_df = get_df()
    charts = [
        (
            "Margins by Type",
            _room_margin_bar,
            {"RoomRevenue", "TotalRoomCost", "RoomType"},
        ),
        ("Profit Pareto", _pareto_profit, {"RoomType", "Profit"}),
        (
            "ADR vs Profit",
            _adr_profit_bubble,
            {"RoomType", "ADR", "Profit", "OccupiedRooms"},
        ),
        ("Upsell Lift", _upsell_profit, {"RoomType", "UpsellRevenue", "Profit"}),
    ]
    return _build_page("Room-Type Profitability", charts, base_df)
