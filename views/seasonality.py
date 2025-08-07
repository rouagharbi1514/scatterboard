# flake8: noqa
# views/seasonality.py
"""
Seasonality Analysis view
-------------------------
* Date-range picker (start / end) + Apply button
* Eight charts that matter (see spec below)
* Graceful degradation if columns are absent
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional

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

from data.helpers import get_df  # global cached DataFrame loader
from views.utils import data_required, create_plotly_widget  # decorator you already use


# ──────────────────────────────────────────────────────────────
# helper utilities
# ──────────────────────────────────────────────────────────────
def _mask_by_date(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    # Check if 'Date' or 'date' column exists and use the correct one
    date_col = 'Date' if 'Date' in df.columns else 'date'

    # Return a copy of the filtered DataFrame
    if date_col in df.columns:
        return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
    else:
        # Return empty DataFrame if no date column exists
        print(f"Warning: No '{date_col}' column found in the DataFrame")
        return pd.DataFrame(columns=df.columns)


def _add_season_col(df: pd.DataFrame) -> None:
    """Add a 'Season' column in-place if not present."""
    if "Season" not in df.columns:
        df["Season"] = pd.cut(
            df["Date"].dt.month,
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
            right=False,
        )


def _classify_performance(value: float) -> str:
    """Classify performance based on value percentage."""
    if value >= 30:
        return "strong"
    elif value >= 15:
        return "moderate"
    else:
        return "weak"


def _required(
    tabs: QTabWidget,
    title: str,
    builder: Callable[[pd.DataFrame], Tuple[go.Figure, str]],
    df: pd.DataFrame,
    cols: set[str],
) -> Optional[QWidget]:
    """Add chart tab only if all required columns exist & non-empty."""
    if cols.issubset(df.columns) and not df.empty:
        fig, explanation = builder(df)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(create_plotly_widget(fig))
        tabs.addTab(tab, title)
        return tab
    return None


# ──────────────────────────────────────────────────────────────
# chart builders
# return a plotly Figure and explanation text
# ──────────────────────────────────────────────────────────────
def _monthly_occ(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    # First check if we have the right column
    occ_col = None
    if "OccupancyRate" in df.columns:
        occ_col = "OccupancyRate"
    elif "occupancy" in df.columns:
        occ_col = "occupancy"
    elif "Occupancy" in df.columns:
        occ_col = "Occupancy"

    if occ_col is None:
        print("Warning: No occupancy column found in the data")
        fig = go.Figure()
        fig.add_annotation(
            text="Occupancy data not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(
            title="Monthly Occupancy",
            showlegend=False,
            height=500
        )
        return fig, "Occupancy data not available."

    # Create a copy to avoid modifying original data
    df_copy = df.copy()

    # Normalize occupancy to percentage (0-1 range)
    if df_copy[occ_col].max() > 1:
        # If occupancy is already in percentage (e.g., 75 for 75%)
        df_copy[occ_col] = df_copy[occ_col] / 100

    # Group by month
    grp = df_copy.groupby(df_copy["Date"].dt.month)[occ_col]
    m = grp.mean()
    std = grp.std()

    # Calculate KPI
    avg_occ = m.mean() * 100
    std_ratio = (std.mean() / m.mean()) * 100 if m.mean() > 0 else 0
    seasonality_class = _classify_performance(std_ratio)

    high_month = m.idxmax()
    high_month_name = pd.Timestamp(2023, high_month, 1).strftime("%B")
    low_month = m.idxmin()
    low_month_name = pd.Timestamp(2023, low_month, 1).strftime("%B")
    month_diff_pct = ((m.max() - m.min()) / m.min() * 100) if m.min() > 0 else 0

    # Build explanation text
    explanation = (
        f"Average occupancy is {avg_occ:.1f}%, with {seasonality_class} demand seasonality "
        f"(±{std_ratio:.1f}% variability). "
        f"{high_month_name} has the highest occupancy at {m.max()*100:.1f}%, while "
        f"{low_month_name} has the lowest at {m.min()*100:.1f}% "
        f"({month_diff_pct:.1f}% difference)."
    )

    # Create Plotly figure
    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in m.index]

    fig = go.Figure()

    # Add occupancy line with error bars
    fig.add_trace(go.Scatter(
        x=month_names,
        y=m.values * 100,  # Convert to percentage
        mode='lines+markers',
        name="Avg Occupancy",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8),
        error_y=dict(
            type='data',
            array=std.values * 100,
            visible=True,
            color='rgba(52, 152, 219, 0.3)'
        )
    ))

    # Add shaded area for standard deviation
    upper_bound = (m + std).values * 100
    lower_bound = (m - std).values * 100

    fig.add_trace(go.Scatter(
        x=month_names + month_names[::-1],
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name="±1σ Range",
        showlegend=True
    ))

    fig.update_layout(
        title="Monthly Occupancy Trend",
        xaxis_title="Month",
        yaxis_title="Occupancy (%)",
        showlegend=True,
        height=500,
        width=800,
        plot_bgcolor='white',
        font=dict(size=10)
    )

    return fig, explanation


def _adr_revpar(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    m = df.groupby(df["Date"].dt.month)[["ADR", "RevPAR"]].mean()

    # Calculate KPIs
    avg_adr = m["ADR"].mean()
    avg_revpar = m["RevPAR"].mean()
    high_season_months = m["RevPAR"].nlargest(3).index
    low_season_months = m["RevPAR"].nsmallest(3).index

    high_revpar = m.loc[high_season_months, "RevPAR"].mean()
    low_revpar = m.loc[low_season_months, "RevPAR"].mean()
    revpar_uplift = ((high_revpar - low_revpar) / low_revpar) * 100
    seasonality_class = _classify_performance(revpar_uplift)

    # Format high/low season months as text
    high_months_text = ", ".join([pd.Timestamp(2023, month, 1).strftime("%b") for month in high_season_months])
    low_months_text = ", ".join([pd.Timestamp(2023, month, 1).strftime("%b") for month in low_season_months])

    # Build explanation text
    explanation = (
        f"RevPAR shows {seasonality_class} seasonal variation with {revpar_uplift:.1f}% higher values "
        f"in peak months ({high_months_text}) compared to low season ({low_months_text}). "
        f"Average ADR is ${avg_adr:.2f} and RevPAR is ${avg_revpar:.2f}."
    )

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in m.index]

    # Add ADR and RevPAR lines
    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=m["ADR"],
            mode='lines+markers',
            name="ADR ($)",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8, symbol="circle")
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=month_names,
            y=m["RevPAR"],
            mode='lines+markers',
            name="RevPAR ($)",
            line=dict(color="#3498db", width=3),
            marker=dict(size=8, symbol="square")
        ),
        secondary_y=False,
    )

    # Add occupancy on secondary y-axis if available
    if "OccupancyRate" in df.columns:
        occ_data = df.groupby(df["Date"].dt.month)["OccupancyRate"].mean()
        fig.add_trace(
            go.Scatter(
                x=month_names,
                y=occ_data * 100,  # Convert to percentage
                mode='lines+markers',
                name="Occupancy (%)",
                line=dict(color="#27ae60", width=2, dash="dash"),
                marker=dict(size=6, symbol="triangle-up")
            ),
            secondary_y=True,
        )

    # Set x-axis title
    fig.update_xaxes(title_text="Month")

    # Set y-axes titles
    fig.update_yaxes(title_text="ADR / RevPAR ($)", secondary_y=False)
    fig.update_yaxes(title_text="Occupancy (%)", secondary_y=True)

    fig.update_layout(
        title="ADR & RevPAR Seasonality",
        height=500,
        width=800,
        plot_bgcolor='white',
        font=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, explanation


def _rev_vs_ly(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    max_year = df["Date"].dt.year.max()
    ly = max_year - 1
    this_year = df[df["Date"].dt.year == max_year]
    last_year = df[df["Date"].dt.year == ly]

    ty = this_year.groupby(this_year["Date"].dt.month)["TotalRevenue"].sum()
    ly_ = last_year.groupby(last_year["Date"].dt.month)["TotalRevenue"].sum()

    # Calculate KPIs
    if not ty.empty and not ly_.empty:
        total_ty = ty.sum()
        total_ly = ly_.sum()
        yoy_change = ((total_ty - total_ly) / total_ly) * 100
        performance_class = _classify_performance(abs(yoy_change))
        trend = "growth" if yoy_change > 0 else "decline"

        # Identify the month with biggest change
        common_months = set(ty.index).intersection(set(ly_.index))
        if common_months:
            pct_changes = {m: ((ty.get(m, 0) - ly_.get(m, 0)) / ly_.get(m, 1)) * 100 for m in common_months}
            max_change_month = max(pct_changes.items(), key=lambda x: abs(x[1]))
            month_name = pd.Timestamp(2023, max_change_month[0], 1).strftime("%B")
            month_change = max_change_month[1]
            month_trend = "increase" if month_change > 0 else "decrease"

            explanation = (
                f"Year-over-year revenue shows {performance_class} {trend} of {abs(yoy_change):.1f}%. "
                f"The most significant change was in {month_name} with a {abs(month_change):.1f}% {month_trend}. "
                f"Total revenue: ${total_ty:,.0f} vs ${total_ly:,.0f} last year."
            )
        else:
            explanation = (
                f"Year-over-year revenue shows {performance_class} {trend} of {abs(yoy_change):.1f}%. "
                f"Total revenue: ${total_ty:,.0f} vs ${total_ly:,.0f} last year."
            )
    else:
        explanation = "Not enough data to calculate year-over-year comparison."

    if ty.empty or ly_.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough years of data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title="Revenue by Month – YoY", height=500)
        return fig, explanation

    # Create Plotly grouped bar chart
    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in range(1, 13)]

    fig = go.Figure()

    # Add bars for current year
    fig.add_trace(go.Bar(
        x=[month_names[i-1] for i in ty.index],
        y=ty.values,
        name=str(max_year),
        marker_color="#3498db",
        width=0.4
    ))

    # Add bars for last year
    fig.add_trace(go.Bar(
        x=[month_names[i-1] for i in ly_.index],
        y=ly_.values,
        name=str(ly),
        marker_color="#e74c3c",
        width=0.4
    ))

    fig.update_layout(
        title="Revenue by Month – YoY",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        barmode='group',
        height=500,
        width=800,
        plot_bgcolor='white',
        font=dict(size=10),
        yaxis=dict(tickformat='$,.0f')
    )

    return fig, explanation


def _segment_season(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    _add_season_col(df)
    pivot = (
        df.groupby(["Season", "MarketSegment"])["TotalRevenue"]
        .sum()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )

    # Calculate KPIs
    # Find most prevalent segment in each season
    top_segments = {season: pivot.loc[season].idxmax() for season in pivot.index}

    # Find season with highest dependency on a single segment
    max_dependency = pivot.max(axis=1).idxmax()
    top_segment_in_max = pivot.loc[max_dependency].idxmax()
    max_dependency_pct = pivot.loc[max_dependency, top_segment_in_max] * 100

    # Check diversity score - higher coefficient of variation means less diversity
    segment_diversity = {season: (pivot.loc[season].std() / pivot.loc[season].mean()) * 100
                        for season in pivot.index}
    least_diverse = max(segment_diversity.items(), key=lambda x: x[1])[0]
    most_diverse = min(segment_diversity.items(), key=lambda x: x[1])[0]

    diversity_diff = segment_diversity[least_diverse] - segment_diversity[most_diverse]
    diversity_class = _classify_performance(diversity_diff)

    explanation = (
        f"{least_diverse} shows {diversity_class} segment concentration with "
        f"{top_segments[least_diverse]} generating {pivot.loc[least_diverse, top_segments[least_diverse]]*100:.1f}% of revenue. "
        f"{most_diverse} has the most balanced segment mix. "
        f"Overall, {max_dependency} has the highest dependence on a single segment ({top_segment_in_max}: {max_dependency_pct:.1f}%)."
    )

    # Create stacked bar chart
    fig = go.Figure()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']

    for i, segment in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[segment],
            name=segment,
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title="Segment Mix per Season (share of revenue)",
        xaxis_title="Season",
        yaxis_title="Revenue Share",
        barmode='stack',
        height=500,
        width=800,
        plot_bgcolor='white',
        font=dict(size=10),
        yaxis=dict(tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, explanation


def _booking_heatmap(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    if {"BookingDate", "CheckInDate"}.issubset(df.columns):
        df = df.copy()
        df["LeadTime"] = (df["CheckInDate"] - df["BookingDate"]).dt.days.clip(lower=0)
        bins = pd.cut(
            df["LeadTime"],
            bins=[-1, 7, 30, 60, 90, 180, 365, np.inf],
            labels=["0-7", "8-30", "31-60", "61-90", "91-180", "181-365", "365+"],
        )
        heat = df.pivot_table(
            index=bins,
            columns=df["CheckInDate"].dt.month,
            values="TotalRevenue",
            aggfunc="sum",
        ).fillna(0)

        # Calculate KPIs
        total_revenue = heat.sum().sum()

        # Revenue by lead time category
        lead_time_rev = heat.sum(axis=1)
        top_leadtime = lead_time_rev.idxmax()
        top_leadtime_pct = (lead_time_rev[top_leadtime] / total_revenue) * 100

        # Revenue by month
        month_rev = heat.sum(axis=0)
        top_month = month_rev.idxmax()
        top_month_name = pd.Timestamp(2023, top_month, 1).strftime("%B")

        # Lead time for top month
        if not heat[top_month].empty:
            top_month_leadtime = heat[top_month].idxmax()

            # Compare most common booking window across months
            booking_windows = {col: heat[col].idxmax() for col in heat.columns}
            common_window = max(booking_windows.values(), key=list(booking_windows.values()).count)
            consistent_months = sum(1 for win in booking_windows.values() if win == common_window)
            consistency_pct = (consistent_months / len(booking_windows)) * 100
            consistency_class = "high" if consistency_pct >= 75 else "moderate" if consistency_pct >= 50 else "low"

            explanation = (
                f"The most common booking window is {top_leadtime} days in advance ({top_leadtime_pct:.1f}% of revenue). "
                f"{top_month_name} shows peak bookings with lead time of {top_month_leadtime}. "
                f"Booking pattern consistency is {consistency_class} with {consistency_pct:.1f}% of months "
                f"having {common_window} as the dominant booking window."
            )
        else:
            explanation = f"The most common booking window is {top_leadtime} days in advance ({top_leadtime_pct:.1f}% of revenue)."

        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heat.values,
            x=[f"Month {i}" for i in range(1, 13)],
            y=heat.index.astype(str),
            colorscale="Viridis",
            showscale=True,
            hoverongaps=False,
            hovertemplate="Month: %{x}<br>Lead Time: %{y}<br>Revenue: $%{z:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            title="Booking Window Heatmap (Revenue)",
            xaxis_title="Month",
            yaxis_title="Lead Time (days)",
            height=500,
            width=800
        )
        return fig, explanation
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="BookingDate / CheckInDate columns missing",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=500)
        explanation = "Cannot analyze booking windows: required date columns are missing from the dataset."
        return fig, explanation


def _weekday_weekend(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    grp = (
        df.groupby([df["Date"].dt.month, df["Date"].dt.weekday < 5])["RevPAR"]
        .mean()
        .unstack()
    )
    grp.columns = ["Weekend", "Weekday"]

    # Calculate KPIs
    weekend_avg = grp["Weekend"].mean()
    weekday_avg = grp["Weekday"].mean()
    diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
    diff_class = _classify_performance(abs(diff_pct))

    # Find month with biggest weekend premium
    premium_by_month = ((grp["Weekend"] - grp["Weekday"]) / grp["Weekday"]) * 100
    top_premium_month = premium_by_month.idxmax()
    top_premium = premium_by_month.max()
    top_month_name = pd.Timestamp(2023, top_premium_month, 1).strftime("%B")

    trend = "premium" if weekend_avg > weekday_avg else "discount"

    explanation = (
        f"Weekend RevPAR shows a {diff_class} {trend} of {abs(diff_pct):.1f}% "
        f"compared to weekdays (${weekend_avg:.2f} vs ${weekday_avg:.2f}). "
        f"{top_month_name} has the highest weekend {trend} at {abs(top_premium):.1f}%."
    )

    # Create Plotly bar chart
    fig = go.Figure()

    # Add Weekend bars
    fig.add_trace(go.Bar(
        name="Weekend",
        x=grp.index,
        y=grp["Weekend"],
        marker_color="#ff6b6b",
        text=[f"${val:.0f}" for val in grp["Weekend"]],
        textposition="outside"
    ))

    # Add Weekday bars
    fig.add_trace(go.Bar(
        name="Weekday",
        x=grp.index,
        y=grp["Weekday"],
        marker_color="#4ecdc4",
        text=[f"${val:.0f}" for val in grp["Weekday"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="Weekday vs Weekend RevPAR",
        xaxis_title="Month",
        yaxis_title="RevPAR ($)",
        barmode="group",
        height=500,
        width=800,
        yaxis=dict(tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, explanation


def _seasonality_index(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    month_rev = df.groupby(df["Date"].dt.month)["TotalRevenue"].sum()
    idx = month_rev / month_rev.mean()

    # Calculate KPIs
    peak_month = idx.idxmax()
    peak_month_name = pd.Timestamp(2023, peak_month, 1).strftime("%B")
    low_month = idx.idxmin()
    low_month_name = pd.Timestamp(2023, low_month, 1).strftime("%B")

    peak_val = idx.max()
    low_val = idx.min()
    range_pct = ((peak_val - low_val) / low_val) * 100
    seasonality_class = _classify_performance(range_pct)

    # Count months above and below average
    above_avg = sum(1 for val in idx if val > 1)
    below_avg = sum(1 for val in idx if val < 1)

    explanation = (
        f"Revenue shows {seasonality_class} seasonality with {range_pct:.1f}% difference "
        f"between peak ({peak_month_name}: {peak_val:.2f}x average) and "
        f"low season ({low_month_name}: {low_val:.2f}x average). "
        f"{above_avg} months perform above average, {below_avg} below average."
    )

    # Create Plotly bar chart
    fig = go.Figure()

    # Add seasonality index bars
    fig.add_trace(go.Bar(
        x=idx.index,
        y=idx.values,
        marker_color="#4ade80",
        text=[f"{val:.2f}x" for val in idx.values],
        textposition="outside",
        name="Seasonality Index"
    ))

    # Add average line at 1.0
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#facc15",
        line_width=2,
        annotation_text="Average (1.0x)"
    )

    fig.update_layout(
        title="Seasonality Index (1.0 = Avg Month Revenue)",
        xaxis_title="Month",
        yaxis_title="Revenue Multiple",
        height=500,
        width=800,
        showlegend=False
    )

    return fig, explanation


def _season_summary(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    _add_season_col(df)
    summary = df.groupby("Season")[["TotalRevenue", "RevPAR", "OccupancyRate"]].mean()
    summary["OccupancyRate"] *= 100  # for display %

    # Calculate KPIs
    top_season = summary["TotalRevenue"].idxmax()
    bottom_season = summary["TotalRevenue"].idxmin()
    top_rev = summary.loc[top_season, "TotalRevenue"]
    bottom_rev = summary.loc[bottom_season, "TotalRevenue"]

    diff_pct = ((top_rev - bottom_rev) / bottom_rev) * 100
    seasonality_class = _classify_performance(diff_pct)

    top_occ = summary.loc[top_season, "OccupancyRate"]
    top_revpar = summary.loc[top_season, "RevPAR"]

    explanation = (
        f"{top_season} is the peak season with {diff_pct:.1f}% higher revenue than {bottom_season}, "
        f"indicating {seasonality_class} seasonal variation. "
        f"Peak season occupancy is {top_occ:.1f}% with RevPAR of ${top_revpar:.2f}."
    )

    # Create Plotly subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add revenue and RevPAR bars
    fig.add_trace(go.Bar(
        name="Total Revenue",
        x=summary.index,
        y=summary["TotalRevenue"],
        marker_color="#3b82f6",
        yaxis="y"
    ))

    fig.add_trace(go.Bar(
        name="RevPAR",
        x=summary.index,
        y=summary["RevPAR"],
        marker_color="#8b5cf6",
        yaxis="y"
    ))

    # Add occupancy line on secondary y-axis
    fig.add_trace(go.Scatter(
        name="Occupancy %",
        x=summary.index,
        y=summary["OccupancyRate"],
        mode="lines+markers",
        line=dict(color="#14b8a6", width=3),
        marker=dict(size=8),
        yaxis="y2"
    ), secondary_y=True)

    # Update layout
    fig.update_xaxes(title_text="Season")
    fig.update_yaxes(title_text="Revenue / RevPAR ($)", secondary_y=False, tickformat="$,.0f")
    fig.update_yaxes(title_text="Occupancy (%)", secondary_y=True, tickformat=".1f")

    fig.update_layout(
        title="Season Summary",
        height=500,
        width=800,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, explanation


# ──────────────────────────────────────────────────────────────
# main display widget
# ──────────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:
    base_df: pd.DataFrame = get_df()  # full dataset (already typed as datetime)

    # Define _collapsible function here to avoid circular imports
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

    root = QWidget()
    root.setLayout(QVBoxLayout())

    # Header label
    header = QLabel("Seasonality Analysis")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    root.layout().addWidget(header)

    # Date-filter row
    filter_row = QHBoxLayout()
    root.layout().addLayout(filter_row)

    min_date = base_df["Date"].min().date()
    max_date = base_df["Date"].max().date()

    start_picker = QDateEdit(QDate(min_date.year, min_date.month, min_date.day))
    end_picker = QDateEdit(QDate(max_date.year, max_date.month, max_date.day))
    for p in (start_picker, end_picker):
        p.setCalendarPopup(True)

    filter_row.addWidget(QLabel("Date Range:"))
    filter_row.addWidget(start_picker)
    filter_row.addWidget(QLabel(" to "))
    filter_row.addWidget(end_picker)
    apply_btn = QPushButton("Apply")
    filter_row.addWidget(apply_btn)
    filter_row.addStretch()

    # Tabs for charts
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    # ------------- update logic -------------
    def _render():
        start = pd.Timestamp(start_picker.date().toPython()) # type: ignore
        end = pd.Timestamp(end_picker.date().toPython()) # type: ignore

        # Use the original DataFrame and filter it fresh each time
        df = base_df.copy()
        df = _mask_by_date(df, start, end)

        tabs.clear()  # wipe old charts

        chart_map: list[Tuple[str, Callable[[pd.DataFrame], Tuple[go.Figure, str]], set[str]]] = [
            # For Occupancy, check for multiple possible column names
            ("Occupancy", _monthly_occ, {"Date", *{"OccupancyRate", "occupancy", "Occupancy"} & set(df.columns)}),
            ("ADR & RevPAR", _adr_revpar, {"ADR", "RevPAR", "OccupancyRate", "Date"}),
            ("Revenue YoY", _rev_vs_ly, {"TotalRevenue", "Date"}),
            ("Segment Mix", _segment_season, {"MarketSegment", "TotalRevenue", "Date"}),
            (
                "Booking Window",
                _booking_heatmap,
                {"BookingDate", "CheckInDate", "TotalRevenue"},
            ),
            ("Weekday/Weekend", _weekday_weekend, {"RevPAR", "Date"}),
            ("Index", _seasonality_index, {"TotalRevenue", "Date"}),
            (
                "Season Summary",
                _season_summary,
                {"TotalRevenue", "RevPAR", "OccupancyRate", "Date"},
            ),
        ]

        for title, builder, req in chart_map:
            if req.issubset(df.columns) and not df.empty:
                fig, explanation = builder(df)
                tab = QWidget()
                tab_layout = QVBoxLayout(tab)
                tab_layout.setContentsMargins(5, 5, 5, 5)
                tab_layout.addWidget(create_plotly_widget(fig))
                tab_layout.addWidget(_collapsible(explanation))
                tabs.addTab(tab, title)

        if tabs.count() == 0:
            lbl = QLabel("No seasonality-related fields in this dataset / date range.")
            lbl.setStyleSheet("font-size: 14pt; color: #f97316;")
            tabs.addTab(QWidget(), "Info")  # placeholder
            tabs.setTabEnabled(0, False)
            root.layout().addWidget(lbl)

    # connect button
    apply_btn.clicked.connect(_render)

    # initial render
    _render()

    return root
