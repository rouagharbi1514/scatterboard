# views/seasonality.py
"""
Seasonality Analysis view
-------------------------
* Date-range picker (start / end) + Apply button
* Eight charts that matter (see spec below)
* Graceful degradation if columns are absent
(Design modernisé : thème clair pastel, cartes, onglets arrondis)
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDateEdit,
    QPushButton, QTabWidget, QFrame
)

from data.helpers import get_df  # global cached DataFrame loader
from views.utils import data_required, create_plotly_widget  # decorator you already use


# ──────────────────────────────────────────────────────────────
# Palette douce + helpers Plotly
# ──────────────────────────────────────────────────────────────
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

def _apply_soft_layout(fig: go.Figure, title: str, height: int = 500, width: Optional[int] = None):
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


# ──────────────────────────────────────────────────────────────
# UI: panneau repliable (global, utilisé partout)
# ──────────────────────────────────────────────────────────────
def _collapsible(text: str) -> QWidget:
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


# ──────────────────────────────────────────────────────────────
# helpers data
# ──────────────────────────────────────────────────────────────
def _mask_by_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    date_col = "Date" if "Date" in df.columns else "date"
    if date_col in df.columns:
        return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
    print(f"Warning: No '{date_col}' column found in the DataFrame")
    return pd.DataFrame(columns=df.columns)


def _add_season_col(df: pd.DataFrame) -> None:
    if "Season" not in df.columns:
        df["Season"] = pd.cut(
            df["Date"].dt.month,
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
            right=False,
        )


def _classify_performance(value: float) -> str:
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
    """Ajoute l'onglet si colonnes req. présentes."""
    if cols.issubset(df.columns) and not df.empty:
        fig, explanation = builder(df)

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
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
        tabs.addTab(tab, title)
        return tab
    return None


# ──────────────────────────────────────────────────────────────
# charts
# ──────────────────────────────────────────────────────────────
def _monthly_occ(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    occ_col = None
    if "OccupancyRate" in df.columns: occ_col = "OccupancyRate"
    elif "occupancy" in df.columns:  occ_col = "occupancy"
    elif "Occupancy" in df.columns:  occ_col = "Occupancy"

    if occ_col is None:
        fig = go.Figure()
        fig.add_annotation(text="Occupancy data not found",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=FONT_COLOR))
        _apply_soft_layout(fig, "Monthly Occupancy")
        return fig, "Occupancy data not available."

    df_copy = df.copy()
    if df_copy[occ_col].max() > 1:
        df_copy[occ_col] = df_copy[occ_col] / 100

    grp = df_copy.groupby(df_copy["Date"].dt.month)[occ_col]
    m   = grp.mean()
    std = grp.std()

    avg_occ = m.mean() * 100
    std_ratio = (std.mean() / m.mean()) * 100 if m.mean() > 0 else 0
    seasonality_class = _classify_performance(std_ratio)
    high_month = m.idxmax(); low_month = m.idxmin()
    high_month_name = pd.Timestamp(2023, high_month, 1).strftime("%B")
    low_month_name  = pd.Timestamp(2023, low_month, 1).strftime("%B")
    month_diff_pct = ((m.max() - m.min()) / m.min() * 100) if m.min() > 0 else 0

    explanation = (
        f"Average occupancy is {avg_occ:.1f}%, with {seasonality_class} demand seasonality "
        f"(±{std_ratio:.1f}% variability). "
        f"{high_month_name} has the highest at {m.max()*100:.1f}%, "
        f"{low_month_name} the lowest at {m.min()*100:.1f}% "
        f"({month_diff_pct:.1f}% difference)."
    )

    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in m.index]
    fig = go.Figure()

    upper = (m + std).values * 100
    lower = (m - std).values * 100
    fig.add_trace(go.Scatter(
        x=month_names + month_names[::-1],
        y=list(upper) + list(lower[::-1]),
        fill='toself', fillcolor='rgba(100,181,246,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name="±1σ Range"
    ))
    fig.add_trace(go.Scatter(
        x=month_names, y=m.values * 100,
        mode='lines+markers', name="Avg Occupancy",
        line=dict(color=SOFT_BLUE, width=3),
        marker=dict(size=7, line=dict(color="white", width=1)),
        error_y=dict(type='data', array=std.values * 100, visible=True, color='rgba(100,181,246,0.35)')
    ))

    _apply_soft_layout(fig, "Monthly Occupancy Trend")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Occupancy (%)")
    return fig, explanation


def _adr_revpar(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    m = df.groupby(df["Date"].dt.month)[["ADR", "RevPAR"]].mean()

    avg_adr = m["ADR"].mean()
    avg_revpar = m["RevPAR"].mean()
    high_season_months = m["RevPAR"].nlargest(3).index
    low_season_months  = m["RevPAR"].nsmallest(3).index

    high_revpar = m.loc[high_season_months, "RevPAR"].mean()
    low_revpar  = m.loc[low_season_months, "RevPAR"].mean()
    revpar_uplift = ((high_revpar - low_revpar) / max(low_revpar, 1e-9)) * 100
    seasonality_class = _classify_performance(revpar_uplift)

    high_months_text = ", ".join([pd.Timestamp(2023, m_, 1).strftime("%b") for m_ in high_season_months])
    low_months_text  = ", ".join([pd.Timestamp(2023, m_, 1).strftime("%b") for m_ in low_season_months])

    explanation = (
        f"RevPAR shows {seasonality_class} seasonal variation with {revpar_uplift:.1f}% higher values "
        f"in peak months ({high_months_text}) vs low season ({low_months_text}). "
        f"Average ADR ${avg_adr:.2f}, RevPAR ${avg_revpar:.2f}."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in m.index]

    fig.add_trace(go.Scatter(
        x=month_names, y=m["ADR"], mode='lines+markers', name="ADR ($)",
        line=dict(color=SOFT_RED, width=3),
        marker=dict(size=7, symbol="circle", line=dict(color="white", width=1))
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=month_names, y=m["RevPAR"], mode='lines+markers', name="RevPAR ($)",
        line=dict(color=SOFT_BLUE, width=3),
        marker=dict(size=7, symbol="square", line=dict(color="white", width=1))
    ), secondary_y=False)

    if "OccupancyRate" in df.columns:
        occ = df.groupby(df["Date"].dt.month)["OccupancyRate"].mean() * 100
        fig.add_trace(go.Scatter(
            x=month_names, y=occ, mode='lines+markers', name="Occupancy (%)",
            line=dict(color=SOFT_TEAL, width=2, dash="dash"),
            marker=dict(size=6, symbol="triangle-up", line=dict(color="white", width=1))
        ), secondary_y=True)

    _apply_soft_layout(fig, "ADR & RevPAR Seasonality")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="ADR / RevPAR ($)", secondary_y=False)
    fig.update_yaxes(title_text="Occupancy (%)",   secondary_y=True)
    return fig, explanation


def _rev_vs_ly(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    max_year = df["Date"].dt.year.max()
    ly = max_year - 1
    this_year = df[df["Date"].dt.year == max_year]
    last_year = df[df["Date"].dt.year == ly]

    ty = this_year.groupby(this_year["Date"].dt.month)["TotalRevenue"].sum()
    ly_ = last_year.groupby(last_year["Date"].dt.month)["TotalRevenue"].sum()

    if not ty.empty and not ly_.empty:
        total_ty, total_ly = ty.sum(), ly_.sum()
        yoy_change = ((total_ty - total_ly) / max(total_ly, 1e-9)) * 100
        performance_class = _classify_performance(abs(yoy_change))
        trend = "growth" if yoy_change > 0 else "decline"

        common = set(ty.index).intersection(ly_.index)
        if common:
            pct_changes = {m: ((ty.get(m, 0) - ly_.get(m, 0)) / max(ly_.get(m, 1), 1e-9)) * 100 for m in common}
            mo, chg = max(pct_changes.items(), key=lambda x: abs(x[1]))
            month_name = pd.Timestamp(2023, mo, 1).strftime("%B")
            month_trend = "increase" if chg > 0 else "decrease"
            explanation = (
                f"YoY revenue shows {performance_class} {trend} of {abs(yoy_change):.1f}%. "
                f"Largest change: {month_name} ({abs(chg):.1f}% {month_trend}). "
                f"Total: ${total_ty:,.0f} vs ${total_ly:,.0f}."
            )
        else:
            explanation = (
                f"YoY revenue shows {performance_class} {trend} of {abs(yoy_change):.1f}%. "
                f"Total: ${total_ty:,.0f} vs ${total_ly:,.0f}."
            )
    else:
        explanation = "Not enough data to calculate year-over-year comparison."

    if ty.empty or ly_.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough years of data",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color=FONT_COLOR))
        _apply_soft_layout(fig, "Revenue by Month – YoY")
        return fig, explanation

    month_names = [pd.Timestamp(2023, i, 1).strftime("%b") for i in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[month_names[i-1] for i in ty.index], y=ty.values,
        name=str(max_year), marker_color=SOFT_BLUE, width=0.4
    ))
    fig.add_trace(go.Bar(
        x=[month_names[i-1] for i in ly_.index], y=ly_.values,
        name=str(ly), marker_color=SOFT_RED, width=0.4
    ))
    _apply_soft_layout(fig, "Revenue by Month – YoY")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Revenue ($)", tickformat='$,.0f')
    fig.update_layout(barmode='group')
    return fig, explanation


def _segment_season(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    _add_season_col(df)
    pivot = (
        df.groupby(["Season", "MarketSegment"])["TotalRevenue"]
        .sum().unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )

    top_segments = {s: pivot.loc[s].idxmax() for s in pivot.index}
    max_dependency = pivot.max(axis=1).idxmax()
    top_segment_in_max = pivot.loc[max_dependency].idxmax()
    max_dependency_pct = pivot.loc[max_dependency, top_segment_in_max] * 100
    segment_diversity = {s: (pivot.loc[s].std() / pivot.loc[s].mean()) * 100 for s in pivot.index}
    least_diverse = max(segment_diversity.items(), key=lambda x: x[1])[0]
    most_diverse  = min(segment_diversity.items(), key=lambda x: x[1])[0]
    diversity_diff = segment_diversity[least_diverse] - segment_diversity[most_diverse]
    diversity_class = _classify_performance(diversity_diff)

    explanation = (
        f"{least_diverse} shows {diversity_class} segment concentration with "
        f"{top_segments[least_diverse]} at {pivot.loc[least_diverse, top_segments[least_diverse]]*100:.1f}% of revenue. "
        f"{most_diverse} has the most balanced mix. "
        f"Highest single-segment dependence in {max_dependency} ({top_segment_in_max}: {max_dependency_pct:.1f}%)."
    )

    fig = go.Figure()
    palette = [SOFT_BLUE, SOFT_RED, SOFT_TEAL, "#f6d28f", SOFT_LILAC, SOFT_MINT, "#b0bec5", "#ffd3b6"]
    for i, seg in enumerate(pivot.columns):
        fig.add_trace(go.Bar(x=pivot.index, y=pivot[seg], name=seg, marker_color=palette[i % len(palette)]))

    _apply_soft_layout(fig, "Segment Mix per Season (share of revenue)")
    fig.update_xaxes(title="Season")
    fig.update_yaxes(title="Revenue Share", tickformat='.0%')
    fig.update_layout(barmode='stack')
    return fig, explanation


def _booking_heatmap(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    if {"BookingDate", "CheckInDate"}.issubset(df.columns):
        df = df.copy()
        df["LeadTime"] = (df["CheckInDate"] - df["BookingDate"]).dt.days.clip(lower=0)
        bins = pd.cut(df["LeadTime"], bins=[-1, 7, 30, 60, 90, 180, 365, np.inf],
                      labels=["0-7", "8-30", "31-60", "61-90", "91-180", "181-365", "365+"])
        heat = df.pivot_table(index=bins, columns=df["CheckInDate"].dt.month,
                              values="TotalRevenue", aggfunc="sum").fillna(0)

        total_revenue = heat.sum().sum()
        lead_time_rev = heat.sum(axis=1)
        top_leadtime = lead_time_rev.idxmax()
        top_leadtime_pct = (lead_time_rev[top_leadtime] / max(total_revenue, 1e-9)) * 100
        month_rev = heat.sum(axis=0)
        top_month = month_rev.idxmax()
        top_month_name = pd.Timestamp(2023, top_month, 1).strftime("%B")

        if not heat[top_month].empty:
            top_month_leadtime = heat[top_month].idxmax()
            booking_windows = {col: heat[col].idxmax() for col in heat.columns}
            common_window = max(booking_windows.values(), key=list(booking_windows.values()).count)
            consistent_months = sum(1 for win in booking_windows.values() if win == common_window)
            consistency_pct = (consistent_months / len(booking_windows)) * 100
            consistency_class = "high" if consistency_pct >= 75 else "moderate" if consistency_pct >= 50 else "low"
            explanation = (
                f"The most common booking window is {top_leadtime} days ({top_leadtime_pct:.1f}% of revenue). "
                f"{top_month_name} peaks with window {top_month_leadtime}. "
                f"Pattern consistency is {consistency_class} ({consistency_pct:.1f}% of months share {common_window})."
            )
        else:
            explanation = f"The most common booking window is {top_leadtime} days ({top_leadtime_pct:.1f}% of revenue)."

        fig = go.Figure(data=go.Heatmap(
            z=heat.values,
            x=[f"Month {i}" for i in heat.columns],
            y=heat.index.astype(str),
            colorscale="YlGnBu",
            showscale=True,
            hoverongaps=False,
            colorbar=dict(len=0.8, thickness=14, outlinewidth=0, title="Revenue"),
            hovertemplate="Month: %{x}<br>Lead Time: %{y}<br>Revenue: $%{z:,.0f}<extra></extra>",
        ))
        _apply_soft_layout(fig, "Booking Window Heatmap (Revenue)")
        fig.update_xaxes(title="Month")
        fig.update_yaxes(title="Lead Time (days)")
        return fig, explanation

    fig = go.Figure()
    fig.add_annotation(text="BookingDate / CheckInDate columns missing",
                       x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=FONT_COLOR))
    _apply_soft_layout(fig, "Booking Window Heatmap (Revenue)")
    explanation = "Cannot analyze booking windows: required date columns are missing."
    return fig, explanation


def _weekday_weekend(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    grp = (df.groupby([df["Date"].dt.month, df["Date"].dt.weekday < 5])["RevPAR"].mean().unstack())
    grp.columns = ["Weekend", "Weekday"]

    weekend_avg = grp["Weekend"].mean()
    weekday_avg = grp["Weekday"].mean()
    diff_pct = ((weekend_avg - weekday_avg) / max(weekday_avg, 1e-9)) * 100
    diff_class = _classify_performance(abs(diff_pct))

    premium_by_month = ((grp["Weekend"] - grp["Weekday"]) / grp["Weekday"]).replace([np.inf, -np.inf], np.nan) * 100
    top_premium_month = premium_by_month.idxmax()
    top_premium = np.nanmax(premium_by_month.values)
    top_month_name = pd.Timestamp(2023, top_premium_month, 1).strftime("%B") if pd.notna(top_premium_month) else "—"
    trend = "premium" if weekend_avg > weekday_avg else "discount"

    explanation = (
        f"Weekend RevPAR shows a {diff_class} {trend} of {abs(diff_pct):.1f}% "
        f"vs weekdays (${weekend_avg:.2f} vs ${weekday_avg:.2f}). "
        f"{top_month_name} has the highest weekend {trend} at {abs(top_premium):.1f}%."
        if pd.notna(top_premium) else
        f"Weekend vs Weekday comparison indicates a {diff_class} {trend}."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Weekend", x=grp.index, y=grp["Weekend"],
        marker_color=SOFT_CORAL, text=[f"${val:.0f}" for val in grp["Weekend"]],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Weekday", x=grp.index, y=grp["Weekday"],
        marker_color=SOFT_CYAN, text=[f"${val:.0f}" for val in grp["Weekday"]],
        textposition="outside"
    ))
    _apply_soft_layout(fig, "Weekday vs Weekend RevPAR")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="RevPAR ($)", tickformat="$,.0f")
    fig.update_layout(barmode="group")
    return fig, explanation


def _seasonality_index(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    month_rev = df.groupby(df["Date"].dt.month)["TotalRevenue"].sum()
    idx = month_rev / month_rev.mean()

    peak_month = idx.idxmax(); low_month = idx.idxmin()
    peak_month_name = pd.Timestamp(2023, peak_month, 1).strftime("%B")
    low_month_name  = pd.Timestamp(2023, low_month, 1).strftime("%B")
    peak_val, low_val = idx.max(), idx.min()
    range_pct = ((peak_val - low_val) / max(low_val, 1e-9)) * 100
    seasonality_class = _classify_performance(range_pct)
    above_avg = sum(1 for v in idx if v > 1)
    below_avg = sum(1 for v in idx if v < 1)

    explanation = (
        f"Revenue shows {seasonality_class} seasonality with {range_pct:.1f}% difference "
        f"between peak ({peak_month_name}: {peak_val:.2f}x avg) and "
        f"low season ({low_month_name}: {low_val:.2f}x avg). "
        f"{above_avg} months above average, {below_avg} below."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=idx.index, y=idx.values, marker_color=SOFT_MINT,
        text=[f"{v:.2f}x" for v in idx.values], textposition="outside",
        name="Seasonality Index"
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=SOFT_AMBER, line_width=2,
                  annotation_text="Average (1.0x)", annotation_position="top left")
    _apply_soft_layout(fig, "Seasonality Index (1.0 = Avg Month Revenue)")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Revenue Multiple")
    return fig, explanation


def _season_summary(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    _add_season_col(df)
    summary = df.groupby("Season")[["TotalRevenue", "RevPAR", "OccupancyRate"]].mean()
    summary["OccupancyRate"] *= 100

    top_season = summary["TotalRevenue"].idxmax()
    bottom_season = summary["TotalRevenue"].idxmin()
    top_rev = summary.loc[top_season, "TotalRevenue"]
    bottom_rev = summary.loc[bottom_season, "TotalRevenue"]
    diff_pct = ((top_rev - bottom_rev) / max(bottom_rev, 1e-9)) * 100
    seasonality_class = _classify_performance(diff_pct)
    top_occ = summary.loc[top_season, "OccupancyRate"]
    top_revpar = summary.loc[top_season, "RevPAR"]

    explanation = (
        f"{top_season} is the peak season with {diff_pct:.1f}% higher revenue than {bottom_season} "
        f"({seasonality_class} variation). Peak occupancy {top_occ:.1f}% | RevPAR ${top_revpar:.2f}."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Total Revenue", x=summary.index, y=summary["TotalRevenue"], marker_color="#93c5fd"))
    fig.add_trace(go.Bar(name="RevPAR",         x=summary.index, y=summary["RevPAR"],        marker_color=SOFT_LILAC))
    fig.add_trace(go.Scatter(
        name="Occupancy %", x=summary.index, y=summary["OccupancyRate"], mode="lines+markers",
        line=dict(color="#99f6e4", width=3),
        marker=dict(size=7, line=dict(color="white", width=1))
    ), secondary_y=True)

    _apply_soft_layout(fig, "Season Summary")
    fig.update_xaxes(title_text="Season")
    fig.update_yaxes(title_text="Revenue / RevPAR ($)", secondary_y=False, tickformat="$,.0f")
    fig.update_yaxes(title_text="Occupancy (%)", secondary_y=True, tickformat=".1f")
    fig.update_layout(barmode="group")
    return fig, explanation


# ──────────────────────────────────────────────────────────────
# Main display
# ──────────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:
    base_df: pd.DataFrame = get_df()

    root = QWidget()
    root.setObjectName("SeasonalityRoot")
    root.setLayout(QVBoxLayout())
    root.layout().setContentsMargins(12, 12, 12, 12)
    root.layout().setSpacing(10)

    # QSS global (pastel)
    root.setStyleSheet("""
        QWidget#SeasonalityRoot {
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
    header = QLabel("Seasonality Analysis")
    header.setObjectName("header")
    root.layout().addWidget(header)

    # Toolbar (date range) dans une carte
    toolbar = QFrame()
    toolbar.setObjectName("toolbar")
    tl = QHBoxLayout(toolbar)
    tl.setContentsMargins(12, 10, 12, 10)
    tl.setSpacing(8)

    min_date = base_df["Date"].min().date()
    max_date = base_df["Date"].max().date()

    start_picker = QDateEdit(QDate(min_date.year, min_date.month, min_date.day))
    end_picker   = QDateEdit(QDate(max_date.year, max_date.month, max_date.day))
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

    # Rendu
    def _render():
        start = pd.Timestamp(start_picker.date().toPython())  # type: ignore
        end   = pd.Timestamp(end_picker.date().toPython())    # type: ignore
        df = _mask_by_date(base_df.copy(), start, end)

        tabs.clear()
        chart_map: list[Tuple[str, Callable[[pd.DataFrame], Tuple[go.Figure, str]], set[str]]] = [
            ("Occupancy",        _monthly_occ,       {"Date", *{"OccupancyRate", "occupancy", "Occupancy"} & set(df.columns)}),
            ("ADR & RevPAR",     _adr_revpar,        {"ADR", "RevPAR", "OccupancyRate", "Date"}),
            ("Revenue YoY",      _rev_vs_ly,         {"TotalRevenue", "Date"}),
            ("Segment Mix",      _segment_season,    {"MarketSegment", "TotalRevenue", "Date"}),
            ("Booking Window",   _booking_heatmap,   {"BookingDate", "CheckInDate", "TotalRevenue"}),
            ("Weekday/Weekend",  _weekday_weekend,   {"RevPAR", "Date"}),
            ("Index",            _seasonality_index, {"TotalRevenue", "Date"}),
            ("Season Summary",   _season_summary,    {"TotalRevenue", "RevPAR", "OccupancyRate", "Date"}),
        ]

        any_added = False
        for title, builder, req in chart_map:
            if req.issubset(df.columns) and not df.empty:
                _required(tabs, title, builder, df, req)
                any_added = True

        if not any_added:
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
            lbl = QLabel("No seasonality-related fields in this dataset / date range.")
            lbl.setStyleSheet("font-size: 14px; color: #8b5e34;")
            v.addWidget(lbl)
            empty_tab = QWidget()
            l2 = QVBoxLayout(empty_tab)
            l2.addWidget(info_card)
            tabs.addTab(empty_tab, "Info")
            tabs.setTabEnabled(0, True)

    apply_btn.clicked.connect(_render)
    _render()
    return root

