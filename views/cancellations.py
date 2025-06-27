# views/cancellations.py
"""
Cancellations & No-Show Analysis
--------------------------------
* Date-range picker auto-syncs to the DataFrame’s min/max dates.
* KPI tiles (cancel % etc.) recalc each filter.
* Nine insightful charts, each shown only if required columns exist.
* Exposes `display.refresh_date_pickers` for external calls after data upload.
"""

from __future__ import annotations
from views.utils import data_required, kpi_tile
from data.helpers import get_df
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDateEdit,
    QPushButton,
    QTabWidget,
    QSizePolicy,
    QGridLayout,
)
from PySide6.QtCore import QDate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")


def _canvas(fig: Figure) -> Canvas:
    """Wrap a matplotlib Figure in a Qt Canvas."""
    c = Canvas(fig)
    c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return c


def _filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows where Date is between start and end (inclusive)."""
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def _maybe(tabs: QTabWidget, title: str, builder, df: pd.DataFrame, req: set[str]):
    """Add a tab only if required columns exist and df is non-empty."""
    if req.issubset(df.columns) and not df.empty:
        tabs.addTab(_canvas(builder(df)), title)


# ──────────────────────────────────────────────────────────────────────────
# Chart 1: Monthly Cancellation Trend
# ──────────────────────────────────────────────────────────────────────────
def _chart_cancellation_trend(df: pd.DataFrame) -> Figure:
    m = (
        df[df["ReservationStatus"] == "Canceled"]
        .groupby(df["Date"].dt.to_period("M"))["ReservationID"]
        .count()
    )
    m.index = m.index.to_timestamp()
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    ax.plot(m.index, m.values, marker="o", color="#e74c3c")
    ax.set_title("Monthly Cancellation Count")
    ax.set_ylabel("Cancellations")
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 2: Monthly No-Show Trend
# ──────────────────────────────────────────────────────────────────────────
def _chart_noshow_trend(df: pd.DataFrame) -> Figure:
    m = (
        df[df["ReservationStatus"] == "No-Show"]
        .groupby(df["Date"].dt.to_period("M"))["ReservationID"]
        .count()
    )
    m.index = m.index.to_timestamp()
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    ax.plot(m.index, m.values, marker="s", color="#f39c12")
    ax.set_title("Monthly No-Show Count")
    ax.set_ylabel("No-Shows")
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 3: Cancellation Rate vs Booking Volume
# ──────────────────────────────────────────────────────────────────────────
def _chart_cancel_vs_volume(df: pd.DataFrame) -> Figure:
    monthly_total = df.groupby(df["Date"].dt.to_period("M"))["ReservationID"].count()
    monthly_cxl = (
        df[df["ReservationStatus"] == "Canceled"]
        .groupby(df["Date"].dt.to_period("M"))["ReservationID"]
        .count()
    )
    cancel_pct = (monthly_cxl / monthly_total).fillna(0)

    total_ts = monthly_total.index.to_timestamp()
    pct_ts = cancel_pct.index.to_timestamp()

    fig = Figure(figsize=(6, 3.5))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    ax1.bar(
        total_ts,
        monthly_total.values,
        width=20,
        label="Total Bookings",
        color="#3498db",
        alpha=0.6,
    )
    ax2.plot(
        pct_ts,
        cancel_pct.values,
        marker="o",
        color="#e74c3c",
        label="Cancel %",
    )

    ax1.set_title("Bookings vs Cancellation Rate")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Bookings")
    ax2.set_ylabel("Cancel %")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.xaxis.set_tick_params(rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 4: Lead-Time Distribution for Cancellations
# ──────────────────────────────────────────────────────────────────────────
def _chart_lead_time_hist(df: pd.DataFrame) -> Figure:
    df_cxl = df[df["ReservationStatus"] == "Canceled"].copy()
    df_cxl["LeadTime"] = (
        df_cxl["CancellationDate"] - df_cxl["BookingDate"]
    ).dt.days.clip(lower=0)
    data = df_cxl["LeadTime"].dropna()
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    ax.hist(data, bins=20, color="#9b59b6", alpha=0.7)
    ax.set_title("Lead Time Distribution for Cancellations")
    ax.set_xlabel("Days Before Arrival")
    ax.set_ylabel("Number of Cancellations")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 5: Cancellations by Booking Source
# ──────────────────────────────────────────────────────────────────────────
def _chart_cancel_by_source(df: pd.DataFrame) -> Figure:
    pivot = (
        df[df["ReservationStatus"] == "Canceled"]
        .groupby([df["Date"].dt.to_period("M"), "BookingSource"])["ReservationID"]
        .count()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.to_timestamp()
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_title("Cancellations by Booking Source")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cancellations")
    ax.legend(fontsize=6, ncol=3)
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 6: Segment-wise Cancellation Rate
# ──────────────────────────────────────────────────────────────────────────
def _chart_segment_cancel_rate(df: pd.DataFrame) -> Figure:
    tot = df.groupby("MarketSegment")["ReservationID"].count()
    cxl = (
        df[df["ReservationStatus"] == "Canceled"]
        .groupby("MarketSegment")["ReservationID"]
        .count()
    )
    cancel_pct = (cxl / tot).fillna(0)
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot()
    cancel_pct.sort_values().plot(kind="barh", ax=ax, color="#e74c3c")
    ax.set_title("Cancellation Rate by Market Segment")
    ax.set_xlabel("Cancel %")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 7: Revenue Loss Heatmap (Weekday vs Month)
# ──────────────────────────────────────────────────────────────────────────
def _chart_revenue_loss_heatmap(df: pd.DataFrame) -> Figure:
    df_cxl = df[df["ReservationStatus"] == "Canceled"].copy()
    if "ADR" in df_cxl.columns and "NightsStayed" in df_cxl.columns:
        df_cxl["LostRevenue"] = df_cxl["ADR"] * df_cxl["NightsStayed"]
    else:
        df_cxl["LostRevenue"] = 0

    pivot = df_cxl.pivot_table(
        index=df_cxl["CancellationDate"].dt.weekday,
        columns=df_cxl["CancellationDate"].dt.month,
        values="LostRevenue",
        aggfunc="sum",
    ).fillna(0)

    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot()
    im = ax.imshow(pivot, aspect="auto", cmap="viridis")
    ax.set_title("Revenue Loss Heatmap (Weekday vs Month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Weekday (Mon=0)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 8: Refund vs Chargeback Analysis
# ──────────────────────────────────────────────────────────────────────────
def _chart_refund_chargeback(df: pd.DataFrame) -> Figure:
    pivot = (
        df[df["ReservationStatus"].isin(["Refunded", "Chargeback"])]
        .groupby([df["Date"].dt.to_period("M"), "ReservationStatus"])["ReservationID"]
        .count()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.to_timestamp()
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Refund vs Chargeback Count by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, ncol=2)
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Chart 9: Top Cancellation Reasons
# ──────────────────────────────────────────────────────────────────────────
def _chart_reason_bar(df: pd.DataFrame) -> Figure:
    reasons = (
        df[df["ReservationStatus"] == "Canceled"]["CancellationReason"]
        .value_counts()
        .head(10)
    )
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot()
    reasons.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Top Cancellation Reasons")
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Main display for cancellations
# ──────────────────────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:
    base_df = get_df()

    root = QWidget()
    root.setLayout(QVBoxLayout())

    header = QLabel("Cancellations & No-Show Analysis")
    header.setStyleSheet("font-size:18pt;font-weight:bold;")
    root.layout().addWidget(header)

    # Date pickers
    start_picker = QDateEdit()
    end_picker = QDateEdit()
    for p in (start_picker, end_picker):
        p.setCalendarPopup(True)

    def refresh_date_pickers():
        if "Date" not in base_df.columns:
            return
        d0 = base_df["Date"].min().date()
        d1 = base_df["Date"].max().date()
        start_picker.setDate(QDate(d0.year, d0.month, d0.day))
        end_picker.setDate(QDate(d1.year, d1.month, d1.day))

    refresh_date_pickers()

    filter_row = QHBoxLayout()
    filter_row.addWidget(QLabel("Date Range:"))
    filter_row.addWidget(start_picker)
    filter_row.addWidget(QLabel(" to "))
    filter_row.addWidget(end_picker)
    apply_btn = QPushButton("Apply")
    filter_row.addWidget(apply_btn)
    filter_row.addStretch()
    root.layout().addLayout(filter_row)

    # KPI grid
    kpi_grid = QGridLayout()
    kpi_grid.setSpacing(12)
    root.layout().addLayout(kpi_grid)

    # Tabs
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    def _render():
        d0 = pd.Timestamp(start_picker.date().toPython())
        d1 = pd.Timestamp(end_picker.date().toPython())
        df = _filter(base_df, d0, d1)

        # Clear old KPIs
        while kpi_grid.count():
            w = kpi_grid.takeAt(0).widget()
            if w:
                w.deleteLater()

        tabs.clear()

        if df.empty:
            tabs.addTab(QWidget(), "No data")
            return

        # KPI calculations
        total_bookings = (
            df["ReservationID"].nunique() if "ReservationID" in df.columns else 0
        )
        canceled = df[df["ReservationStatus"] == "Canceled"]
        no_show = df[df["ReservationStatus"] == "No-Show"]
        num_canceled = (
            canceled["ReservationID"].nunique()
            if "ReservationID" in canceled.columns
            else 0
        )
        num_no_show = (
            no_show["ReservationID"].nunique()
            if "ReservationID" in no_show.columns
            else 0
        )

        cancel_rate = num_canceled / total_bookings if total_bookings else 0
        noshow_rate = num_no_show / total_bookings if total_bookings else 0

        if (
            not canceled.empty
            and "CancellationDate" in canceled.columns
            and "BookingDate" in canceled.columns
        ):
            lead_time = (canceled["CancellationDate"] - canceled["BookingDate"]).dt.days
            avg_lead = lead_time.mean()
        else:
            avg_lead = np.nan

        if not canceled.empty:
            if "RefundAmount" in canceled:
                avg_refund = canceled["RefundAmount"].mean()
            else:
                if "ADR" in canceled and "NightsStayed" in canceled:
                    canceled["LostRevenue"] = canceled["ADR"] * canceled["NightsStayed"]
                    avg_refund = canceled["LostRevenue"].mean()
                else:
                    avg_refund = np.nan
        else:
            avg_refund = np.nan

        if "RoomRevenue" in canceled and "RoomRevenue" in df:
            revenue_at_risk = (
                canceled["RoomRevenue"].sum() / df["RoomRevenue"].sum()
                if df["RoomRevenue"].sum()
                else 0
            )
        else:
            revenue_at_risk = np.nan

        kpis = [
            ("Cancellation Rate", f"{cancel_rate:.1%}"),
            ("No-Show Rate", f"{noshow_rate:.1%}"),
            (
                "Avg Lead Time (days)",
                f"{avg_lead:.1f}" if not np.isnan(avg_lead) else "—",
            ),
            (
                "Avg Refund/Lost",
                f"${avg_refund:,.0f}" if not np.isnan(avg_refund) else "—",
            ),
            (
                "% Revenue At-Risk",
                f"{revenue_at_risk:.1%}" if not np.isnan(revenue_at_risk) else "—",
            ),
        ]
        for i, (lbl, val) in enumerate(kpis):
            kpi_grid.addWidget(kpi_tile(lbl, val), i // 3, i % 3)

        # Chart specs
        chart_spec = [
            (
                "Cancel Trend",
                _chart_cancellation_trend,
                {"Date", "ReservationStatus", "ReservationID"},
            ),
            (
                "No-Show Trend",
                _chart_noshow_trend,
                {"Date", "ReservationStatus", "ReservationID"},
            ),
            (
                "Bookings vs Cancel Rate",
                _chart_cancel_vs_volume,
                {"Date", "ReservationStatus", "ReservationID"},
            ),
            (
                "Lead Time Histo",
                _chart_lead_time_hist,
                {"ReservationStatus", "CancellationDate", "BookingDate"},
            ),
            (
                "Cancel by Source",
                _chart_cancel_by_source,
                {"Date", "ReservationStatus", "ReservationID", "BookingSource"},
            ),
            (
                "Segment Cancel %",
                _chart_segment_cancel_rate,
                {"MarketSegment", "ReservationStatus", "ReservationID"},
            ),
            (
                "Revenue Loss Heatmap",
                _chart_revenue_loss_heatmap,
                {"ReservationStatus", "CancellationDate", "ADR", "NightsStayed"},
            ),
            (
                "Refund vs Chargeback",
                _chart_refund_chargeback,
                {"Date", "ReservationStatus", "ReservationID"},
            ),
            (
                "Top Reasons",
                _chart_reason_bar,
                {"ReservationStatus", "CancellationReason"},
            ),
        ]
        for t, fn, req in chart_spec:
            _maybe(tabs, t, fn, df, req)

    apply_btn.clicked.connect(_render)
    _render()

    display.refresh_date_pickers = refresh_date_pickers
    return root
