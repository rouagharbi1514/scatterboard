# flake8: noqa
# views/revenue.py
"""
Revenue Analysis View — modern soft UI
======================================

• Palette pastel, cartes, onglets arrondis (mêmes styles que les autres vues)
• Date-range toolbar compacte
• Panneaux "Show insight" repliables
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDateEdit, QPushButton,
    QTabWidget, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QDate
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.helpers import get_df
from views.utils import data_required

# (Facultatif) import du validateur si vous l’utilisez ailleurs
from views.data_validator import DataValidator, validate_revenue_data  # noqa: F401


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


def _plotly_view(fig: go.Figure) -> QWebEngineView:
    """Create a QWebEngineView widget from a Plotly figure."""
    from io import StringIO
    html_buf = StringIO()
    fig.write_html(html_buf, include_plotlyjs="cdn", full_html=False)
    view = QWebEngineView()
    view.setHtml(html_buf.getvalue())
    view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return view


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
    if value >= 30:
        return "strong"
    elif value >= 15:
        return "moderate"
    else:
        return "weak"


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


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:
    """Display revenue analysis dashboard (modern UI)."""
    try:
        base_df = get_df()

        # UI root + QSS global (même design que les autres vues)
        root = QWidget()
        root.setObjectName("RevenueRoot")
        root.setLayout(QVBoxLayout())
        root.layout().setContentsMargins(12, 12, 12, 12)
        root.layout().setSpacing(10)

        root.setStyleSheet("""
            QWidget#RevenueRoot {
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
        header = QLabel("Revenue Analysis")
        header.setObjectName("header")
        root.layout().addWidget(header)

        # Toolbar (date range)
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 10, 12, 10)
        tl.setSpacing(8)

        # Date pickers
        start_picker = QDateEdit()
        end_picker = QDateEdit()
        for p in (start_picker, end_picker):
            p.setCalendarPopup(True)

        # Set initial date range
        def refresh_date_pickers():
            try:
                col = "date" if "date" in base_df.columns else "Date"
                if col in base_df.columns:
                    d0 = pd.to_datetime(base_df[col]).min().date()
                    d1 = pd.to_datetime(base_df[col]).max().date()
                    start_picker.setDate(QDate(d0.year, d0.month, d0.day))
                    end_picker.setDate(QDate(d1.year, d1.month, d1.day))
            except Exception as e:
                print(f"Error setting date range: {e}")

        refresh_date_pickers()

        tl.addWidget(QLabel("Date Range:"))
        tl.addWidget(start_picker)
        tl.addWidget(QLabel(" to "))
        tl.addWidget(end_picker)
        tl.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("applyBtn")
        tl.addWidget(apply_btn)
        root.layout().addWidget(toolbar)

        # Tabs container
        content = QTabWidget()
        root.layout().addWidget(content, 1)

        # Helpers
        def _ensure_date_col(df: pd.DataFrame) -> str:
            return "date" if "date" in df.columns else "Date"

        def filter_data(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
            col = _ensure_date_col(df)
            d = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(d[col]):
                d[col] = pd.to_datetime(d[col])
            mask = (d[col] >= start_date) & (d[col] <= end_date)
            return d.loc[mask]

        # Render
        def render():
            try:
                content.clear()

                start_date = pd.Timestamp(start_picker.date().toPython())
                end_date = pd.Timestamp(end_picker.date().toPython())
                df = filter_data(base_df, start_date, end_date)

                if df.empty:
                    empty = QWidget()
                    lay = QVBoxLayout(empty)
                    msg = QLabel("No data available for the selected date range")
                    msg.setStyleSheet("font-size: 14pt; color: #8b5e34;")
                    lay.addWidget(msg, 0, Qt.AlignCenter)
                    content.addTab(empty, "Info")
                    return

                # Normalisations
                date_col = _ensure_date_col(df)
                df = df.copy()
                df.rename(columns={date_col: "date"}, inplace=True)

                # Calculs auxiliaires
                ROOM_COUNT = 100
                if "rate" in df.columns and "occupancy" in df.columns:
                    df["revpar"] = df["rate"] * df["occupancy"]
                    df["calculated_revenue"] = df["rate"] * df["occupancy"] * ROOM_COUNT

                if "cost_per_occupied_room" not in df.columns and "rate" in df.columns:
                    df["cost_per_occupied_room"] = df["rate"] * 0.4

                if {"cost_per_occupied_room", "occupancy"}.issubset(df.columns):
                    df["cost"] = df["cost_per_occupied_room"] * df["occupancy"] * ROOM_COUNT

                # KPIs
                avg_occ = (df["occupancy"].mean() * 100) if "occupancy" in df.columns else 0
                avg_rate = df["rate"].mean() if "rate" in df.columns else 0
                avg_revpar = (
                    df["revpar"].mean()
                    if "revpar" in df.columns
                    else (df["rate"] * df["occupancy"]).mean() if {"rate", "occupancy"}.issubset(df.columns) else 0
                )
                total_revenue = df["calculated_revenue"].sum() if "calculated_revenue" in df.columns else 0
                total_cost = df["cost"].sum() if "cost" in df.columns else 0
                profit = total_revenue - total_cost
                profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0

                # ========================
                # Dashboard Tab
                # ========================
                dashboard_tab = QWidget()
                dashboard_tab.setLayout(QVBoxLayout())

                # Subplots: 2x2 (avec axes secondaires où nécessaire)
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[
                        [{"secondary_y": False}, {"type": "domain"}],
                        [{"secondary_y": True}, {"secondary_y": True}],
                    ],
                    subplot_titles=(
                        "Revenue & Cost Trend", "Revenue by Day of Week",
                        "Occupancy & Rate Trend", "Profit & Margin"
                    ),
                    vertical_spacing=0.15, horizontal_spacing=0.15
                )

                # 1) Revenue & Cost Trend (1,1)
                daily = df.groupby("date").agg(
                    calculated_revenue=("calculated_revenue", "sum"),
                    cost=("cost", "sum")
                ).reset_index()

                fig.add_trace(go.Scatter(
                    x=daily["date"], y=daily["calculated_revenue"],
                    name="Revenue", mode="lines+markers",
                    line=dict(color=SOFT_BLUE, width=3),
                    marker=dict(size=7, line=dict(color="white", width=1))
                ), row=1, col=1)

                if "cost" in daily.columns and daily["cost"].notna().any():
                    fig.add_trace(go.Scatter(
                        x=daily["date"], y=daily["cost"],
                        name="Cost", mode="lines+markers",
                        line=dict(color=SOFT_RED, width=3),
                        marker=dict(size=7, line=dict(color="white", width=1))
                    ), row=1, col=1)

                # 2) Revenue by Day of Week (1,2)
                if "calculated_revenue" in df.columns:
                    df["weekday"] = df["date"].dt.day_name()
                    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    w = (df.groupby("weekday")["calculated_revenue"].sum()
                         .reindex(weekday_order)
                         .fillna(0)
                         .reset_index())

                    fig.add_trace(go.Pie(
                        labels=w["weekday"], values=w["calculated_revenue"], hole=0.45,
                        marker_colors=[SOFT_BLUE, "#93c5fd", "#bfdbfe", "#dbeafe", "#eff6ff", SOFT_MINT, SOFT_TEAL],
                        textinfo="percent+label", hoverinfo="label+value+percent", textposition="inside",
                        name="Revenue by Weekday"
                    ), row=1, col=2)
                else:
                    fig.add_trace(go.Pie(labels=["No revenue"], values=[1], hole=0.45), row=1, col=2)

                # 3) Occupancy & Rate Trend (2,1) — axes secondaires
                if {"occupancy", "rate"}.issubset(df.columns):
                    daily_m = df.groupby("date").agg(occupancy=("occupancy", "mean"), rate=("rate", "mean")).reset_index()
                    fig.add_trace(go.Scatter(
                        x=daily_m["date"], y=daily_m["occupancy"] * 100,
                        name="Occupancy (%)", mode="lines+markers",
                        line=dict(color=SOFT_TEAL, width=3),
                        marker=dict(size=7, line=dict(color="white", width=1))
                    ), row=2, col=1, secondary_y=False)

                    fig.add_trace(go.Scatter(
                        x=daily_m["date"], y=daily_m["rate"],
                        name="Avg Rate ($)", mode="lines+markers",
                        line=dict(color=SOFT_AMBER, width=3, dash="dash"),
                        marker=dict(size=7, line=dict(color="white", width=1))
                    ), row=2, col=1, secondary_y=True)

                    fig.update_yaxes(title_text="Occupancy (%)", secondary_y=False, row=2, col=1)
                    fig.update_yaxes(title_text="Avg Rate ($)", secondary_y=True, row=2, col=1)

                # 4) Profit & Margin (2,2)
                if {"calculated_revenue", "cost"}.issubset(df.columns):
                    pr = df.groupby("date").agg(
                        revenue=("calculated_revenue", "sum"),
                        cost=("cost", "sum")
                    ).reset_index()
                    pr["profit"] = pr["revenue"] - pr["cost"]
                    pr["margin%"] = np.where(pr["revenue"] > 0, pr["profit"] / pr["revenue"] * 100, 0.0)

                    fig.add_trace(go.Bar(
                        x=pr["date"], y=pr["profit"], name="Profit", marker_color=SOFT_GREEN,
                        text=[f"${v:,.0f}" for v in pr["profit"]], textposition="outside"
                    ), row=2, col=2, secondary_y=False)

                    fig.add_trace(go.Scatter(
                        x=pr["date"], y=pr["margin%"], name="Profit Margin (%)",
                        mode="lines+markers", line=dict(color=SOFT_LILAC, width=3),
                        marker=dict(size=7, line=dict(color="white", width=1))
                    ), row=2, col=2, secondary_y=True)

                    fig.update_yaxes(title_text="Profit ($)", tickformat="$,.0f", secondary_y=False, row=2, col=2)
                    fig.update_yaxes(title_text="Margin (%)", ticksuffix="%", secondary_y=True, row=2, col=2)

                _apply_soft_layout(fig, "Dashboard", height=900)

                # Highest weekday for insight
                if "calculated_revenue" in df.columns:
                    w2 = df.groupby(df["date"].dt.day_name())["calculated_revenue"].sum()
                    highest_day = w2.idxmax()
                    highest_val = w2.max()
                else:
                    highest_day, highest_val = "—", 0

                perf_class = _classify_performance(profit_margin)
                dashboard_expl = (
                    f"Overall profit margin is **{profit_margin:.1f}%** (**{perf_class}**). "
                    f"A widening gap between revenue (blue) and cost (red) signale une marge en amélioration. "
                    f"\n\nLe jour générant le plus de revenu est **{highest_day}** (≈ ${highest_val:,.0f}). "
                    f"ADR moyen **${avg_rate:.2f}**, Occupancy **{avg_occ:.1f}%**, RevPAR **${avg_revpar:.2f}**."
                )

                # Card + insight
                dash_card = _card_wrap(_plotly_view(fig))
                dashboard_tab.layout().addWidget(dash_card)
                dashboard_tab.layout().addWidget(_collapsible(dashboard_expl))
                content.addTab(dashboard_tab, "Dashboard")

                # ========================
                # Detailed Analysis Tab
                # ========================
                analysis_tab = QWidget()
                analysis_tab.setLayout(QVBoxLayout())

                analysis_fig = make_subplots(
                    rows=2, cols=1,
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                    subplot_titles=("Revenue Composition", "Performance Metrics"),
                    vertical_spacing=0.18
                )

                # Revenue Composition (by room_type) + Avg Rate (secondary)
                if {"room_type", "calculated_revenue"}.issubset(df.columns):
                    room_rev = df.groupby("room_type")["calculated_revenue"].sum().sort_values(ascending=False)
                    analysis_fig.add_trace(go.Bar(
                        x=room_rev.index, y=room_rev.values, name="Revenue by Room Type",
                        marker_color=SOFT_BLUE, opacity=0.9
                    ), row=1, col=1, secondary_y=False)

                    room_rate = df.groupby("room_type")["rate"].mean().reindex(room_rev.index) if "rate" in df.columns else None
                    if room_rate is not None:
                        analysis_fig.add_trace(go.Scatter(
                            x=room_rate.index, y=room_rate.values, name="Avg Rate",
                            mode="lines+markers",
                            line=dict(color=SOFT_AMBER, width=3),
                            marker=dict(size=7, line=dict(color="white", width=1))
                        ), row=1, col=1, secondary_y=True)

                    analysis_fig.update_yaxes(title_text="Revenue ($)", tickformat="$,.0f", row=1, col=1)
                    analysis_fig.update_yaxes(title_text="ADR ($)", secondary_y=True, row=1, col=1)

                # Performance Metrics: Occupancy (primary) + ADR & RevPAR (secondary)
                if "date" in df.columns and ("occupancy" in df.columns or "rate" in df.columns or "revpar" in df.columns):
                    met = df.groupby("date").agg(
                        occupancy=("occupancy", "mean") if "occupancy" in df.columns else ("date", "count"),
                        rate=("rate", "mean") if "rate" in df.columns else ("date", "count"),
                        revpar=("revpar", "mean") if "revpar" in df.columns else ("date", "count")
                    ).reset_index()

                    if "occupancy" in df.columns:
                        analysis_fig.add_trace(go.Scatter(
                            x=met["date"], y=met["occupancy"] * 100, name="Occupancy (%)",
                            mode="lines", line=dict(color=SOFT_TEAL, width=3)
                        ), row=2, col=1, secondary_y=False)

                    if "rate" in df.columns:
                        analysis_fig.add_trace(go.Scatter(
                            x=met["date"], y=met["rate"], name="ADR",
                            mode="lines", line=dict(color=SOFT_BLUE, width=3)
                        ), row=2, col=1, secondary_y=True)

                    if "revpar" in df.columns:
                        analysis_fig.add_trace(go.Scatter(
                            x=met["date"], y=met["revpar"], name="RevPAR",
                            mode="lines", line=dict(color=SOFT_LILAC, width=3)
                        ), row=2, col=1, secondary_y=True)

                    analysis_fig.update_yaxes(title_text="Occupancy (%)", secondary_y=False, row=2, col=1)
                    analysis_fig.update_yaxes(title_text="ADR / RevPAR ($)", secondary_y=True, row=2, col=1)

                _apply_soft_layout(analysis_fig, "Detailed Revenue Analysis", height=780)

                # Insight
                if "revpar" in df.columns:
                    met2 = df.groupby("date")["revpar"].mean()
                    if len(met2) > 1 and met2.iloc[0] != 0:
                        revpar_change = (met2.iloc[-1] / met2.iloc[0] - 1) * 100
                        trend = "increasing" if revpar_change > 0 else "decreasing"
                        perf = _classify_performance(abs(revpar_change))
                    else:
                        revpar_change, trend, perf = 0, "flat", "weak"
                else:
                    revpar_change, trend, perf = 0, "flat", "weak"

                analysis_expl = (
                    f"RevPAR shows a **{perf} {trend}** trend ({revpar_change:.1f}%). "
                    f"Average occupancy **{avg_occ:.1f}%** and ADR **${avg_rate:.2f}**."
                )

                analysis_tab.layout().addWidget(_card_wrap(_plotly_view(analysis_fig)))
                analysis_tab.layout().addWidget(_collapsible(analysis_expl))
                content.addTab(analysis_tab, "Detailed Analysis")

                # ========================
                # Forecast Tab
                # ========================
                forecast_tab = QWidget()
                forecast_tab.setLayout(QVBoxLayout())

                forecast_fig = go.Figure()
                if "calculated_revenue" in daily.columns and len(daily) >= 5:
                    last_date = daily["date"].max()
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

                    x = np.arange(len(daily))
                    y = daily["calculated_revenue"].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    fc_vals = p(np.arange(len(daily), len(daily) + 7))

                    forecast_fig.add_trace(go.Scatter(
                        x=daily["date"], y=daily["calculated_revenue"],
                        name="Actual Revenue", mode="lines+markers",
                        line=dict(color=SOFT_BLUE, width=3)
                    ))
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast_dates, y=fc_vals,
                        name="Forecast", mode="lines+markers",
                        line=dict(color=SOFT_RED, width=3, dash="dot")
                    ))
                    _apply_soft_layout(forecast_fig, "7-Day Revenue Forecast")

                    # Insight
                    fc_trend = "increasing" if fc_vals[-1] > fc_vals[0] else "decreasing"
                    base = daily["calculated_revenue"].iloc[-1] if daily["calculated_revenue"].iloc[-1] != 0 else 1
                    fc_change = (fc_vals[-1] / base - 1) * 100
                    fc_strength = _classify_performance(abs(fc_change))
                    total_fc = float(np.sum(fc_vals))
                    fc_profit = total_fc * (profit_margin / 100)

                    forecast_expl = (
                        f"Projected **{fc_trend}** trend over 7 days (**{fc_strength}**, {fc_change:.1f}%). "
                        f"Total forecasted revenue ≈ **${total_fc:,.0f}**, "
                        f"expected profit ≈ **${fc_profit:,.0f}** (margin {profit_margin:.1f}%)."
                    )

                    forecast_tab.layout().addWidget(_card_wrap(_plotly_view(forecast_fig)))
                    forecast_tab.layout().addWidget(_collapsible(forecast_expl))
                    content.addTab(forecast_tab, "Forecast")
                else:
                    nof = QWidget()
                    lnf = QVBoxLayout(nof)
                    lnf.addWidget(QLabel("Not enough data to build a forecast."), 0, Qt.AlignCenter)
                    forecast_tab.layout().addWidget(_card_wrap(nof))
                    content.addTab(forecast_tab, "Forecast")

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_widget = QWidget()
                lay = QVBoxLayout(error_widget)
                lbl = QLabel(f"Error: {str(e)}")
                lbl.setStyleSheet("color: red;")
                lay.addWidget(lbl, 0, Qt.AlignCenter)
                content.addTab(error_widget, "Error")

        apply_btn.clicked.connect(render)
        render()

        # exposer pour resync externe
        display.refresh_date_pickers = refresh_date_pickers  # type: ignore

        return root

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_label = QLabel(f"Error loading Revenue View: {str(e)}")
        error_label.setStyleSheet("color: red; font-size: 14pt;")
        error_layout.addWidget(error_label, alignment=Qt.AlignCenter)
        return error_widget


# Debug helper
def debug_view_loading():
    try:
        display()
    except Exception as e:
        print(f"Error loading revenue view: {e}")
        import traceback
        traceback.print_exc()
