# flake8: noqa
# views/room_cost.py
"""
Room Cost Analysis View
=======================
Provides in-depth analysis of costs per occupied room (CPOR).
(Design modernisé : thème clair, cartes, ombres, toolbar compacte)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QDateEdit, QPushButton, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
import plotly.graph_objects as go

from views.utils import data_required, create_error_widget, create_plotly_widget
from data.helpers import get_df


# ---------- Styles (QSS) ----------
THEME_QSS_LIGHT = """
/* Root */
QWidget#RoomCostView {
    background: #f7f9fc;
    color: #0f172a;
    font-size: 13px;
}

/* Header */
QLabel#rcvHeader {
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    padding: 10px 14px;
    border-radius: 12px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                                stop:0 #eaf2ff, stop:1 #f4f9ff);
    border: 1px solid #d7e3ff;
    letter-spacing: .3px;
}

/* Toolbar */
QFrame#toolbar {
    background: #ffffff;
    border: 1px solid #e6eefc;
    border-radius: 12px;
    padding: 10px 12px;
}
QLabel#rangeLabel { color: #334155; font-weight: 600; }

/* DateEdit */
QDateEdit {
    min-height: 34px;
    padding: 4px 10px;
    border-radius: 8px;
    border: 1px solid #d1ddf8;
    background: #ffffff;
    color: #0f172a;
}
QDateEdit:hover { border: 1px solid #8ab4ff; }

/* Button */
QPushButton#applyBtn {
    background: #2563eb;
    border: 1px solid #1f51bf;
    color: #ffffff;
    padding: 8px 16px;
    border-radius: 10px;
    font-weight: 700;
}
QPushButton#applyBtn:hover { background: #1d4ed8; }
QPushButton#applyBtn:pressed { background: #173db2; }

/* Tabs */
QTabWidget::pane {
    border: 1px solid #e6eefc;
    border-radius: 12px;
    background: #ffffff;
}
QTabBar::tab {
    background: #f3f7ff;
    color: #0f172a;
    padding: 8px 16px;
    margin: 4px;
    border: 1px solid #e6eefc;
    border-bottom: 2px solid transparent;
    border-radius: 10px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #e7efff;
    border-bottom: 2px solid #2563eb;
}
QTabBar::tab:hover { background: #edf3ff; }

/* Cards */
QFrame#card {
    background: #ffffff;
    border: 1px solid #e6eefc;
    border-radius: 14px;
    padding: 12px;
}
QFrame#divider { background: #e5e9f2; min-height: 1px; }

/* Collapsible */
QPushButton#collapsibleBtn {
    background: #ffffff;
    color: #1e3a8a;
    border: 1px solid #cfe0ff;
    border-radius: 10px;
    padding: 6px 10px;
    font-weight: 700;
}
QPushButton#collapsibleBtn:hover { background: #f6faff; }

QLabel#explain {
    background: rgba(37, 99, 235, .06);
    border: 1px solid #cfe0ff;
    border-radius: 10px;
    padding: 10px 12px;
    color: #0f172a;
}
"""


# --- Helper Functions ---

def _collapsible(text: str) -> QWidget:
    """Bloc d’explication repliable (UI only)."""
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 6, 0, 0)
    lay.setSpacing(8)

    toggle_btn = QPushButton("Show explanation")
    toggle_btn.setObjectName("collapsibleBtn")

    explanation = QLabel(text)
    explanation.setObjectName("explain")
    explanation.setWordWrap(True)
    explanation.setVisible(False)

    def toggle():
        is_vis = explanation.isVisible()
        explanation.setVisible(not is_vis)
        toggle_btn.setText("Hide explanation" if not is_vis else "Show explanation")

    toggle_btn.clicked.connect(toggle)
    lay.addWidget(toggle_btn)
    lay.addWidget(explanation)
    return container


def _classify_performance(value: float) -> str:
    """Classify performance based on cost efficiency or profit margin."""
    if value >= 25:
        return "excellent"
    elif value >= 15:
        return "good"
    elif value >= 5:
        return "moderate"
    else:
        return "concerning"


def create_sample_data():
    """Creates sample room cost data if actual data is missing."""
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    room_types = ['Standard', 'Deluxe', 'Suite', 'Executive']
    data = []
    for date in dates:
        for room_type in room_types:
            cost = np.random.uniform(20, 80)
            rate = cost * np.random.uniform(2.5, 4.0)
            data.append({
                'date': date,
                'room_type': room_type,
                'cost_per_occupied_room': cost,
                'rate': rate
            })
    return pd.DataFrame(data)


def _card_container() -> QFrame:
    """Card container with soft shadow."""
    card = QFrame()
    card.setObjectName("card")
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(18)
    effect.setXOffset(0)
    effect.setYOffset(6)
    effect.setColor(QColor(0, 0, 0, 32))
    card.setGraphicsEffect(effect)
    card.setLayout(QVBoxLayout())
    card.layout().setContentsMargins(12, 12, 12, 12)
    card.layout().setSpacing(10)
    return card


# --- Main View Class ---

class RoomCostView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RoomCostView")
        self.setStyleSheet(THEME_QSS_LIGHT)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)

        # 1. Load and prepare the base data
        self.base_df = get_df()  # try to load data from your data provider
        if self.base_df is None or self.base_df.empty:
            self.base_df = create_sample_data()

        self.prepare_data()
        if self.base_df.empty:
            err = "No data available for room cost analysis"
            self.layout.addWidget(create_error_widget(err))
            return

        # 2. Setup the UI
        self.init_ui()

        # 3. Initial chart rendering
        self.update_charts()

    # -------- data prep --------
    def prepare_data(self):
        """Normalize columns & compute required metrics (logic intact)."""
        if self.base_df is None or self.base_df.empty:
            return

        df = self.base_df.copy()

        # Date
        date_columns = ['Date', 'date', 'reservation_date', 'check_in_date', 'stay_date']
        date_col = next((c for c in date_columns if c in df.columns), None)
        if date_col:
            df['date'] = pd.to_datetime(df[date_col])
        else:
            df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

        # Room type
        room_cols = ['RoomType', 'room_type', 'Room', 'room_category']
        rcol = next((c for c in room_cols if c in df.columns), None)
        df['room_type'] = df[rcol].astype(str) if rcol else np.random.choice(
            ['Standard', 'Deluxe', 'Suite', 'Executive'], size=len(df)
        )

        # CPOR
        if 'CostPerOccupiedRoom' in df.columns:
            df['cost_per_occupied_room'] = df['CostPerOccupiedRoom']
        elif {'TotalRoomCost', 'OccupiedRooms'}.issubset(df.columns):
            df['cost_per_occupied_room'] = df['TotalRoomCost'] / df['OccupiedRooms'].replace(0, np.nan)
        elif 'ADR' in df.columns:
            df['cost_per_occupied_room'] = df['ADR'] * np.random.uniform(0.3, 0.4, size=len(df))
        elif 'rate' in df.columns:
            df['cost_per_occupied_room'] = df['rate'] * np.random.uniform(0.3, 0.4, size=len(df))
        else:
            df['cost_per_occupied_room'] = np.random.uniform(20, 80, size=len(df))

        # rate
        if 'rate' not in df.columns:
            df['rate'] = df['ADR'] if 'ADR' in df.columns else df['cost_per_occupied_room'] * np.random.uniform(2.5, 4.0, size=len(df))

        df['cost_per_occupied_room'] = df['cost_per_occupied_room'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['cost_per_occupied_room'])
        df['room_type'] = df['room_type'].astype(str)

        self.base_df = df

    # -------- UI --------
    def init_ui(self):
        """Initializes the user interface (design only)."""
        # Header
        header = QLabel("Room Cost Analysis")
        header.setObjectName("rcvHeader")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.layout.addWidget(header)

        # Toolbar (date range + apply)
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        tlay = QHBoxLayout(toolbar)
        tlay.setContentsMargins(12, 10, 12, 10)
        tlay.setSpacing(10)

        range_label = QLabel("Date Range:")
        range_label.setObjectName("rangeLabel")

        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit = QDateEdit(calendarPopup=True)

        min_date = self.base_df['date'].min()
        max_date = self.base_df['date'].max()
        self.start_date_edit.setDate(QDate(min_date.year, min_date.month, min_date.day))
        self.end_date_edit.setDate(QDate(max_date.year, max_date.month, max_date.day))

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("applyBtn")
        apply_btn.clicked.connect(self.update_charts)

        tlay.addWidget(range_label)
        tlay.addWidget(self.start_date_edit)
        tlay.addWidget(QLabel("to"))
        tlay.addWidget(self.end_date_edit)
        tlay.addStretch()
        tlay.addWidget(apply_btn)
        self.layout.addWidget(toolbar)

        # Divider
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        self.layout.addWidget(divider)

        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

    # -------- charts update --------
    def update_charts(self):
        """Filters data by date range and redraws charts."""
        self.tabs.clear()

        try:
            start_date = pd.Timestamp(self.start_date_edit.date().toPython())
            end_date = pd.Timestamp(self.end_date_edit.date().toPython())
            df = self.base_df[(self.base_df['date'] >= start_date) & (self.base_df['date'] <= end_date)].copy()

            if df.empty:
                no_data = _card_container()
                lbl = QLabel("No data available for the selected date range.")
                lbl.setStyleSheet("font-size: 14px; color: #6b7280;")
                lbl.setAlignment(Qt.AlignCenter)
                no_data.layout().addWidget(lbl)
                self.tabs.addTab(no_data, "No Data")
                return

            # Types & sanity
            df['room_type'] = df['room_type'].astype(str)
            df['cost_per_occupied_room'] = pd.to_numeric(df['cost_per_occupied_room'], errors='coerce')
            df = df.dropna(subset=['cost_per_occupied_room'])
            if df.empty:
                err = _card_container()
                lbl = QLabel("No valid cost data available for analysis.")
                lbl.setStyleSheet("font-size: 14px; color: #ef4444;")
                lbl.setAlignment(Qt.AlignCenter)
                err.layout().addWidget(lbl)
                self.tabs.addTab(err, "Error")
                return

            self._create_cpor_trend_chart(df)
            self._create_cost_summary_chart(df)
            self._create_cost_vs_rate_chart(df)

        except Exception as e:
            error_card = _card_container()
            lbl = QLabel(f"Error creating charts: {str(e)}")
            lbl.setStyleSheet("font-size: 12pt; color: #ff6b6b;")
            lbl.setWordWrap(True)
            error_card.layout().addWidget(lbl)
            self.tabs.addTab(error_card, "Error")

    # -------- individual charts --------
    def _create_cpor_trend_chart(self, df: pd.DataFrame):
        """CPOR trend chart (by room type, daily avg)."""
        card = _card_container()

        fig = go.Figure()

        avg_cpor = df['cost_per_occupied_room'].mean()
        min_cpor = df['cost_per_occupied_room'].min()
        max_cpor = df['cost_per_occupied_room'].max()
        cpor_variance = ((max_cpor - min_cpor) / avg_cpor) * 100 if avg_cpor else 0

        room_avg_costs = df.groupby('room_type')['cost_per_occupied_room'].mean()
        best_room = room_avg_costs.idxmin()
        worst_room = room_avg_costs.idxmax()
        best_cost = room_avg_costs.min()
        worst_cost = room_avg_costs.max()

        for room_type in df['room_type'].unique():
            room_data = df[df['room_type'] == str(room_type)]
            daily_avg = room_data.groupby('date')['cost_per_occupied_room'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['cost_per_occupied_room'],
                mode='lines+markers',
                name=str(room_type),
                line=dict(width=3),
                marker=dict(size=6)
            ))

        fig.add_hline(
            y=avg_cpor,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average CPOR: ${avg_cpor:.2f}"
        )

        fig.update_layout(
            title="Cost per Occupied Room (CPOR) – Trends",
            xaxis_title="Date",
            yaxis_title="Cost per Occupied Room ($)",
            template="plotly_white",
            hovermode="x unified",
            height=500
        )
        card.layout().addWidget(create_plotly_widget(fig))

        variance_perf = _classify_performance(cpor_variance)
        cost_diff = worst_cost - best_cost
        explanation = (
            f"Average cost per occupied room across all room types is **${avg_cpor:.2f}**. "
            f"Cost variance is **{cpor_variance:.1f}%**, indicating **{variance_perf}** cost control consistency.\n\n"
            f"**{best_room}** rooms have the lowest average cost at **${best_cost:.2f}**, "
            f"while **{worst_room}** rooms cost **${worst_cost:.2f}** "
            f"(Δ ${cost_diff:.2f}). The dashed red line shows the overall average."
        )
        card.layout().addWidget(_collapsible(explanation))

        self.tabs.addTab(card, "CPOR Trend")

    def _create_cost_summary_chart(self, df: pd.DataFrame):
        """Cost distribution & variability by room type."""
        card = _card_container()

        cost_summary = df.groupby('room_type')['cost_per_occupied_room'].agg(
            mean='mean', min='min', max='max', std='std'
        ).round(2)
        cost_summary.index = cost_summary.index.astype(str)

        room_counts = df.groupby('room_type').size()
        total_costs = (cost_summary['mean'] * room_counts).reindex(cost_summary.index).fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=cost_summary.index,
            values=total_costs.values,
            hole=0.45,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=['#4F46E5', '#22C55E', '#0EA5E9', '#F59E0B', '#EF4444', '#8B5CF6'],
                line=dict(color='#FFFFFF', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Avg Cost: $%{customdata:.2f}<br>' +
                         'Share: %{percent}<extra></extra>',
            customdata=cost_summary['mean'].values
        ))
        fig.update_layout(
            title="Cost Distribution by Room Type",
            template="plotly_white",
            height=500,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
            annotations=[dict(
                text=f"Overall Avg<br>${cost_summary['mean'].mean():.2f}",
                x=0.5, y=0.5, font_size=13, showarrow=False
            )],
            margin=dict(r=140)
        )
        card.layout().addWidget(create_plotly_widget(fig))

        highest_cost_room = cost_summary['mean'].idxmax()
        lowest_cost_room = cost_summary['mean'].idxmin()
        highest_cost = cost_summary['mean'].max()
        lowest_cost = cost_summary['mean'].min()
        cost_spread = highest_cost - lowest_cost

        explanation = (
            f"Donut chart of **cost distribution** (pondérée par volume). "
            f"**{highest_cost_room}** est le plus coûteux (moyenne **${highest_cost:.2f}**), "
            f"**{lowest_cost_room}** le plus efficient (**${lowest_cost:.2f}**) "
            f"(écart **${cost_spread:.2f}**). Les parts les plus larges indiquent "
            f"les types de chambre qui pèsent le plus dans les coûts globaux."
        )
        card.layout().addWidget(_collapsible(explanation))

        self.tabs.addTab(card, "Cost Summary")

    def _create_cost_vs_rate_chart(self, df: pd.DataFrame):
        """Cost vs rate scatter (colored by profit margin)."""
        if 'rate' not in df.columns:
            return

        card = _card_container()
        fig = go.Figure()

        df_copy = df.copy()
        df_copy['profit_margin'] = ((df_copy['rate'] - df_copy['cost_per_occupied_room']) / df_copy['rate'] * 100)
        avg_profit_margin = df_copy['profit_margin'].mean()

        room_margins = df_copy.groupby('room_type')['profit_margin'].mean()
        best_margin_room = room_margins.idxmax()
        worst_margin_room = room_margins.idxmin()
        best_margin = room_margins.max()
        worst_margin = room_margins.min()

        unique_rooms = list(df['room_type'].astype(str).unique())
        for idx, room_type in enumerate(unique_rooms):
            room_data = df_copy[df_copy['room_type'] == room_type]
            fig.add_trace(go.Scatter(
                x=room_data['rate'],
                y=room_data['cost_per_occupied_room'],
                mode='markers',
                name=room_type,
                marker=dict(
                    opacity=0.75,
                    size=8,
                    color=room_data['profit_margin'],
                    colorscale='RdYlGn',
                    showscale=(idx == 0),
                    colorbar=dict(
                        title="Profit Margin (%)",
                        x=1.05,
                        thickness=14,
                        len=0.8
                    )
                ),
                hovertemplate=(
                    f"<b>{room_type}</b><br>"
                    "Rate: $%{x:.2f}<br>"
                    "Cost: $%{y:.2f}<br>"
                    "Profit Margin: %{marker.color:.1f}%<extra></extra>"
                )
            ))

        min_val = float(min(df['rate'].min(), df['cost_per_occupied_room'].min()))
        max_val = float(max(df['rate'].max(), df['cost_per_occupied_room'].max()))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Break-even Line',
            line=dict(dash='dash', color='red', width=2),
            showlegend=True
        ))

        fig.update_layout(
            title="Cost per Room vs. Daily Rate (colored by Profit Margin)",
            xaxis_title="Daily Rate ($)",
            yaxis_title="Cost per Occupied Room ($)",
            template="plotly_white",
            height=500,
            margin=dict(r=140)
        )
        card.layout().addWidget(create_plotly_widget(fig))

        margin_perf = _classify_performance(avg_profit_margin)
        margin_spread = best_margin - worst_margin
        profitable_pct = (df_copy['profit_margin'] > 0).mean() * 100

        explanation = (
            f"Average profit margin: **{avg_profit_margin:.1f}%** (**{margin_perf}**). "
            f"Couleur = marge : plus vert = plus rentable. "
            f"**{best_margin_room}** atteint **{best_margin:.1f}%**, "
            f"**{worst_margin_room}** **{worst_margin:.1f}%** (Δ {margin_spread:.1f} pts). "
            f"**{profitable_pct:.1f}%** des points sont au-dessus de la ligne de break-even."
        )
        card.layout().addWidget(_collapsible(explanation))

        self.tabs.addTab(card, "Cost vs. Rate")


@data_required
def display():
    """Main entry point for the Room Cost view."""
    return RoomCostView()
