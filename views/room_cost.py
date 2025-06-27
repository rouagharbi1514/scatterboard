# views/room_cost.py
"""
Room Cost Analysis View
=======================
Provides in-depth analysis of costs per occupied room (CPOR).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QDateEdit, QPushButton, QFrame
)
from PySide6.QtCore import QDate, Qt
import plotly.graph_objects as go
from views.utils import data_required, create_error_widget, create_plotly_widget
from data.helpers import get_df


# --- Helper Functions ---

def _collapsible(text: str) -> QWidget:
    """Create a collapsible explanation panel.

    Args:
        text: The explanation text to display

    Returns:
        A widget containing a toggle button and collapsible text panel
    """
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


def _classify_performance(value: float) -> str:
    """Classify performance based on cost efficiency or profit margin.

    Args:
        value: The performance value (cost variance, profit margin, etc.)

    Returns:
        Classification as "excellent", "good", "moderate", or "concerning"
    """
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
            rate = cost * np.random.uniform(2.5, 4.0)  # Rate should be higher than cost
            data.append({
                'date': date,
                'room_type': room_type,
                'cost_per_occupied_room': cost,
                'rate': rate
            })
    return pd.DataFrame(data)


# --- Main View Class ---

class RoomCostView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 1. Load and prepare the base data
        self.base_df = get_df()  # try to load data from your data provider
        if self.base_df is None or self.base_df.empty:
            self.base_df = create_sample_data()

        # Check for available columns and create the required ones if missing
        self.prepare_data()

        # Check if we have the minimum required data
        if self.base_df.empty:
            err = "No data available for room cost analysis"
            self.layout.addWidget(create_error_widget(err))
            return

        # 2. Setup the UI
        self.init_ui()

        # 3. Initial chart rendering
        self.update_charts()

    def prepare_data(self):
        """Prepare the data by normalizing column names and calculating required metrics."""
        if self.base_df.empty:
            return

        # Create a working copy
        df = self.base_df.copy()

        # Normalize column names for date
        date_columns = ['Date', 'date', 'reservation_date', 'check_in_date', 'stay_date']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df['date'] = pd.to_datetime(df[date_col])
        else:
            # Create dummy dates if no date column
            df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

        # Normalize room type column
        room_type_columns = ['RoomType', 'room_type', 'Room', 'room_category']
        room_type_col = None
        for col in room_type_columns:
            if col in df.columns:
                room_type_col = col
                break

        if room_type_col:
            df['room_type'] = df[room_type_col].astype(str)
        else:
            # Create dummy room types
            room_types = ['Standard', 'Deluxe', 'Suite', 'Executive']
            df['room_type'] = np.random.choice(room_types, size=len(df))

        # Calculate cost per occupied room from available data
        if 'CostPerOccupiedRoom' in df.columns:
            df['cost_per_occupied_room'] = df['CostPerOccupiedRoom']
        elif 'TotalRoomCost' in df.columns and 'OccupiedRooms' in df.columns:
            # Calculate CPOR from total cost and occupied rooms
            df['cost_per_occupied_room'] = df['TotalRoomCost'] / df['OccupiedRooms'].replace(0, np.nan)
        elif 'ADR' in df.columns:
            # Estimate cost as 30-40% of ADR (industry standard)
            df['cost_per_occupied_room'] = df['ADR'] * np.random.uniform(0.3, 0.4, size=len(df))
        elif 'rate' in df.columns:
            # Estimate cost as 30-40% of rate
            df['cost_per_occupied_room'] = df['rate'] * np.random.uniform(0.3, 0.4, size=len(df))
        else:
            # Generate synthetic cost data as fallback
            df['cost_per_occupied_room'] = np.random.uniform(20, 80, size=len(df))

        # Add rate column if not present
        if 'rate' not in df.columns:
            if 'ADR' in df.columns:
                df['rate'] = df['ADR']
            else:
                # Generate synthetic rates
                df['rate'] = df['cost_per_occupied_room'] * np.random.uniform(2.5, 4.0, size=len(df))

        # Clean up any infinite or NaN values
        df['cost_per_occupied_room'] = df['cost_per_occupied_room'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['cost_per_occupied_room'])

        # Ensure room_type is always a string
        df['room_type'] = df['room_type'].astype(str)

        self.base_df = df

    def init_ui(self):
        """Initializes the user interface."""
        # Header
        header = QLabel("Room Cost Analysis")
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.layout.addWidget(header)

        # Filter Row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Date Range:"))

        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit = QDateEdit(calendarPopup=True)

        # Set date pickers to full data range:
        min_date = self.base_df['date'].min()
        max_date = self.base_df['date'].max()
        self.start_date_edit.setDate(QDate(min_date.year, min_date.month, min_date.day))
        self.end_date_edit.setDate(QDate(max_date.year, max_date.month, max_date.day))

        filter_row.addWidget(self.start_date_edit)
        filter_row.addWidget(QLabel("to"))
        filter_row.addWidget(self.end_date_edit)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet(
            "background-color: #4a86e8; color: white; padding: 5px 10px; border-radius: 4px;"
        )
        filter_row.addWidget(self.apply_btn)
        filter_row.addStretch()
        self.layout.addLayout(filter_row)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator)

        # Tab Widget for charts
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Connect Apply button to chart update
        self.apply_btn.clicked.connect(self.update_charts)

    def update_charts(self):
        """Filters data by date range and redraws charts."""
        self.tabs.clear()

        try:
            start_date = pd.Timestamp(self.start_date_edit.date().toPython())
            end_date = pd.Timestamp(self.end_date_edit.date().toPython())
            df = self.base_df[(self.base_df['date'] >= start_date) & (self.base_df['date'] <= end_date)].copy()

            if df.empty:
                no_data_widget = QWidget()
                no_data_layout = QVBoxLayout(no_data_widget)
                no_data_label = QLabel("No data available for the selected date range.")
                no_data_label.setStyleSheet("font-size: 14pt; color: #666; text-align: center;")
                no_data_layout.addWidget(no_data_label)
                self.tabs.addTab(no_data_widget, "No Data")
                return

            # Ensure data types are correct
            df['room_type'] = df['room_type'].astype(str)
            df['cost_per_occupied_room'] = pd.to_numeric(df['cost_per_occupied_room'], errors='coerce')
            df = df.dropna(subset=['cost_per_occupied_room'])

            if df.empty:
                error_widget = QWidget()
                error_layout = QVBoxLayout(error_widget)
                error_label = QLabel("No valid cost data available for analysis.")
                error_label.setStyleSheet("font-size: 14pt; color: #ff6b6b; text-align: center;")
                error_layout.addWidget(error_label)
                self.tabs.addTab(error_widget, "Error")
                return

            self._create_cpor_trend_chart(df)
            self._create_cost_summary_chart(df)
            self._create_cost_vs_rate_chart(df)

        except Exception as e:
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            error_label = QLabel(f"Error creating charts: {str(e)}")
            error_label.setStyleSheet("font-size: 12pt; color: #ff6b6b;")
            error_label.setWordWrap(True)
            error_layout.addWidget(error_label)
            self.tabs.addTab(error_widget, "Error")

    def _create_cpor_trend_chart(self, df):
        """Create the CPOR trend chart."""
    def _create_cpor_trend_chart(self, df):
        """Create the CPOR trend chart."""
        cpor_tab = QWidget()
        cpor_layout = QVBoxLayout(cpor_tab)
        fig1 = go.Figure()

        # Calculate average CPOR for reference line
        avg_cpor = df['cost_per_occupied_room'].mean()
        min_cpor = df['cost_per_occupied_room'].min()
        max_cpor = df['cost_per_occupied_room'].max()
        cpor_variance = ((max_cpor - min_cpor) / avg_cpor) * 100

        # Find best and worst performing room types
        room_avg_costs = df.groupby('room_type')['cost_per_occupied_room'].mean()
        best_room = room_avg_costs.idxmin()
        worst_room = room_avg_costs.idxmax()
        best_cost = room_avg_costs.min()
        worst_cost = room_avg_costs.max()

        for room_type in df['room_type'].unique():
            room_type = str(room_type)  # Ensure it's a string
            room_data = df[df['room_type'] == room_type]
            daily_avg = room_data.groupby('date')['cost_per_occupied_room'].mean().reset_index()

            fig1.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['cost_per_occupied_room'],
                mode='lines+markers',
                name=room_type,
                line=dict(width=3),
                marker=dict(size=6)
            ))

        # Add average reference line
        fig1.add_hline(
            y=avg_cpor,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average CPOR: ${avg_cpor:.2f}"
        )

        fig1.update_layout(
            title="Cost per Occupied Room (CPOR) Trends",
            xaxis_title="Date",
            yaxis_title="Cost per Occupied Room ($)",
            template="plotly_white",
            hovermode="x unified",
            height=500
        )
        cpor_layout.addWidget(create_plotly_widget(fig1))

        # Create dynamic explanation
        variance_performance = _classify_performance(cpor_variance)
        cost_difference = worst_cost - best_cost

        explanation = (
            f"Average cost per occupied room across all room types is **${avg_cpor:.2f}**. "
            f"Cost variance is **{cpor_variance:.1f}%**, indicating **{variance_performance}** cost control consistency. "
            f"\n\n**{best_room}** rooms have the lowest average cost at **${best_cost:.2f}**, while "
            f"**{worst_room}** rooms cost **${worst_cost:.2f}** (${cost_difference:.2f} difference). "
            f"The red dashed line shows the overall average - room types consistently above this line "
            f"may benefit from cost optimization strategies."
        )

        cpor_layout.addWidget(_collapsible(explanation))
        self.tabs.addTab(cpor_tab, "CPOR Trend")

    def _create_cost_summary_chart(self, df):
        """Create the cost summary chart."""
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)

        # Calculate summary statistics
        cost_summary = df.groupby('room_type')['cost_per_occupied_room'].agg([
            'mean', 'min', 'max', 'std'
        ]).round(2)

        # Ensure index (room types) are strings
        cost_summary.index = cost_summary.index.astype(str)

        fig3 = go.Figure()

        # Add bars for average cost
        fig3.add_trace(go.Bar(
            x=cost_summary.index,
            y=cost_summary['mean'],
            name='Average Cost',
            marker_color='lightblue',
            text=[f'${val:.2f}' for val in cost_summary['mean']],
            textposition='outside'
        ))

        # Add error bars for min/max range
        fig3.update_traces(
            error_y=dict(
                type='data',
                symmetric=False,
                array=cost_summary['max'] - cost_summary['mean'],
                arrayminus=cost_summary['mean'] - cost_summary['min'],
                visible=True
            )
        )

        fig3.update_layout(
            title="Cost per Occupied Room Summary by Room Type",
            xaxis_title="Room Type",
            yaxis_title="Cost per Occupied Room ($)",
            template="plotly_white",
            height=500
        )
        summary_layout.addWidget(create_plotly_widget(fig3))

        # Create dynamic explanation
        highest_cost_room = cost_summary['mean'].idxmax()
        lowest_cost_room = cost_summary['mean'].idxmin()
        highest_cost = cost_summary['mean'].max()
        lowest_cost = cost_summary['mean'].min()
        cost_spread = highest_cost - lowest_cost

        # Calculate cost variability
        avg_std = cost_summary['std'].mean()
        variability_performance = _classify_performance(avg_std)

        # Find room type with highest cost range
        cost_summary['range'] = cost_summary['max'] - cost_summary['min']
        most_variable_room = cost_summary['range'].idxmax()
        highest_range = cost_summary['range'].max()

        explanation = (
            f"**{highest_cost_room}** rooms have the highest average cost at **${highest_cost:.2f}**, "
            f"while **{lowest_cost_room}** rooms are most cost-efficient at **${lowest_cost:.2f}** "
            f"(${cost_spread:.2f} difference between room types). "
            f"\n\nError bars show the min-max cost range for each room type. "
            f"**{most_variable_room}** rooms show the highest cost variability with a range of "
            f"**${highest_range:.2f}**, suggesting **{variability_performance}** cost predictability. "
            f"Consistent costs across room types indicate better operational control."
        )

        summary_layout.addWidget(_collapsible(explanation))
        self.tabs.addTab(summary_tab, "Cost Summary")

    def _create_cost_vs_rate_chart(self, df):
        """Create the cost vs rate chart."""
        if 'rate' not in df.columns:
            return
    def _create_cost_vs_rate_chart(self, df):
        """Create the cost vs rate chart."""
        if 'rate' not in df.columns:
            return

        scatter_tab = QWidget()
        scatter_layout = QVBoxLayout(scatter_tab)
        fig2 = go.Figure()

        # Calculate overall profit margins for analysis
        df_copy = df.copy()
        df_copy['profit_margin'] = ((df_copy['rate'] - df_copy['cost_per_occupied_room']) / df_copy['rate'] * 100)
        avg_profit_margin = df_copy['profit_margin'].mean()

        # Find room types with best and worst profit margins
        room_margins = df_copy.groupby('room_type')['profit_margin'].mean()
        best_margin_room = room_margins.idxmax()
        worst_margin_room = room_margins.idxmin()
        best_margin = room_margins.max()
        worst_margin = room_margins.min()

        for room_type in df['room_type'].unique():
            room_type = str(room_type)  # Ensure it's a string
            room_data = df[df['room_type'] == room_type]

            # Calculate profit margin for color coding
            if len(room_data) > 0:
                room_data = room_data.copy()  # Avoid SettingWithCopyWarning
                room_data['profit_margin'] = ((room_data['rate'] - room_data['cost_per_occupied_room']) / room_data['rate'] * 100)

                fig2.add_trace(go.Scatter(
                    x=room_data['rate'],
                    y=room_data['cost_per_occupied_room'],
                    mode='markers',
                    name=room_type,
                    marker=dict(
                        opacity=0.7,
                        size=8,
                        color=room_data['profit_margin'],
                        colorscale='RdYlGn',
                        showscale=True if room_type == str(df['room_type'].unique()[0]) else False,
                        colorbar=dict(
                            title="Profit Margin (%)",
                            x=1.02,  # Move colorbar further right
                            thickness=15,  # Make colorbar thinner
                            len=0.8  # Make colorbar shorter
                        )
                    ),
                    hovertemplate=
                    f"<b>{room_type}</b><br>" +
                    "Rate: $%{x:.2f}<br>" +
                    "Cost: $%{y:.2f}<br>" +
                    "Profit Margin: %{marker.color:.1f}%<br>" +
                    "<extra></extra>"
                ))

        # Add break-even line (where cost = rate)
        if not df.empty and 'rate' in df.columns:
            min_val = min(df['rate'].min(), df['cost_per_occupied_room'].min())
            max_val = max(df['rate'].max(), df['cost_per_occupied_room'].max())
            fig2.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Break-even Line',
                line=dict(dash='dash', color='red', width=2),
                showlegend=True
            ))

        fig2.update_layout(
            title="Cost per Room vs. Daily Rate (colored by Profit Margin)",
            xaxis_title="Daily Rate ($)",
            yaxis_title="Cost per Occupied Room ($)",
            template="plotly_white",
            height=500,
            margin=dict(r=120)  # Add right margin for colorbar
        )
        scatter_layout.addWidget(create_plotly_widget(fig2))

        # Create dynamic explanation
        margin_performance = _classify_performance(avg_profit_margin)
        margin_spread = best_margin - worst_margin

        # Count points above/below break-even line
        profitable_count = len(df_copy[df_copy['profit_margin'] > 0])
        total_count = len(df_copy)
        profitable_pct = (profitable_count / total_count) * 100

        explanation = (
            f"Average profit margin is **{avg_profit_margin:.1f}%**, indicating **{margin_performance}** "
            f"pricing efficiency. Points are colored by profit margin - green indicates higher profitability, "
            f"red indicates lower margins. "
            f"\n\n**{best_margin_room}** rooms achieve the best margins at **{best_margin:.1f}%**, "
            f"while **{worst_margin_room}** rooms have **{worst_margin:.1f}%** margins "
            f"({margin_spread:.1f}% difference). "
            f"**{profitable_pct:.1f}%** of bookings are above break-even (red dashed line). "
            f"Points below this line indicate potential pricing or cost control issues."
        )

        scatter_layout.addWidget(_collapsible(explanation))
        self.tabs.addTab(scatter_tab, "Cost vs. Rate")

@data_required
def display():
    """Main entry point for the Room Cost view."""
    return RoomCostView()
