# flake8: noqa
# views/revenue.py
"""
Revenue Analysis View
====================

Shows key revenue metrics and interactive visualizations.
"""

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
from data.helpers import get_df
from PySide6.QtCore import Qt, QDate
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import numpy as np

# Import the data_required decorator from utils
from views.utils import data_required


def _plotly_view(fig):
    """Create a QWebEngineView widget from a Plotly figure."""
    from io import StringIO

    # Generate HTML for the plot
    html = StringIO()
    fig.write_html(html, include_plotlyjs='cdn', full_html=False)
    html = html.getvalue()

    # Create web view
    view = QWebEngineView()
    view.setHtml(html)
    view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return view


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
    """Classify performance based on profit margin or RevPAR uplift.

    Args:
        value: The profit margin or RevPAR uplift percentage

    Returns:
        Classification as "strong", "moderate", or "weak"
    """
    if value >= 30:
        return "strong"
    elif value >= 15:
        return "moderate"
    else:
        return "weak"


@data_required
def display() -> QWidget:
    """Display revenue analysis dashboard."""
    try:
        base_df = get_df()

        # Create UI structure
        root = QWidget()
        root.setLayout(QVBoxLayout())
        header = QLabel("Revenue Analysis")
        header.setStyleSheet("font-size:18pt;font-weight:bold;margin-bottom:15px;")
        root.layout().addWidget(header)

        # Date pickers
        start_picker = QDateEdit()
        end_picker = QDateEdit()
        for p in (start_picker, end_picker):
            p.setCalendarPopup(True)
            p.setStyleSheet("padding:5px;")
            p.setFixedWidth(120)

        # Set initial date range
        def refresh_date_pickers():
            try:
                if "date" in base_df.columns:
                    d0 = base_df["date"].min().date()
                    d1 = base_df["date"].max().date()
                    start_picker.setDate(QDate(d0.year, d0.month, d0.day))
                    end_picker.setDate(QDate(d1.year, d1.month, d1.day))
            except Exception as e:
                print(f"Error setting date range: {e}")

        refresh_date_pickers()  # initial sync

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Date Range:"))
        filter_row.addWidget(start_picker)
        filter_row.addWidget(QLabel(" to "))
        filter_row.addWidget(end_picker)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        filter_row.addWidget(apply_btn)
        filter_row.addStretch()
        root.layout().addLayout(filter_row)

        # Content area
        content = QTabWidget()
        content.setTabPosition(QTabWidget.North)
        content.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background: rgba(40, 40, 40, 0.7);  /* Transparent black */
                color: white;  /* White text for contrast */
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid #555555;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: rgba(0, 0, 0, 0.8);  /* Darker transparent black when selected */
                border-bottom: 2px solid #4a86e8;  /* Blue indicator at bottom */
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(60, 60, 60, 0.7);  /* Slightly lighter on hover */
            }
        """)
        root.layout().addWidget(content, 1)

        # Function to filter data by date range
        def filter_data(df, start_date, end_date):
            if "date" not in df.columns:
                return df

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])

            # Filter by date range
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            return df[mask]

        # Render function to update charts
        def render():
            try:
                # Clear previous content
                content.clear()

                # Get date range
                start_date = pd.Timestamp(start_picker.date().toPython())
                end_date = pd.Timestamp(end_picker.date().toPython())

                # Filter data
                df = filter_data(base_df, start_date, end_date)

                if df.empty:
                    empty_widget = QWidget()
                    empty_layout = QVBoxLayout(empty_widget)
                    empty_message = QLabel("No data available for the selected date range")
                    empty_message.setStyleSheet("font-size: 14pt; color: #aaaaaa;")
                    empty_layout.addWidget(empty_message, 0, Qt.AlignCenter)
                    content.addTab(empty_widget, "No Data")
                    return

                # Calculate KPIs (keep the calculations for use in explanations, but don't display)
                # For demo, assume we have 100 rooms
                room_count = 100

                # Create calculated columns if needed
                if "rate" in df.columns and "occupancy" in df.columns:
                    df["calculated_revenue"] = df["rate"] * df["occupancy"] * room_count
                    df["revpar"] = df["rate"] * df["occupancy"]

                # Get average values with checks for column existence
                if "occupancy" in df.columns:
                    # Convert to percentage
                    avg_occupancy = df["occupancy"].mean() * 100
                else:
                    avg_occupancy = 0

                avg_rate = df["rate"].mean() if "rate" in df.columns else 0

                if "revpar" in df.columns:
                    avg_revpar = df["revpar"].mean()
                elif "rate" in df.columns and "occupancy" in df.columns:
                    avg_revpar = (df["rate"] * df["occupancy"]).mean()
                else:
                    avg_revpar = 0

                # Calculate total revenue
                if "calculated_revenue" in df.columns:
                    total_revenue = df["calculated_revenue"].sum()
                else:
                    total_revenue = 0

                # Generate synthetic cost data if needed
                if "cost_per_occupied_room" not in df.columns and "rate" in df.columns:
                    # Base cost on 40% of room rate
                    df["cost_per_occupied_room"] = df["rate"] * 0.4

                if "cost_per_occupied_room" in df.columns and "occupancy" in df.columns:
                    df["cost"] = df["cost_per_occupied_room"] * df["occupancy"] * room_count
                    total_cost = df["cost"].sum()
                    profit = total_revenue - total_cost
                    profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
                else:
                    total_cost = 0
                    profit = 0
                    profit_margin = 0

                # ========================
                # Dashboard Tab
                # ========================
                dashboard_tab = QWidget()
                dashboard_tab.setLayout(QVBoxLayout())

                # Create main dashboard figure
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[
                        [{"type": "xy", "rowspan": 1}, {"type": "domain"}],
                        [{"type": "xy"}, {"type": "xy"}]
                    ],
                    subplot_titles=(
                        "Revenue & Cost Trend",
                        "Revenue by Day of Week",
                        "Occupancy & Rate Trend",
                        "Profit Margin Analysis"
                    ),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )

                # Chart 1: Revenue & Cost Trend (Top-left)
                daily_data = df.groupby("date").agg({
                    "calculated_revenue": "sum",
                    "cost": "sum" if "cost" in df.columns else None
                }).reset_index()

                fig.add_trace(
                    go.Scatter(
                        x=daily_data["date"],
                        y=daily_data["calculated_revenue"],
                        name="Revenue",
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )

                if "cost" in daily_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=daily_data["date"],
                            y=daily_data["cost"],
                            name="Cost",
                            mode="lines+markers",
                            line=dict(color="#ef4444", width=3),
                            marker=dict(size=8)
                        ),
                        row=1, col=1
                    )

                # Chart 2: Revenue by Day of Week (Top-right)
                df["weekday"] = df["date"].dt.day_name()
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_revenue = df.groupby("weekday")["calculated_revenue"].sum().reset_index()
                weekday_revenue["weekday"] = pd.Categorical(
                    weekday_revenue["weekday"],
                    categories=weekday_order,
                    ordered=True
                )
                weekday_revenue = weekday_revenue.sort_values("weekday")

                fig.add_trace(
                    go.Pie(
                        labels=weekday_revenue["weekday"],
                        values=weekday_revenue["calculated_revenue"],
                        name="Revenue by Weekday",
                        hole=0.4,
                        marker_colors=["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe", "#eff6ff", "#f0f9ff"],
                        textinfo="percent+label",
                        hoverinfo="label+value+percent",
                        textposition="inside"
                    ),
                    row=1, col=2
                )

                # Chart 3: Occupancy & Rate Trend (Bottom-left)
                daily_metrics = df.groupby("date").agg({
                    "occupancy": "mean",
                    "rate": "mean"
                }).reset_index()

                fig.add_trace(
                    go.Scatter(
                        x=daily_metrics["date"],
                        y=daily_metrics["occupancy"],
                        name="Occupancy",
                        mode="lines+markers",
                        line=dict(color="#10b981", width=3),
                        marker=dict(size=8),
                        yaxis="y3"
                    ),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=daily_metrics["date"],
                        y=daily_metrics["rate"],
                        name="Avg Rate",
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=3, dash="dash"),
                        marker=dict(size=8),
                        yaxis="y4"
                    ),
                    row=2, col=1
                )

                # Configure secondary axes
                fig.update_layout(
                    yaxis3=dict(title="Occupancy", showgrid=True),
                    yaxis4=dict(
                        title="Avg Rate ($)",
                        overlaying="y3",
                        side="right",
                        showgrid=False
                    )
                )

                # Chart 4: Profit Margin Analysis (Bottom-right)
                if "cost" in df.columns:
                    profit_data = df.groupby("date").agg({
                        "calculated_revenue": "sum",
                        "cost": "sum"
                    }).reset_index()
                    profit_data["profit"] = profit_data["calculated_revenue"] - profit_data["cost"]
                    profit_data["profit_margin"] = (profit_data["profit"] / profit_data["calculated_revenue"]) * 100

                    fig.add_trace(
                        go.Bar(
                            x=profit_data["date"],
                            y=profit_data["profit"],
                            name="Profit",
                            marker_color="#10b981",
                            opacity=0.7
                        ),
                        row=2, col=2
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=profit_data["date"],
                            y=profit_data["profit_margin"],
                            name="Profit Margin",
                            mode="lines+markers",
                            line=dict(color="#8b5cf6", width=3),
                            marker=dict(size=8),
                            yaxis="y5"
                        ),
                        row=2, col=2
                    )

                    # Configure secondary axis
                    fig.update_layout(
                        yaxis5=dict(
                            title="Profit Margin (%)",
                            overlaying="y",
                            side="right",
                            showgrid=False,
                            range=[0, 100]
                        )
                    )

                # Update overall layout
                fig.update_layout(
                    height=900,
                    title_font_size=20,
                    title_x=0.5,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified",
                    template="plotly_white",
                    font=dict(color='black'),  # Ensure text is black for readability
                    margin=dict(t=50, b=50, l=50, r=50)  # Reduced top margin since title is gone
                )

                dashboard_tab.layout().addWidget(_plotly_view(fig))

                # Create explanation text for dashboard
                performance_class = _classify_performance(profit_margin)

                # Find highest revenue day
                highest_revenue_day = weekday_revenue.iloc[weekday_revenue["calculated_revenue"].argmax()]["weekday"]

                dashboard_explanation = (
                    f"Overall profit margin is {profit_margin:.1f}%, which is considered **{performance_class}** "
                    f"potential. The top-left chart shows revenue vs cost trend; a widening gap signals healthier profitability. "
                    f"\n\nThe pie chart reveals {highest_revenue_day} generates the highest revenue at "
                    f"${weekday_revenue['calculated_revenue'].max():.0f}. "
                    f"Average daily rate is ${avg_rate:.2f} with {avg_occupancy:.1f}% occupancy, "
                    f"yielding RevPAR of ${avg_revpar:.2f}."
                )

                dashboard_tab.layout().addWidget(_collapsible(dashboard_explanation))
                content.addTab(dashboard_tab, "Dashboard")

                # ========================
                # Detailed Analysis Tab
                # ========================
                analysis_tab = QWidget()
                analysis_tab.setLayout(QVBoxLayout())

                # Create analysis figure
                analysis_fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Revenue Composition", "Performance Metrics"),
                    vertical_spacing=0.2
                )

                # Revenue Composition
                if "room_type" in df.columns:
                    room_revenue = df.groupby("room_type")["calculated_revenue"].sum().reset_index()

                    analysis_fig.add_trace(
                        go.Bar(
                            x=room_revenue["room_type"],
                            y=room_revenue["calculated_revenue"],
                            name="Revenue by Room Type",
                            marker_color="#3b82f6",
                            opacity=0.8
                        ),
                        row=1, col=1
                    )

                    # Add average rate to the same chart
                    room_rates = df.groupby("room_type")["rate"].mean().reset_index()

                    analysis_fig.add_trace(
                        go.Scatter(
                            x=room_rates["room_type"],
                            y=room_rates["rate"],
                            name="Avg Rate",
                            mode="lines+markers",
                            line=dict(color="#f59e0b", width=3),
                            marker=dict(size=10),
                            yaxis="y2"
                        ),
                        row=1, col=1
                    )

                    # Configure secondary axis
                    analysis_fig.update_layout(
                        yaxis2=dict(
                            title="Avg Rate ($)",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )

                # Performance Metrics
                metrics_data = df.groupby("date").agg({
                    "occupancy": "mean",
                    "rate": "mean",
                    "revpar": "mean"
                }).reset_index()

                analysis_fig.add_trace(
                    go.Scatter(
                        x=metrics_data["date"],
                        y=metrics_data["occupancy"],
                        name="Occupancy",
                        mode="lines",
                        line=dict(color="#10b981", width=3),
                        yaxis="y3"
                    ),
                    row=2, col=1
                )

                analysis_fig.add_trace(
                    go.Scatter(
                        x=metrics_data["date"],
                        y=metrics_data["rate"],
                        name="ADR",
                        mode="lines",
                        line=dict(color="#3b82f6", width=3),
                        yaxis="y4"
                    ),
                    row=2, col=1
                )

                analysis_fig.add_trace(
                    go.Scatter(
                        x=metrics_data["date"],
                        y=metrics_data["revpar"],
                        name="RevPAR",
                        mode="lines",
                        line=dict(color="#8b5cf6", width=3),
                        yaxis="y5"
                    ),
                    row=2, col=1
                )

                # Configure axes
                analysis_fig.update_layout(
                    yaxis3=dict(title="Occupancy", showgrid=True),
                    yaxis4=dict(
                        title="ADR ($)",
                        overlaying="y3",
                        side="right",
                        showgrid=False,
                        anchor="free",
                        position=0.85
                    ),
                    yaxis5=dict(
                        title="RevPAR ($)",
                        overlaying="y3",
                        side="right",
                        showgrid=False,
                        anchor="free",
                        position=0.95
                    )
                )

                # Update layout
                analysis_fig.update_layout(
                    height=800,
                    title_text="Detailed Revenue Analysis",
                    title_font_size=18,
                    title_x=0.5,
                    showlegend=True,
                    hovermode="x unified",
                    template="plotly_white"
                )

                analysis_tab.layout().addWidget(_plotly_view(analysis_fig))

                # Create explanation text for detailed analysis
                revpar_trend = "increasing" if metrics_data["revpar"].iloc[-1] > metrics_data["revpar"].iloc[0] else "decreasing"
                revpar_change = ((metrics_data["revpar"].iloc[-1] / metrics_data["revpar"].iloc[0]) - 1) * 100 if len(metrics_data) > 1 else 0
                revpar_performance = _classify_performance(abs(revpar_change))

                analysis_explanation = (
                    f"Detailed analysis shows a {revpar_performance} {revpar_trend} trend in RevPAR "
                    f"({revpar_change:.1f}% {revpar_trend}) over the period. "
                    f"RevPAR is calculated as Average Daily Rate (ADR) × Occupancy rate.\n\n"
                    f"Occupancy averages {avg_occupancy:.1f}% with ADR of ${avg_rate:.2f}. "
                )

                if "room_type" in df.columns:
                    top_room = room_revenue.iloc[room_revenue["calculated_revenue"].argmax()]["room_type"]
                    highest_rate = room_rates["rate"].max()
                    highest_rate_room = room_rates.iloc[room_rates["rate"].argmax()]["room_type"]

                    analysis_explanation += (
                        f"The '{top_room}' room type generates the most revenue, while '{highest_rate_room}' "
                        f"commands the highest rate at ${highest_rate:.2f}."
                    )

                analysis_tab.layout().addWidget(_collapsible(analysis_explanation))
                content.addTab(analysis_tab, "Detailed Analysis")

                # ========================
                # Forecast Tab
                # ========================
                forecast_tab = QWidget()
                forecast_tab.setLayout(QVBoxLayout())

                # Create forecast figure
                forecast_fig = go.Figure()

                # Generate forecast data (synthetic for demo)
                if len(daily_data) > 5:
                    # Simple linear forecast for demo
                    last_date = daily_data["date"].max()
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

                    # Create trend line
                    x = np.arange(len(daily_data))
                    y = daily_data["calculated_revenue"].values
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)

                    forecast_values = p(np.arange(len(daily_data), len(daily_data)+6))

                    # Add actual data
                    forecast_fig.add_trace(
                        go.Scatter(
                            x=daily_data["date"],
                            y=daily_data["calculated_revenue"],
                            name="Actual Revenue",
                            mode="lines+markers",
                            line=dict(color="#3b82f6", width=3)
                        )
                    )

                    # Add forecast
                    forecast_fig.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=forecast_values[-7:],
                            name="Forecast",
                            mode="lines+markers",
                            line=dict(color="#ef4444", width=3, dash="dot")
                        )
                    )

                    # Update layout
                    forecast_fig.update_layout(
                        height=500,
                        title="7-Day Revenue Forecast",
                        title_font_size=18,
                        title_x=0.5,
                        xaxis_title="Date",
                        yaxis_title="Revenue ($)",
                        template="plotly_white"
                    )

                    forecast_tab.layout().addWidget(_plotly_view(forecast_fig))

                    # Create explanation text for forecast
                    forecast_trend = "increasing" if forecast_values[-1] > forecast_values[0] else "decreasing"
                    forecast_change = ((forecast_values[-1] / daily_data["calculated_revenue"].iloc[-1]) - 1) * 100
                    trend_strength = _classify_performance(abs(forecast_change))

                    # Calculate 7-day total forecast
                    total_forecast = sum(forecast_values[-7:])
                    # forecast_margin = profit_margin  # Assuming same profit margin
                    forecast_profit = total_forecast * (profit_margin / 100)

                    forecast_explanation = (
                        f"The 7-day forecast shows a {trend_strength} {forecast_trend} trend with "
                        f"a projected {forecast_change:.1f}% {forecast_trend} in daily revenue. "
                        f"Total forecasted revenue is ${total_forecast:,.0f} with projected profit "
                        f"of ${forecast_profit:,.0f} (assuming consistent {profit_margin:.1f}% profit margin).\n\n"
                        f"This trend analysis is based on historical performance and may be influenced by "
                        f"seasonality, events, or other external factors not captured in the model."
                    )

                    forecast_tab.layout().addWidget(_collapsible(forecast_explanation))
                    content.addTab(forecast_tab, "Forecast")

            except Exception as e:
                import traceback
                traceback.print_exc()

                error_widget = QWidget()
                error_layout = QVBoxLayout(error_widget)
                error_label = QLabel(f"Error: {str(e)}")
                error_label.setStyleSheet("color: red;")
                error_layout.addWidget(error_label)
                content.addTab(error_widget, "Error")

        # Connect Apply button
        apply_btn.clicked.connect(render)

        # Initial render
        render()

        # Expose refresh_date_pickers for data sync
        display.refresh_date_pickers = refresh_date_pickers

        return root
    except Exception as e:
        # Catch any errors and return a widget with an error message
        import traceback
        traceback.print_exc()

        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_label = QLabel(f"Error loading Revenue View: {str(e)}")
        error_label.setStyleSheet("color: red; font-size: 14pt;")
        error_layout.addWidget(error_label, alignment=Qt.AlignCenter)

        return error_widget


# Add this debugging code to identify the issue
def debug_view_loading():
    try:
        # Replace this with your actual view loading code
        revenue_view_function = display  # Reference to the display function
        if revenue_view_function is None:
            print("ERROR: Revenue view function is None!")
            return
        revenue_view_function()
    except Exception as e:
        print(f"Error loading revenue view: {e}")
        import traceback
        traceback.print_exc()
