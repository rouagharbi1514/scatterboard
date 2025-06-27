"""
Defines the navigation routes grouped into categories for the sidebar.
"""

from views import (
    file_upload_display,
    overview_display as dashboard_display,
    kpis_display as kpi_display,
    revenue_display,
    seasonality_display,
    room_cost_display,
    display_room_profit as profitability_room_display,
    display_room_type_profitability as profitability_type_display,
    placeholder_display,
    marketing_display,
    operations_fb_display,
    operations_efficiency_display,
    operations_custom_charts_display,
    housekeeping_display,
)
from views.what_if import display as what_if_display
from views.what_if_turbo_simple import display as what_if_turbo_display

# Group routes by category for the sidebar
ROUTES_GROUPED = {
    "Overview": {
        "Dashboard Overview": dashboard_display,
    },
    "Performance": {
        "Key Performance Indicators": kpi_display,
        "Revenue Analysis": revenue_display,
        "Seasonality Analysis": seasonality_display,
        "Room Cost Analysis": room_cost_display,
        "Profitability by Room": profitability_room_display,
        "Profitability by Room Type": profitability_type_display,
    },
    "Guest Analysis": {
        "Guest Overview": placeholder_display,
    },
    "Marketing": {
        "Marketing Performance": marketing_display,
    },
    "Operations": {
        "Housekeeping": housekeeping_display,
        "F&B Analysis": operations_fb_display,
        "Operational Efficiency": operations_efficiency_display,
        "Custom Charts": operations_custom_charts_display,
    },
    "Advanced": {
        "ML Pricing Optimization": placeholder_display,
        "ML Guest Segmentation": placeholder_display,
        "Scenario Planning": placeholder_display,
        "Cancellation Prediction": placeholder_display,
        "Data Storytelling": placeholder_display,
        "Advanced Analytics": placeholder_display,
        "Dig Deeper": placeholder_display,
    },
    "Data": {
        "Upload Data": file_upload_display,
        "Give Feedback": placeholder_display,
    },
    "What If": {
        "What If Analysis": what_if_display,
        "What-If Turbo": what_if_turbo_display,
    },
}
