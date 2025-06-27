# data/scenario_calculator.py
"""
Scenario Calculator Module
==========================

NumPy-based calculations for what-if scenario analysis.
Computes 30-day P&L projections based on input parameters.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Any


def simulate_30day_pl(
    inputs: Dict[str, Any],
    baseline_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Simulate 30-day P&L based on scenario inputs and baseline data.

    Args:
        inputs: Dictionary containing scenario parameters:
            - room_rate: float, new room rate
            - occupancy_pct: float, target occupancy percentage (0-100)
            - housekeeping_staff: int, number of housekeeping staff
            - fb_staff: int, number of F&B staff
            - promotion_active: bool, whether promotion is active
        baseline_data: DataFrame with historical hotel data

    Returns:
        Dictionary with calculated KPIs:
            - revpar: Revenue per available room
            - goppar: Gross operating profit per available room
            - profit_delta: Change in profit vs baseline
            - total_revenue: Total projected revenue
            - total_costs: Total projected costs
            - occupancy_rooms: Number of occupied rooms
    """
    if baseline_data.empty:
        return _default_kpis()

    # Extract inputs with defaults
    room_rate = inputs.get('room_rate', 150.0)
    occupancy_pct = inputs.get('occupancy_pct', 75.0) / 100.0
    housekeeping_staff = inputs.get('housekeeping_staff', 10)
    fb_staff = inputs.get('fb_staff', 8)
    promotion_active = inputs.get('promotion_active', False)

    # Calculate baseline metrics
    baseline_avg_rate = baseline_data.get('rate', pd.Series([150])).mean()
    baseline_occupancy = baseline_data.get('occupancy', pd.Series([0.75])).mean()

    # Assume 100 rooms for calculations (adjustable based on data)
    total_rooms = 100
    days = 30

    # Calculate scenario metrics
    occupied_rooms = total_rooms * occupancy_pct

    # Room revenue calculation
    room_revenue = room_rate * occupied_rooms * days

    # Apply promotion discount if active
    if promotion_active:
        room_revenue *= 0.9  # 10% discount

    # F&B revenue (estimated as 30% of room revenue)
    fb_revenue = room_revenue * 0.3

    # Total revenue
    total_revenue = room_revenue + fb_revenue

    # Cost calculations
    # Housekeeping: $80/day per staff
    housekeeping_cost = housekeeping_staff * 80 * days

    # F&B staff: $90/day per staff
    fb_cost = fb_staff * 90 * days

    # Variable costs (20% of revenue)
    variable_costs = total_revenue * 0.2

    # Fixed costs (utilities, admin, etc.)
    fixed_costs = 50000  # Monthly fixed costs

    total_costs = housekeeping_cost + fb_cost + variable_costs + fixed_costs

    # Calculate KPIs
    revpar = room_revenue / (total_rooms * days)
    gross_profit = total_revenue - total_costs
    goppar = gross_profit / (total_rooms * days)

    # Calculate baseline for comparison
    baseline_room_revenue = baseline_avg_rate * (
        total_rooms * baseline_occupancy * days
    )
    baseline_fb_revenue = baseline_room_revenue * 0.3
    baseline_total_revenue = baseline_room_revenue + baseline_fb_revenue
    baseline_total_costs = (
        10 * 80 * days +  # Default housekeeping
        8 * 90 * days +   # Default F&B
        baseline_total_revenue * 0.2 +  # Variable costs
        50000  # Fixed costs
    )
    baseline_profit = baseline_total_revenue - baseline_total_costs
    profit_delta = gross_profit - baseline_profit

    return {
        'revpar': round(revpar, 2),
        'goppar': round(goppar, 2),
        'profit_delta': round(profit_delta, 2),
        'total_revenue': round(total_revenue, 2),
        'total_costs': round(total_costs, 2),
        'occupancy_rooms': round(occupied_rooms, 1),
        'gross_profit': round(gross_profit, 2),
    }


def _default_kpis() -> Dict[str, float]:
    """Return default KPIs when no baseline data is available."""
    return {
        'revpar': 0.0,
        'goppar': 0.0,
        'profit_delta': 0.0,
        'total_revenue': 0.0,
        'total_costs': 0.0,
        'occupancy_rooms': 0.0,
        'gross_profit': 0.0,
    }


def calculate_waterfall_data(
    baseline_kpis: Dict[str, float],
    scenario_kpis: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate waterfall chart data showing profit drivers.

    Args:
        baseline_kpis: Baseline scenario KPIs
        scenario_kpis: New scenario KPIs

    Returns:
        Dictionary with waterfall chart data for Plotly
    """
    # Calculate individual impact components
    revenue_impact = (
        scenario_kpis['total_revenue'] - baseline_kpis['total_revenue']
    )
    cost_impact = (
        scenario_kpis['total_costs'] - baseline_kpis['total_costs']
    )
    net_impact = revenue_impact - cost_impact

    # Waterfall data
    categories = [
        'Baseline Profit',
        'Revenue Change',
        'Cost Change',
        'Scenario Profit'
    ]

    values = [
        baseline_kpis.get('gross_profit', 0),
        revenue_impact,
        -cost_impact,  # Negative because cost increase reduces profit
        scenario_kpis.get('gross_profit', 0),
    ]

    # Determine colors (green for positive, red for negative)
    colors = []
    for i, val in enumerate(values):
        if i == 0 or i == len(values) - 1:  # First and last are totals
            colors.append('blue')
        elif val >= 0:
            colors.append('green')
        else:
            colors.append('red')

    return {
        'categories': categories,
        'values': values,
        'colors': colors,
        'net_change': net_impact,
    }
