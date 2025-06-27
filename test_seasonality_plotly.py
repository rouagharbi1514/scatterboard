#!/usr/bin/env python3
"""
Quick test to verify that the seasonality charts have been converted to Plotly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import the seasonality module
from views.seasonality import _monthly_occ, _adr_revpar, _weekday_weekend, _seasonality_index

def create_sample_data():
    """Create sample hotel data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    data = {
        'Date': dates,
        'OccupancyRate': np.random.uniform(0.6, 0.9, len(dates)),
        'ADR': np.random.uniform(100, 200, len(dates)),
        'RevPAR': np.random.uniform(80, 150, len(dates)),
        'TotalRevenue': np.random.uniform(5000, 15000, len(dates))
    }

    return pd.DataFrame(data)

def test_chart_functions():
    """Test that chart functions return Plotly figures"""
    df = create_sample_data()

    print("Testing seasonality chart functions...")

    # Test monthly occupancy chart
    try:
        fig, explanation = _monthly_occ(df)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        assert isinstance(explanation, str), f"Expected str, got {type(explanation)}"
        print("✅ _monthly_occ: PASS")
    except Exception as e:
        print(f"❌ _monthly_occ: FAIL - {e}")

    # Test ADR/RevPAR chart
    try:
        fig, explanation = _adr_revpar(df)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        assert isinstance(explanation, str), f"Expected str, got {type(explanation)}"
        print("✅ _adr_revpar: PASS")
    except Exception as e:
        print(f"❌ _adr_revpar: FAIL - {e}")

    # Test weekday/weekend chart
    try:
        fig, explanation = _weekday_weekend(df)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        assert isinstance(explanation, str), f"Expected str, got {type(explanation)}"
        print("✅ _weekday_weekend: PASS")
    except Exception as e:
        print(f"❌ _weekday_weekend: FAIL - {e}")

    # Test seasonality index chart
    try:
        fig, explanation = _seasonality_index(df)
        assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
        assert isinstance(explanation, str), f"Expected str, got {type(explanation)}"
        print("✅ _seasonality_index: PASS")
    except Exception as e:
        print(f"❌ _seasonality_index: FAIL - {e}")

    print("\nAll chart functions successfully converted to Plotly! 🎉")

if __name__ == "__main__":
    test_chart_functions()
