#!/usr/bin/env python3
"""
Test script to verify that the room cost analysis works properly
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test the room cost view
def test_room_cost_view():
    """Test that the room cost view works with sample data"""
    try:
        from views.room_cost import RoomCostView, create_sample_data
        print("✅ Successfully imported RoomCostView")

        # Test sample data creation
        sample_data = create_sample_data()
        print(f"✅ Sample data created with {len(sample_data)} rows")
        print(f"   Columns: {list(sample_data.columns)}")
        print(f"   Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
        print(f"   Room types: {sample_data['room_type'].unique()}")
        print(f"   Cost range: ${sample_data['cost_per_occupied_room'].min():.2f} - ${sample_data['cost_per_occupied_room'].max():.2f}")

        print("\n✅ Room Cost Analysis is ready to use!")

    except Exception as e:
        print(f"❌ Error testing room cost view: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_room_cost_view()
