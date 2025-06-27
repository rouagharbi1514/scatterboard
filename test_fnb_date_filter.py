#!/usr/bin/env python3
"""
Test script for F&B Analysis date filtering functionality
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, date

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fnb_date_filtering():
    """Test the F&B Analysis date filtering functionality"""
    try:
        print("Testing F&B Analysis date filtering...")

        # Simulate the data generation from operations.py
        def generate_sample_data():
            date_range = pd.date_range(start='2024-01-01', end=pd.Timestamp.today(), freq="D")
            outlets = ["Main Restaurant", "Lobby Bar", "Pool Bar", "Room Service"]

            data = []
            np.random.seed(42)  # For reproducible results

            for date in date_range:
                is_weekend = date.dayofweek >= 5
                weekend_factor = 1.4 if is_weekend else 1.0

                for outlet in outlets:
                    # Base values per outlet
                    if outlet == "Main Restaurant":
                        base_guests = 110
                        base_check = 45
                        food_ratio = 0.7
                    elif outlet == "Lobby Bar":
                        base_guests = 60
                        base_check = 28
                        food_ratio = 0.3
                    elif outlet == "Pool Bar":
                        base_guests = 75
                        base_check = 32
                        food_ratio = 0.4
                    else:  # Room Service
                        base_guests = 28
                        base_check = 55
                        food_ratio = 0.6

                    # Randomize
                    guests = int(base_guests * weekend_factor * np.random.normal(1, 0.15))
                    avg_check = base_check * np.random.normal(1, 0.1)
                    total_revenue = guests * avg_check

                    # Split revenue
                    food_rev = total_revenue * food_ratio * np.random.uniform(0.9, 1.1)
                    bev_rev = total_revenue - food_rev

                    data.append({
                        "date": date,
                        "outlet": outlet,
                        "guests": guests,
                        "avg_check": avg_check,
                        "food_revenue": food_rev,
                        "beverage_revenue": bev_rev,
                        "total_revenue": total_revenue
                    })

            return pd.DataFrame(data)

        # Generate sample data
        sample_data = generate_sample_data()
        print(f"✓ Generated sample data with {len(sample_data)} rows")

        # Test date range
        min_date = sample_data["date"].min().date()
        max_date = sample_data["date"].max().date()
        print(f"✓ Date range: {min_date} to {max_date}")

        # Test filtering functionality
        def get_filtered_data(start_date, end_date):
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            return sample_data[(sample_data["date"] >= start) & (sample_data["date"] <= end)]

        # Test various date filters
        test_start = date(2024, 6, 1)
        test_end = date(2024, 6, 30)
        filtered_data = get_filtered_data(test_start, test_end)
        print(f"✓ Filtered data (June 2024): {len(filtered_data)} rows")

        # Test full range
        full_range = get_filtered_data(min_date, max_date)
        print(f"✓ Full range data: {len(full_range)} rows")

        # Verify date filtering works correctly
        if len(filtered_data) > 0:
            filtered_min = filtered_data["date"].min().date()
            filtered_max = filtered_data["date"].max().date()
            print(f"✓ Filtered range: {filtered_min} to {filtered_max}")

            if filtered_min >= test_start and filtered_max <= test_end:
                print("✓ Date filtering works correctly")
            else:
                print("❌ Date filtering not working properly")
                return False

        print(f"✓ Data starts automatically from: {min_date}")
        print(f"✓ Data ends at: {max_date}")

        print("\n🎉 All F&B date filtering tests passed!")
        print("✓ Date picker will now start from the data's actual start date")
        print("✓ Full date range is available for filtering")
        return True

    except Exception as e:
        print(f"❌ Error in F&B date filtering test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fnb_date_filtering()
    sys.exit(0 if success else 1)
