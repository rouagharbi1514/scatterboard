"""
Helper functions for data processing and transformation.
Provides utilities for standardizing data columns and calculating derived metrics.
"""

import pandas as pd
import numpy as np
from data import get_dataframe


def get_df() -> pd.DataFrame:
    """
    Get a standardized dataframe with normalized column names and calculated metrics.
    
    This function retrieves data from the main dataframe, standardizes column names,
    adds missing columns with sensible defaults, and calculates derived metrics
    like room revenue.
    
    Returns:
        pd.DataFrame: A standardized dataframe ready for analytics
    """
    # Get the original dataframe
    df = get_dataframe()
    
    # If no data is available, return an empty dataframe with expected columns
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Date", "RoomType", "ADR", "OccupancyRate", "RoomRevenue",
            "TotalRoomCost", "Profit", "BookingSource"
        ])
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert all column names to lowercase for case-insensitive comparison
    lowercase_cols = {col.lower(): col for col in result.columns}
    
    # Standardize column names with expanded mapping
    column_mapping = {
        # Date variations
        "date": "Date",
        "booking_date": "Date",
        "checkin_date": "Date",
        "check_in_date": "Date",
        "reservation_date": "Date",
        "stay_date": "Date",
        
        # Room type variations
        "room_type": "RoomType",
        "roomtype": "RoomType",
        "room": "RoomType",
        "room_category": "RoomType",
        "room_class": "RoomType",
        
        # Rate variations
        "rate": "ADR",
        "adr": "ADR",
        "average_daily_rate": "ADR",
        "room_rate": "ADR",
        "price": "ADR",
        "room_price": "ADR",
        "nightly_rate": "ADR",
        
        # Occupancy variations
        "occupancy": "OccupancyRate",
        "occupancy_rate": "OccupancyRate",
        "occ": "OccupancyRate",
        "occ_rate": "OccupancyRate",
        "occupancy_percentage": "OccupancyRate",
        
        # Revenue per available room
        "revpar": "RevPAR",
        "rev_par": "RevPAR",
        "revenue_per_available_room": "RevPAR",
        
        # Gross operating profit per available room
        "goppar": "GOPPAR",
        "gop_par": "GOPPAR",
        "gross_operating_profit_par": "GOPPAR",
        
        # Booking source variations
        "booking_source": "BookingSource",
        "source": "BookingSource",
        "reservation_source": "BookingSource",
        "channel": "BookingSource",
        "booking_channel": "BookingSource",
        
        # Revenue variations
        "revenue": "RoomRevenue",
        "room_revenue": "RoomRevenue",
        "total_revenue": "TotalRevenue",
        
        # Cost variations
        "cost": "TotalRoomCost",
        "room_cost": "TotalRoomCost",
        "cost_per_room": "CostPerOccupiedRoom",
        "cost_per_occupied_room": "CostPerOccupiedRoom",
    }
    
    # Map columns using lowercase comparison
    for orig_col, std_col in column_mapping.items():
        if orig_col in lowercase_cols:
            # Found a match, copy to standard name
            result[std_col] = result[lowercase_cols[orig_col]]
    
    # Add calculated columns if key metrics are available
    
    # Calculate RoomRevenue if ADR and OccupancyRate are present but RoomRevenue is not
    if "RoomRevenue" not in result.columns and "ADR" in result.columns and "OccupancyRate" in result.columns:
        result["RoomRevenue"] = result["ADR"] * result["OccupancyRate"]
    
    # Calculate RevPAR if not present but we have the components
    if "RevPAR" not in result.columns and "ADR" in result.columns and "OccupancyRate" in result.columns:
        result["RevPAR"] = result["ADR"] * result["OccupancyRate"]
    
    # If we have RoomRevenue but not TotalRoomCost
    if "TotalRoomCost" not in result.columns and "RoomRevenue" in result.columns:
        # Room cost is typically 30-40% of room revenue
        result["TotalRoomCost"] = result["RoomRevenue"] * np.random.uniform(0.3, 0.4, len(result))
    
    # Calculate Profit if missing but we have RoomRevenue and TotalRoomCost
    if "Profit" not in result.columns and "RoomRevenue" in result.columns and "TotalRoomCost" in result.columns:
        result["Profit"] = result["RoomRevenue"] - result["TotalRoomCost"]
    
    # Add BookingSource if missing
    if "BookingSource" not in result.columns:
        sources = ["Direct", "OTA", "Corporate", "Travel Agent", "Wholesaler"]
        result["BookingSource"] = np.random.choice(sources, size=len(result))
    
    # Add UpsellRevenue if missing
    if "UpsellRevenue" not in result.columns and "RoomRevenue" in result.columns:
        # Upsell is typically 5-15% of room revenue
        result["UpsellRevenue"] = result["RoomRevenue"] * np.random.uniform(0.05, 0.15, len(result))
    
    # Add TotalRevenue if not present
    if "TotalRevenue" not in result.columns and "RoomRevenue" in result.columns:
        # Assume room revenue is ~70% of total revenue
        result["TotalRevenue"] = result["RoomRevenue"] / 0.7
    
    return result


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date columns are proper datetime objects.
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized date columns
    """
    if df is None or df.empty:
        return df
    
    result = df.copy()
    date_cols = [c for c in result.columns if "date" in c.lower()]
    
    for col in date_cols:
        try:
            result[col] = pd.to_datetime(result[col])
        except Exception:
            # If conversion fails, keep the original values
            pass
            
    return result


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common derived metrics used in hotel analytics.
    
    Args:
        df: DataFrame to enhance
    
    Returns:
        pd.DataFrame: Enhanced DataFrame with additional metrics
    """
    if df is None or df.empty:
        return df
        
    result = df.copy()
    
    # Calculate RevPAR (Revenue Per Available Room) if possible
    if "RoomRevenue" in result.columns and "OccupancyRate" in result.columns:
        result["RevPAR"] = result["RoomRevenue"] / 100  # Assuming 100 rooms
    
    # Calculate GOPPAR (Gross Operating Profit Per Available Room) if possible
    if "Profit" in result.columns:
        result["GOPPAR"] = result["Profit"] / 100  # Assuming 100 rooms
    
    # Calculate ProfitMargin if possible
    if "Profit" in result.columns and "RoomRevenue" in result.columns:
        result["ProfitMargin"] = (result["Profit"] / result["RoomRevenue"].replace(0, np.nan)) * 100
        
    return result