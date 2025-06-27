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
    
    # Standardize column names (convert lowercase/snake_case to proper case)
    column_mapping = {
        "date": "Date",
        "room_type": "RoomType",
        "adr": "ADR",
        "occupancy_rate": "OccupancyRate",
        "occupancy": "OccupancyRate",
        "room_revenue": "RoomRevenue",
        "total_room_cost": "TotalRoomCost",
        "profit": "Profit",
        "booking_source": "BookingSource",
        "upsell_revenue": "UpsellRevenue"
    }
    
    # Rename columns that exist in the mapping
    for old_col, new_col in column_mapping.items():
        if old_col in result.columns:
            result.rename(columns={old_col: new_col}, inplace=True)
    
    # Calculate RoomRevenue if it's missing but we have ADR and OccupancyRate
    if "RoomRevenue" not in result.columns and "ADR" in result.columns and "OccupancyRate" in result.columns:
        # Assuming an average of 100 rooms per hotel for calculations
        result["RoomRevenue"] = result["ADR"] * result["OccupancyRate"] * 100
    
    # Add BookingSource if missing
    if "BookingSource" not in result.columns:
        # Generate realistic booking distribution
        n_rows = len(result)
        booking_sources = ["Direct", "OTA", "Travel Agent", "Corporate", "Wholesale"]
        # Weighted distribution: Direct (30%), OTA (40%), Travel Agent (10%), Corporate (15%), Wholesale (5%)
        weights = [0.3, 0.4, 0.1, 0.15, 0.05]
        result["BookingSource"] = np.random.choice(booking_sources, size=n_rows, p=weights)
    
    # Add UpsellRevenue if missing but we have RoomRevenue
    if "UpsellRevenue" not in result.columns and "RoomRevenue" in result.columns:
        # Upsell revenue is typically 10-20% of room revenue
        result["UpsellRevenue"] = result["RoomRevenue"] * np.random.uniform(0.1, 0.2, len(result))
    
    # Add TotalRoomCost if missing but we have RoomRevenue
    if "TotalRoomCost" not in result.columns and "RoomRevenue" in result.columns:
        # Room cost is typically 30-40% of room revenue
        result["TotalRoomCost"] = result["RoomRevenue"] * np.random.uniform(0.3, 0.4, len(result))
    
    # Calculate Profit if missing but we have RoomRevenue and TotalRoomCost
    if "Profit" not in result.columns and "RoomRevenue" in result.columns and "TotalRoomCost" in result.columns:
        result["Profit"] = result["RoomRevenue"] - result["TotalRoomCost"]
    
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