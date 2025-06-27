import pandas as pd
from unittest import mock
from views.data_helper import get_df


def test_get_df_empty():
    """Test that get_df returns an empty dataframe with expected columns when source is empty"""
    with mock.patch("data.get_dataframe", return_value=None):
        result = get_df()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # Check that expected columns are present
        expected_columns = [
            "Date",
            "RoomType",
            "ADR",
            "OccupancyRate",
            "RoomRevenue",
            "TotalRoomCost",
            "Profit",
            "BookingSource",
        ]
        for col in expected_columns:
            assert col in result.columns


def test_get_df_column_renaming():
    """Test that get_df correctly renames lowercase columns to proper case"""
    # Create test data with lowercase column names
    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3),
            "room_type": ["Standard", "Deluxe", "Suite"],
        }
    )

    with mock.patch("data.get_dataframe", return_value=test_df):
        result = get_df()
        # Check that columns were renamed with capital letters
        assert "Date" in result.columns
        assert "RoomType" in result.columns
        # Original data should be preserved
        assert len(result) == 3
        assert list(result["RoomType"]) == ["Standard", "Deluxe", "Suite"]


def test_get_df_revenue_calculation():
    """Test that get_df calculates RoomRevenue when ADR and OccupancyRate are present"""
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=2),
            "ADR": [100, 200],
            "OccupancyRate": [0.5, 0.75],
        }
    )

    with mock.patch("data.get_dataframe", return_value=test_df):
        result = get_df()
        # Check that RoomRevenue was calculated correctly
        assert "RoomRevenue" in result.columns
        # RoomRevenue = ADR * OccupancyRate * 100 (rooms)
        assert result["RoomRevenue"][0] == 100 * 0.5 * 100
        assert result["RoomRevenue"][1] == 200 * 0.75 * 100


def test_get_df_booking_source_added():
    """Test that get_df adds BookingSource when missing"""
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=5),
        }
    )

    with mock.patch("data.get_dataframe", return_value=test_df):
        result = get_df()
        # Check that BookingSource was added
        assert "BookingSource" in result.columns
        # All entries should have a booking source
        assert result["BookingSource"].notna().all()
        # Should have expected values
        for source in result["BookingSource"]:
            assert source in ["Direct", "OTA", "Travel Agent", "Corporate", "Wholesale"]


def test_get_df_upsell_revenue():
    """Test that get_df adds UpsellRevenue when missing but RoomRevenue is present"""
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=3),
            "RoomRevenue": [5000, 6000, 7000],
        }
    )

    with mock.patch("data.get_dataframe", return_value=test_df):
        result = get_df()
        # Check that UpsellRevenue was added
        assert "UpsellRevenue" in result.columns
        # UpsellRevenue should be between 10-20% of room revenue
        for i in range(len(result)):
            assert (
                0.1 * result["RoomRevenue"][i]
                <= result["UpsellRevenue"][i]
                <= 0.2 * result["RoomRevenue"][i]
            )
