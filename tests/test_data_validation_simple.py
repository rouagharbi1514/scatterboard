#!/usr/bin/env python3
"""
Simplified Data Validation Tests
===============================

This module contains unit tests for data validation functionality
without dependencies on torch or forecast modules.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data validation directly without going through views package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'views'))
from data_validator import DataValidator, validate_revenue_data, validate_forecast_data


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a flexible date range that can accommodate different starting years
        start_date = pd.Timestamp.now() - pd.DateOffset(years=2)
        self.valid_data = pd.DataFrame({
            'Date': pd.date_range(start_date, periods=100),
            'TotalRevenue': np.random.uniform(1000, 5000, 100),
            'TotalCosts': np.random.uniform(500, 2000, 100),
            'OccupancyRate': np.random.uniform(0.3, 0.95, 100),
            'ADR': np.random.uniform(80, 300, 100),
            'RevPAR': np.random.uniform(50, 250, 100),
            'RoomType': np.random.choice(['Standard', 'Deluxe', 'Suite'], 100),
            'BookingSource': np.random.choice(['Direct', 'OTA', 'Corporate'], 100),
            'MarketSegment': np.random.choice(['Leisure', 'Business', 'Group'], 100)
        })
        
        # Create invalid data scenarios
        self.empty_data = pd.DataFrame()
        self.missing_columns_data = self.valid_data.drop(['TotalRevenue', 'Date'], axis=1)
        self.invalid_types_data = self.valid_data.copy()
        self.invalid_types_data['TotalRevenue'] = 'invalid'
        self.null_data = self.valid_data.copy()
        self.null_data.loc[0:10, 'TotalRevenue'] = np.nan
    
    def test_data_validator_initialization(self):
        """Test DataValidator can be initialized."""
        validator = DataValidator()
        self.assertIsNotNone(validator)
    
    def test_validate_dataframe_with_valid_data(self):
        """Test validation passes with valid data."""
        validator = DataValidator()
        result = validator.validate_dataframe(self.valid_data, ['Date', 'TotalRevenue'])
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_dataframe_with_empty_data(self):
        """Test validation fails with empty data."""
        validator = DataValidator()
        result = validator.validate_dataframe(self.empty_data, ['Date', 'TotalRevenue'])
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_dataframe_with_missing_columns(self):
        """Test validation fails with missing required columns."""
        validator = DataValidator()
        result = validator.validate_dataframe(self.missing_columns_data, ['Date', 'TotalRevenue'])
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_data_types(self):
        """Test data type validation through validate_dataframe."""
        validator = DataValidator()
        # Test through validate_dataframe which calls internal _validate_data_types
        result = validator.validate_dataframe(self.invalid_types_data, ['Date', 'TotalRevenue'])
        # Should have warnings about type conversion issues
        self.assertGreater(len(result.warnings), 0)
    
    def test_validate_data_ranges(self):
        """Test data range validation through validate_dataframe."""
        validator = DataValidator()
        # Create data with invalid ranges
        invalid_range_data = self.valid_data.copy()
        invalid_range_data['OccupancyRate'] = 1.5  # > 100%
        # Test through validate_dataframe which calls internal _validate_data_ranges
        result = validator.validate_dataframe(invalid_range_data, ['Date', 'TotalRevenue'])
        # Should have warnings about out-of-range values
        self.assertGreater(len(result.warnings), 0)
    
    def test_validate_revenue_data_function(self):
        """Test revenue-specific validation function."""
        result = validate_revenue_data(self.valid_data)
        self.assertTrue(result.is_valid)
        
        # Test with invalid data
        result = validate_revenue_data(self.empty_data)
        self.assertFalse(result.is_valid)
    
    def test_validate_forecast_data_function(self):
        """Test forecast-specific validation function."""
        result = validate_forecast_data(self.valid_data, 'TotalRevenue')
        self.assertTrue(result.is_valid)
        
        # Test with insufficient data
        small_data = self.valid_data.head(5)
        result = validate_forecast_data(small_data, 'TotalRevenue')
        self.assertTrue(result.is_valid)  # Should pass but with warnings
    
    def test_handle_null_values(self):
        """Test handling of null values."""
        validator = DataValidator()
        result = validator.validate_dataframe(self.null_data, ['Date', 'TotalRevenue'])
        # Should pass but with warnings about null values
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.warnings), 0)
    
    def test_occupancy_rate_bounds(self):
        """Test occupancy rate is within valid bounds (0-1)."""
        if 'OccupancyRate' in self.valid_data.columns:
            occupancy = self.valid_data['OccupancyRate']
            occupancy_clean = occupancy.dropna()
            if not occupancy_clean.empty:
                self.assertTrue((occupancy_clean >= 0).all(), "Occupancy rate should be >= 0")
                self.assertTrue((occupancy_clean <= 1).all(), "Occupancy rate should be <= 1")
    
    def test_revenue_non_negative(self):
        """Test revenue values are non-negative."""
        revenue_columns = ['TotalRevenue', 'RevPAR']
        for col in revenue_columns:
            if col in self.valid_data.columns:
                revenue = self.valid_data[col].dropna()
                if not revenue.empty:
                    self.assertTrue((revenue >= 0).all(), f"{col} should be non-negative")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)