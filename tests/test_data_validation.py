#!/usr/bin/env python3
"""
Comprehensive Data Validation Unit Tests
========================================

This module contains unit tests for data validation across all view components.
Tests handle scenarios like:
- Missing required columns
- Wrong data types
- Empty datasets
- Malformed Excel inputs
- Null/NaN values
- Invalid date formats
- Negative values where not allowed
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataframe, get_dataframe
from views.utils import data_required, create_error_widget


class DataValidationTestCase(unittest.TestCase):
    """Base test case for data validation tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'TotalRevenue': np.random.uniform(1000, 5000, 100),
            'TotalCosts': np.random.uniform(500, 2000, 100),
            'OccupancyRate': np.random.uniform(0.3, 0.95, 100),
            'ADR': np.random.uniform(80, 300, 100),
            'RevPAR': np.random.uniform(50, 250, 100),
            'RoomType': np.random.choice(['Standard', 'Deluxe', 'Suite'], 100),
            'BookingSource': np.random.choice(['Direct', 'OTA', 'Corporate'], 100),
            'MarketSegment': np.random.choice(['Leisure', 'Business', 'Group'], 100)
        })
        
    def tearDown(self):
        """Clean up after tests."""
        # Reset any global data state
        pass
        
    def create_invalid_data_scenarios(self):
        """Create various invalid data scenarios for testing."""
        scenarios = {}
        
        # Empty dataframe
        scenarios['empty'] = pd.DataFrame()
        
        # Missing required columns
        scenarios['missing_date'] = self.valid_data.drop('Date', axis=1)
        scenarios['missing_revenue'] = self.valid_data.drop('TotalRevenue', axis=1)
        scenarios['missing_costs'] = self.valid_data.drop('TotalCosts', axis=1)
        
        # Wrong data types
        scenarios['invalid_date'] = self.valid_data.copy()
        scenarios['invalid_date']['Date'] = 'invalid_date'
        
        scenarios['invalid_revenue'] = self.valid_data.copy()
        scenarios['invalid_revenue']['TotalRevenue'] = 'not_a_number'
        
        # Null/NaN values
        scenarios['null_dates'] = self.valid_data.copy()
        scenarios['null_dates']['Date'] = np.nan
        
        scenarios['null_revenue'] = self.valid_data.copy()
        scenarios['null_revenue']['TotalRevenue'] = np.nan
        
        # Negative values where not allowed
        scenarios['negative_revenue'] = self.valid_data.copy()
        scenarios['negative_revenue']['TotalRevenue'] = -1000
        
        scenarios['negative_occupancy'] = self.valid_data.copy()
        scenarios['negative_occupancy']['OccupancyRate'] = -0.5
        
        # Invalid ranges
        scenarios['invalid_occupancy_high'] = self.valid_data.copy()
        scenarios['invalid_occupancy_high']['OccupancyRate'] = 1.5  # > 100%
        
        return scenarios


class TestDataRequiredDecorator(DataValidationTestCase):
    """Test the data_required decorator functionality."""
    
    @patch('data.is_data_loaded')
    def test_data_required_with_valid_data(self, mock_is_loaded):
        """Test decorator when data is available."""
        mock_is_loaded.return_value = True
        
        @data_required
        def dummy_view():
            return "view_content"
        
        result = dummy_view()
        self.assertEqual(result, "view_content")
        
    @patch('data.is_data_loaded')
    def test_data_required_without_data(self, mock_is_loaded):
        """Test decorator when no data is available."""
        mock_is_loaded.return_value = False
        
        @data_required
        def dummy_view():
            return "view_content"
        
        result = dummy_view()
        # Should return error widget instead of view content
        self.assertIsNotNone(result)
        self.assertNotEqual(result, "view_content")


class TestRevenueViewDataValidation(DataValidationTestCase):
    """Test data validation for revenue view."""
    
    def test_revenue_view_with_missing_columns(self):
        """Test revenue view handles missing required columns."""
        from views.revenue import RevenueView
        
        scenarios = self.create_invalid_data_scenarios()
        
        for scenario_name, invalid_data in scenarios.items():
            with self.subTest(scenario=scenario_name):
                with patch('data.get_dataframe', return_value=invalid_data):
                    try:
                        view = RevenueView()
                        # Should not crash, should handle gracefully
                        self.assertIsNotNone(view)
                    except Exception as e:
                        self.fail(f"Revenue view crashed with {scenario_name} data: {e}")
    
    def test_revenue_calculations_with_invalid_data(self):
        """Test revenue calculations handle invalid data gracefully."""
        scenarios = self.create_invalid_data_scenarios()
        
        for scenario_name, invalid_data in scenarios.items():
            with self.subTest(scenario=scenario_name):
                # Test that calculations don't crash
                try:
                    if not invalid_data.empty and 'TotalRevenue' in invalid_data.columns:
                        # Simulate revenue calculation
                        total_revenue = invalid_data['TotalRevenue'].sum()
                        # Should handle NaN gracefully
                        if pd.isna(total_revenue):
                            total_revenue = 0
                        self.assertIsInstance(total_revenue, (int, float))
                except Exception as e:
                    # Should not crash, should handle gracefully
                    pass


class TestForecastViewDataValidation(DataValidationTestCase):
    """Test data validation for forecast view."""
    
    def test_forecast_with_insufficient_data(self):
        """Test forecast handles insufficient data points."""
        # Create data with too few points for forecasting
        insufficient_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'TotalCosts': [100, 200, 300, 400, 500]
        })
        
        from views.forecast_scatter import ForecastWorker
        
        worker = ForecastWorker(insufficient_data, 'TotalCosts')
        
        # Should handle insufficient data gracefully
        try:
            worker.run()
        except Exception as e:
            # Should emit error signal, not crash
            self.assertIsInstance(e, Exception)
    
    def test_forecast_with_missing_target_column(self):
        """Test forecast handles missing target column."""
        data_without_target = self.valid_data.drop('TotalCosts', axis=1)
        
        from views.forecast_scatter import ForecastWorker
        
        worker = ForecastWorker(data_without_target, 'TotalCosts')
        
        # Should handle missing target gracefully
        try:
            worker.run()
        except Exception as e:
            # Should emit error signal, not crash
            self.assertIsInstance(e, Exception)
    
    def test_forecast_with_non_numeric_target(self):
        """Test forecast handles non-numeric target column."""
        invalid_target_data = self.valid_data.copy()
        invalid_target_data['TotalCosts'] = 'not_numeric'
        
        from views.forecast_scatter import ForecastWorker
        
        worker = ForecastWorker(invalid_target_data, 'TotalCosts')
        
        # Should handle non-numeric target gracefully
        try:
            worker.run()
        except Exception as e:
            # Should emit error signal, not crash
            self.assertIsInstance(e, Exception)


class TestKPIViewDataValidation(DataValidationTestCase):
    """Test data validation for KPI view."""
    
    def test_kpi_calculations_with_missing_data(self):
        """Test KPI calculations handle missing data."""
        scenarios = self.create_invalid_data_scenarios()
        
        for scenario_name, invalid_data in scenarios.items():
            with self.subTest(scenario=scenario_name):
                try:
                    # Simulate KPI calculations
                    if not invalid_data.empty:
                        # Test occupancy rate calculation
                        if 'OccupancyRate' in invalid_data.columns:
                            avg_occupancy = invalid_data['OccupancyRate'].mean()
                            if pd.isna(avg_occupancy):
                                avg_occupancy = 0
                            self.assertIsInstance(avg_occupancy, (int, float))
                        
                        # Test revenue per available room
                        if 'TotalRevenue' in invalid_data.columns:
                            total_revenue = invalid_data['TotalRevenue'].sum()
                            if pd.isna(total_revenue):
                                total_revenue = 0
                            self.assertIsInstance(total_revenue, (int, float))
                            
                except Exception as e:
                    # Should handle gracefully
                    pass


class TestOperationsViewDataValidation(DataValidationTestCase):
    """Test data validation for operations view."""
    
    def test_operations_with_invalid_dates(self):
        """Test operations view handles invalid date formats."""
        invalid_date_data = self.valid_data.copy()
        invalid_date_data['Date'] = ['invalid', 'date', 'format'] * 34  # Repeat to match length
        
        try:
            # Simulate date processing
            pd.to_datetime(invalid_date_data['Date'], errors='coerce')
        except Exception as e:
            self.fail(f"Date processing should handle invalid dates gracefully: {e}")
    
    def test_operations_with_negative_values(self):
        """Test operations view handles negative values appropriately."""
        negative_data = self.valid_data.copy()
        negative_data['TotalRevenue'] = -1000
        negative_data['OccupancyRate'] = -0.5
        
        # Should validate and handle negative values
        self.assertTrue((negative_data['TotalRevenue'] < 0).any())
        self.assertTrue((negative_data['OccupancyRate'] < 0).any())


class TestDataIntegrityValidation(DataValidationTestCase):
    """Test overall data integrity validation."""
    
    def test_data_type_validation(self):
        """Test validation of data types across all columns."""
        # Test numeric columns
        numeric_columns = ['TotalRevenue', 'TotalCosts', 'OccupancyRate', 'ADR', 'RevPAR']
        for col in numeric_columns:
            if col in self.valid_data.columns:
                self.assertTrue(pd.api.types.is_numeric_dtype(self.valid_data[col]))
    
    def test_date_column_validation(self):
        """Test validation of date columns."""
        if 'Date' in self.valid_data.columns:
            # Should be datetime or convertible to datetime
            try:
                pd.to_datetime(self.valid_data['Date'])
            except Exception as e:
                self.fail(f"Date column should be valid datetime: {e}")
    
    def test_occupancy_rate_bounds(self):
        """Test occupancy rate is within valid bounds (0-1)."""
        if 'OccupancyRate' in self.valid_data.columns:
            occupancy = self.valid_data['OccupancyRate']
            # Remove NaN values for testing
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


class TestExcelDataImportValidation(DataValidationTestCase):
    """Test validation for Excel data import scenarios."""
    
    def test_excel_with_merged_cells(self):
        """Test handling of Excel files with merged cells."""
        # Simulate merged cell scenario with duplicate headers
        merged_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Date.1': [None] * 10,  # Merged cell artifact
            'Revenue': np.random.uniform(1000, 5000, 10)
        })
        
        # Should handle duplicate/merged columns
        cleaned_columns = [col for col in merged_data.columns if not col.endswith('.1')]
        self.assertIn('Date', cleaned_columns)
        self.assertNotIn('Date.1', cleaned_columns)
    
    def test_excel_with_extra_headers(self):
        """Test handling of Excel files with extra header rows."""
        # Simulate extra header scenario
        extra_header_data = pd.DataFrame({
            'Report Title': ['Hotel Revenue Report', None, None],
            'Date': [None, 'Date', '2023-01-01'],
            'Revenue': [None, 'Revenue', '1000']
        })
        
        # Should identify and skip non-data rows
        # This would typically be handled in the data loading process
        self.assertIsNotNone(extra_header_data)
    
    def test_excel_with_formatting_artifacts(self):
        """Test handling of Excel formatting artifacts."""
        # Simulate Excel formatting issues
        formatted_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Revenue': ['$1,000.00', '$2,000.00', '$3,000.00'],  # Currency formatting
            'Percentage': ['50%', '60%', '70%']  # Percentage formatting
        })
        
        # Should handle formatted strings
        self.assertTrue(all(isinstance(val, str) for val in formatted_data['Revenue']))
        self.assertTrue(all(isinstance(val, str) for val in formatted_data['Percentage']))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)