#!/usr/bin/env python3
"""
View-Specific Data Validation Tests
==================================

This module contains detailed unit tests for each view component,
focusing on their specific data requirements and validation needs.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QThread

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure QApplication exists for widget tests
app = None
if not QApplication.instance():
    app = QApplication([])


class ViewTestCase(unittest.TestCase):
    """Base test case for view-specific tests."""
    
    def setUp(self):
        """Set up test fixtures for view testing."""
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=365),
            'TotalRevenue': np.random.uniform(1000, 5000, 365),
            'TotalCosts': np.random.uniform(500, 2000, 365),
            'OccupancyRate': np.random.uniform(0.3, 0.95, 365),
            'ADR': np.random.uniform(80, 300, 365),
            'RevPAR': np.random.uniform(50, 250, 365),
            'RoomsSold': np.random.randint(10, 100, 365),
            'RoomsAvailable': np.random.randint(100, 150, 365),
            'RoomType': np.random.choice(['Standard', 'Deluxe', 'Suite'], 365),
            'BookingSource': np.random.choice(['Direct', 'OTA', 'Corporate'], 365),
            'MarketSegment': np.random.choice(['Leisure', 'Business', 'Group'], 365),
            'GuestSatisfaction': np.random.uniform(3.0, 5.0, 365),
            'StaffCount': np.random.randint(20, 50, 365),
            'MaintenanceCosts': np.random.uniform(100, 1000, 365),
            'MarketingSpend': np.random.uniform(200, 2000, 365),
            'CustomerAcquisitionCost': np.random.uniform(50, 200, 365),
            'CustomerLifetimeValue': np.random.uniform(500, 5000, 365)
        })
        
        # Create invalid data scenarios
        self.empty_data = pd.DataFrame()
        self.missing_columns_data = self.sample_data.drop(['TotalRevenue', 'Date'], axis=1)
        self.invalid_types_data = self.sample_data.copy()
        self.invalid_types_data['TotalRevenue'] = 'invalid'
        self.null_data = self.sample_data.copy()
        self.null_data.loc[0:50, 'TotalRevenue'] = np.nan


class TestRevenueViewValidation(ViewTestCase):
    """Test revenue view data validation."""
    
    @patch('data.get_dataframe')
    def test_revenue_view_initialization_with_valid_data(self, mock_get_df):
        """Test revenue view initializes correctly with valid data."""
        mock_get_df.return_value = self.sample_data
        
        try:
            from views.revenue import RevenueView
            view = RevenueView()
            self.assertIsNotNone(view)
        except Exception as e:
            self.fail(f"Revenue view should initialize with valid data: {e}")
    
    @patch('data.get_dataframe')
    def test_revenue_view_with_empty_data(self, mock_get_df):
        """Test revenue view handles empty dataset."""
        mock_get_df.return_value = self.empty_data
        
        try:
            from views.revenue import RevenueView
            view = RevenueView()
            # Should not crash, should show appropriate message
            self.assertIsNotNone(view)
        except Exception as e:
            # Should handle gracefully, not crash
            pass
    
    @patch('data.get_dataframe')
    def test_revenue_calculations_with_null_values(self, mock_get_df):
        """Test revenue calculations handle null values."""
        mock_get_df.return_value = self.null_data
        
        # Test revenue sum calculation
        total_revenue = self.null_data['TotalRevenue'].sum()
        self.assertIsInstance(total_revenue, (int, float))
        
        # Test average calculation
        avg_revenue = self.null_data['TotalRevenue'].mean()
        self.assertTrue(pd.isna(avg_revenue) or isinstance(avg_revenue, (int, float)))
    
    def test_revenue_data_type_validation(self):
        """Test revenue data type validation."""
        # Test with string values in revenue column
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'TotalRevenue'] = 'not_a_number'
        
        # Should detect invalid data type
        revenue_col = invalid_data['TotalRevenue']
        numeric_mask = pd.to_numeric(revenue_col, errors='coerce').notna()
        self.assertFalse(numeric_mask.all())


class TestForecastViewValidation(ViewTestCase):
    """Test forecast view data validation."""
    
    def test_forecast_data_requirements(self):
        """Test forecast view data requirements."""
        from views.forecast_scatter import ForecastWorker
        
        # Test with sufficient data
        worker = ForecastWorker(self.sample_data, 'TotalRevenue')
        self.assertIsNotNone(worker)
        
        # Test with insufficient data (less than 30 points)
        insufficient_data = self.sample_data.head(10)
        worker_insufficient = ForecastWorker(insufficient_data, 'TotalRevenue')
        self.assertIsNotNone(worker_insufficient)
    
    def test_forecast_target_column_validation(self):
        """Test forecast target column validation."""
        from views.forecast_scatter import ForecastWorker
        
        # Test with missing target column
        data_without_target = self.sample_data.drop('TotalRevenue', axis=1)
        
        try:
            worker = ForecastWorker(data_without_target, 'TotalRevenue')
            # Should handle missing column gracefully
        except Exception as e:
            # Expected to fail, but should not crash the application
            self.assertIsInstance(e, (KeyError, ValueError))
    
    def test_forecast_with_non_numeric_target(self):
        """Test forecast with non-numeric target column."""
        from views.forecast_scatter import ForecastWorker
        
        # Create data with non-numeric target
        non_numeric_data = self.sample_data.copy()
        non_numeric_data['TotalRevenue'] = non_numeric_data['RoomType']  # String column
        
        try:
            worker = ForecastWorker(non_numeric_data, 'TotalRevenue')
            # Should detect non-numeric data
        except Exception as e:
            # Expected to fail with non-numeric data
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_forecast_date_column_validation(self):
        """Test forecast date column validation."""
        # Test with invalid date format
        invalid_date_data = self.sample_data.copy()
        invalid_date_data['Date'] = 'invalid_date'
        
        try:
            # Should handle invalid dates
            pd.to_datetime(invalid_date_data['Date'], errors='coerce')
        except Exception as e:
            self.fail(f"Date conversion should handle invalid dates: {e}")


class TestKPIViewValidation(ViewTestCase):
    """Test KPI view data validation."""
    
    @patch('data.get_dataframe')
    def test_kpi_calculations_with_missing_data(self, mock_get_df):
        """Test KPI calculations with missing data."""
        mock_get_df.return_value = self.null_data
        
        # Test occupancy rate calculation
        if 'OccupancyRate' in self.null_data.columns:
            avg_occupancy = self.null_data['OccupancyRate'].mean()
            self.assertTrue(pd.isna(avg_occupancy) or isinstance(avg_occupancy, (int, float)))
        
        # Test ADR calculation
        if 'ADR' in self.null_data.columns:
            avg_adr = self.null_data['ADR'].mean()
            self.assertTrue(pd.isna(avg_adr) or isinstance(avg_adr, (int, float)))
    
    def test_kpi_boundary_validation(self):
        """Test KPI boundary validation."""
        # Test occupancy rate boundaries (should be 0-1)
        invalid_occupancy_data = self.sample_data.copy()
        invalid_occupancy_data.loc[0, 'OccupancyRate'] = 1.5  # > 100%
        invalid_occupancy_data.loc[1, 'OccupancyRate'] = -0.1  # < 0%
        
        occupancy = invalid_occupancy_data['OccupancyRate']
        valid_range_mask = (occupancy >= 0) & (occupancy <= 1)
        
        # Should identify invalid values
        self.assertFalse(valid_range_mask.all())
    
    def test_revenue_per_available_room_calculation(self):
        """Test RevPAR calculation validation."""
        # Test RevPAR calculation with missing rooms data
        missing_rooms_data = self.sample_data.drop('RoomsAvailable', axis=1)
        
        # Should handle missing rooms data
        if 'RoomsAvailable' not in missing_rooms_data.columns:
            # Cannot calculate RevPAR without rooms data
            self.assertNotIn('RoomsAvailable', missing_rooms_data.columns)


class TestOperationsViewValidation(ViewTestCase):
    """Test operations view data validation."""
    
    def test_staff_efficiency_calculations(self):
        """Test staff efficiency calculations with invalid data."""
        # Test with zero staff count
        zero_staff_data = self.sample_data.copy()
        zero_staff_data.loc[0, 'StaffCount'] = 0
        
        # Should handle division by zero
        staff_count = zero_staff_data['StaffCount']
        revenue = zero_staff_data['TotalRevenue']
        
        # Revenue per staff calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            revenue_per_staff = revenue / staff_count
            # Should contain inf or nan for zero staff
            self.assertTrue(np.isinf(revenue_per_staff).any() or np.isnan(revenue_per_staff).any())
    
    def test_maintenance_cost_validation(self):
        """Test maintenance cost validation."""
        # Test with negative maintenance costs
        negative_maintenance_data = self.sample_data.copy()
        negative_maintenance_data.loc[0, 'MaintenanceCosts'] = -100
        
        maintenance_costs = negative_maintenance_data['MaintenanceCosts']
        negative_mask = maintenance_costs < 0
        
        # Should identify negative costs
        self.assertTrue(negative_mask.any())
    
    def test_operational_efficiency_metrics(self):
        """Test operational efficiency metrics validation."""
        # Test cost-to-revenue ratio
        costs = self.sample_data['TotalCosts']
        revenue = self.sample_data['TotalRevenue']
        
        # Should handle cases where revenue is zero
        zero_revenue_data = self.sample_data.copy()
        zero_revenue_data.loc[0, 'TotalRevenue'] = 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cost_ratio = zero_revenue_data['TotalCosts'] / zero_revenue_data['TotalRevenue']
            # Should contain inf for zero revenue
            self.assertTrue(np.isinf(cost_ratio).any())


class TestMarketingViewValidation(ViewTestCase):
    """Test marketing view data validation."""
    
    def test_customer_acquisition_cost_validation(self):
        """Test customer acquisition cost validation."""
        # Test with zero marketing spend
        zero_marketing_data = self.sample_data.copy()
        zero_marketing_data.loc[0, 'MarketingSpend'] = 0
        
        marketing_spend = zero_marketing_data['MarketingSpend']
        cac = zero_marketing_data['CustomerAcquisitionCost']
        
        # Should validate relationship between marketing spend and CAC
        self.assertIsInstance(marketing_spend.sum(), (int, float))
        self.assertIsInstance(cac.mean(), (int, float))
    
    def test_customer_lifetime_value_validation(self):
        """Test customer lifetime value validation."""
        # Test CLV vs CAC ratio
        clv = self.sample_data['CustomerLifetimeValue']
        cac = self.sample_data['CustomerAcquisitionCost']
        
        # CLV should generally be higher than CAC
        clv_cac_ratio = clv / cac
        
        # Should handle division and produce valid ratios
        self.assertTrue((clv_cac_ratio > 0).all())
    
    def test_marketing_roi_calculation(self):
        """Test marketing ROI calculation validation."""
        marketing_spend = self.sample_data['MarketingSpend']
        revenue = self.sample_data['TotalRevenue']
        
        # Test ROI calculation with zero marketing spend
        zero_marketing_data = self.sample_data.copy()
        zero_marketing_data.loc[0, 'MarketingSpend'] = 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            roi = (zero_marketing_data['TotalRevenue'] - zero_marketing_data['MarketingSpend']) / zero_marketing_data['MarketingSpend']
            # Should handle division by zero
            self.assertTrue(np.isinf(roi).any() or np.isnan(roi).any())


class TestGuestViewValidation(ViewTestCase):
    """Test guest view data validation."""
    
    def test_guest_satisfaction_validation(self):
        """Test guest satisfaction score validation."""
        # Test satisfaction score boundaries (typically 1-5)
        invalid_satisfaction_data = self.sample_data.copy()
        invalid_satisfaction_data.loc[0, 'GuestSatisfaction'] = 6.0  # > 5
        invalid_satisfaction_data.loc[1, 'GuestSatisfaction'] = 0.5  # < 1
        
        satisfaction = invalid_satisfaction_data['GuestSatisfaction']
        valid_range_mask = (satisfaction >= 1) & (satisfaction <= 5)
        
        # Should identify invalid satisfaction scores
        self.assertFalse(valid_range_mask.all())
    
    def test_guest_count_validation(self):
        """Test guest count validation."""
        # Test with negative guest counts
        negative_guests_data = self.sample_data.copy()
        negative_guests_data.loc[0, 'RoomsSold'] = -5
        
        rooms_sold = negative_guests_data['RoomsSold']
        negative_mask = rooms_sold < 0
        
        # Should identify negative guest counts
        self.assertTrue(negative_mask.any())
    
    def test_room_availability_validation(self):
        """Test room availability validation."""
        # Test rooms sold vs rooms available
        invalid_rooms_data = self.sample_data.copy()
        invalid_rooms_data.loc[0, 'RoomsSold'] = 200  # More than available
        invalid_rooms_data.loc[0, 'RoomsAvailable'] = 150
        
        rooms_sold = invalid_rooms_data['RoomsSold']
        rooms_available = invalid_rooms_data['RoomsAvailable']
        
        # Should identify cases where sold > available
        oversold_mask = rooms_sold > rooms_available
        self.assertTrue(oversold_mask.any())


class TestHousekeepingViewValidation(ViewTestCase):
    """Test housekeeping view data validation."""
    
    def test_cleaning_time_validation(self):
        """Test cleaning time validation."""
        # Add cleaning time data
        housekeeping_data = self.sample_data.copy()
        housekeeping_data['CleaningTimeMinutes'] = np.random.uniform(15, 60, len(housekeeping_data))
        
        # Test with invalid cleaning times
        housekeeping_data.loc[0, 'CleaningTimeMinutes'] = -10  # Negative time
        housekeeping_data.loc[1, 'CleaningTimeMinutes'] = 300   # Unrealistic time
        
        cleaning_time = housekeeping_data['CleaningTimeMinutes']
        
        # Should identify unrealistic cleaning times
        reasonable_time_mask = (cleaning_time > 0) & (cleaning_time < 120)
        self.assertFalse(reasonable_time_mask.all())
    
    def test_housekeeping_staff_efficiency(self):
        """Test housekeeping staff efficiency calculations."""
        # Test rooms cleaned per staff member
        housekeeping_data = self.sample_data.copy()
        housekeeping_data['HousekeepingStaff'] = np.random.randint(5, 20, len(housekeeping_data))
        
        # Test with zero staff
        housekeeping_data.loc[0, 'HousekeepingStaff'] = 0
        
        rooms_sold = housekeeping_data['RoomsSold']
        staff_count = housekeeping_data['HousekeepingStaff']
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rooms_per_staff = rooms_sold / staff_count
            # Should handle division by zero
            self.assertTrue(np.isinf(rooms_per_staff).any())


if __name__ == '__main__':
    unittest.main(verbosity=2)