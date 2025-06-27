# tests/test_scenario_calculator.py
"""
Tests for Scenario Calculator Module
====================================

Tests the NumPy-based scenario calculation functions.
"""

import pytest
import pandas as pd
import numpy as np
from data.scenario_calculator import (
    simulate_30day_pl,
    calculate_waterfall_data,
    _default_kpis
)


class TestScenarioCalculator:
    """Test suite for scenario calculator functions."""

    def test_simulate_30day_pl_basic(self):
        """Test basic scenario calculation with valid inputs."""
        # Create sample baseline data
        baseline_data = pd.DataFrame({
            'rate': [150, 160, 140, 155],
            'occupancy': [0.75, 0.80, 0.70, 0.78],
            'date': pd.date_range('2024-01-01', periods=4),
        })

        # Test inputs
        inputs = {
            'room_rate': 175.0,
            'occupancy_pct': 80.0,
            'housekeeping_staff': 12,
            'fb_staff': 10,
            'promotion_active': False,
        }

        # Calculate scenario
        result = simulate_30day_pl(inputs, baseline_data)

        # Assertions
        assert isinstance(result, dict)
        assert 'revpar' in result
        assert 'goppar' in result
        assert 'profit_delta' in result
        assert 'total_revenue' in result
        assert 'total_costs' in result
        assert 'occupancy_rooms' in result

        # Check that values are reasonable
        assert result['revpar'] > 0
        assert result['occupancy_rooms'] == 80.0  # 100 rooms * 80%
        assert result['total_revenue'] > 0
        assert result['total_costs'] > 0

    def test_simulate_30day_pl_with_promotion(self):
        """Test scenario calculation with promotion active."""
        baseline_data = pd.DataFrame({
            'rate': [150, 160, 140],
            'occupancy': [0.75, 0.80, 0.70],
        })

        inputs_no_promo = {
            'room_rate': 150.0,
            'occupancy_pct': 75.0,
            'housekeeping_staff': 10,
            'fb_staff': 8,
            'promotion_active': False,
        }

        inputs_with_promo = inputs_no_promo.copy()
        inputs_with_promo['promotion_active'] = True

        result_no_promo = simulate_30day_pl(inputs_no_promo, baseline_data)
        result_with_promo = simulate_30day_pl(inputs_with_promo, baseline_data)

        # Revenue should be lower with promotion (10% discount)
        assert result_with_promo['total_revenue'] < result_no_promo['total_revenue']

        # The difference should be approximately 10% of room revenue
        # Only room revenue gets discount
        expected_discount = result_no_promo['total_revenue'] * 0.1 * (2/3)
        actual_difference = (result_no_promo['total_revenue'] -
                             result_with_promo['total_revenue'])
        assert abs(actual_difference - expected_discount) < 1000

    def test_simulate_30day_pl_empty_baseline(self):
        """Test scenario calculation with empty baseline data."""
        empty_baseline = pd.DataFrame()

        inputs = {
            'room_rate': 150.0,
            'occupancy_pct': 75.0,
            'housekeeping_staff': 10,
            'fb_staff': 8,
            'promotion_active': False,
        }

        result = simulate_30day_pl(inputs, empty_baseline)

        # Should return default KPIs
        expected = _default_kpis()
        assert result == expected

    def test_calculate_waterfall_data(self):
        """Test waterfall chart data calculation."""
        baseline_kpis = {
            'gross_profit': 100000,
            'total_revenue': 200000,
            'total_costs': 100000,
        }

        scenario_kpis = {
            'gross_profit': 120000,
            'total_revenue': 230000,
            'total_costs': 110000,
        }

        result = calculate_waterfall_data(baseline_kpis, scenario_kpis)

        # Check structure
        assert 'categories' in result
        assert 'values' in result
        assert 'colors' in result
        assert 'net_change' in result

        # Check values
        assert len(result['categories']) == 4
        assert len(result['values']) == 4
        assert len(result['colors']) == 4

        # Net change should match profit difference
        expected_net_change = (scenario_kpis['gross_profit'] -
                               baseline_kpis['gross_profit'])
        assert result['net_change'] == expected_net_change

        # Revenue impact should be positive
        revenue_impact = result['values'][1]
        assert revenue_impact == 30000  # 230000 - 200000

        # Cost impact should be negative (in the context of the waterfall)
        cost_impact = result['values'][2]
        assert cost_impact == -10000  # -(110000 - 100000)

    def test_default_kpis(self):
        """Test default KPIs function."""
        result = _default_kpis()

        assert isinstance(result, dict)
        expected_keys = [
            'revpar', 'goppar', 'profit_delta', 'total_revenue',
            'total_costs', 'occupancy_rooms', 'gross_profit'
        ]

        for key in expected_keys:
            assert key in result
            assert result[key] == 0.0

    def test_scenario_with_different_staffing(self):
        """Test scenario with different staffing levels."""
        baseline_data = pd.DataFrame({
            'rate': [150, 160, 140],
            'occupancy': [0.75, 0.80, 0.70],
        })

        base_inputs = {
            'room_rate': 150.0,
            'occupancy_pct': 75.0,
            'housekeeping_staff': 10,
            'fb_staff': 8,
            'promotion_active': False,
        }

        high_staff_inputs = base_inputs.copy()
        high_staff_inputs.update({
            'housekeeping_staff': 15,
            'fb_staff': 12,
        })

        base_result = simulate_30day_pl(base_inputs, baseline_data)
        high_staff_result = simulate_30day_pl(high_staff_inputs, baseline_data)

        # Higher staffing should result in higher costs
        assert high_staff_result['total_costs'] > base_result['total_costs']

        # GOPPAR should be lower with higher staffing costs
        assert high_staff_result['goppar'] < base_result['goppar']

    @pytest.mark.parametrize("occupancy_pct", [50.0, 75.0, 90.0, 100.0])
    def test_occupancy_variations(self, occupancy_pct):
        """Test scenario calculation with various occupancy levels."""
        baseline_data = pd.DataFrame({
            'rate': [150],
            'occupancy': [0.75],
        })

        inputs = {
            'room_rate': 150.0,
            'occupancy_pct': occupancy_pct,
            'housekeeping_staff': 10,
            'fb_staff': 8,
            'promotion_active': False,
        }

        result = simulate_30day_pl(inputs, baseline_data)

        # Occupied rooms should match occupancy percentage
        expected_rooms = 100 * (occupancy_pct / 100)  # 100 total rooms
        assert result['occupancy_rooms'] == expected_rooms

        # Revenue should increase with occupancy
        assert result['total_revenue'] > 0
        assert result['revpar'] > 0


# Additional fixtures and helper functions for testing
@pytest.fixture
def sample_baseline_data():
    """Fixture providing sample baseline data for tests."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'rate': np.random.normal(150, 20, 30),
        'occupancy': np.random.normal(0.75, 0.1, 30),
        'room_type': ['Standard'] * 15 + ['Deluxe'] * 15,
    })


@pytest.fixture
def sample_inputs():
    """Fixture providing sample scenario inputs."""
    return {
        'room_rate': 175.0,
        'occupancy_pct': 80.0,
        'housekeeping_staff': 12,
        'fb_staff': 10,
        'promotion_active': False,
    }
