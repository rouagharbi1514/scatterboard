# tests/test_what_if_calculations.py
import numpy as np
import pytest
from datetime import date, timedelta

# Import the calculation function from What-If module
from views.what_if import WhatIfPanel

# Mock data for tests
@pytest.fixture
def mock_elasticity_matrix():
    """Create a test elasticity matrix"""
    # Shape: (10, 5) - 10 features, 5 outputs [revpar, goppar, revenue, cost, profit]
    matrix = np.zeros((10, 5), dtype=np.float32)
    
    # Set some sample elasticities
    # Room types (4)
    matrix[0, :] = [0.5, 0.3, 10.0, 0.0, 10.0]  # Standard
    matrix[1, :] = [0.7, 0.5, 15.0, 0.0, 15.0]  # Deluxe
    matrix[2, :] = [0.9, 0.7, 25.0, 0.0, 25.0]  # Suite
    matrix[3, :] = [1.2, 1.0, 40.0, 0.0, 40.0]  # Presidential
    
    # Staffing (2)
    matrix[4, :] = [0.0, -0.1, 0.0, 500.0, -500.0]  # Housekeeping
    matrix[5, :] = [0.0, -0.2, 0.0, 800.0, -800.0]  # F&B
    
    # Promotions (4)
    matrix[6, :] = [0.1, 0.05, 5000.0, 2000.0, 3000.0]  # Spa
    matrix[7, :] = [0.2, 0.1, 8000.0, 3000.0, 5000.0]  # Breakfast
    matrix[8, :] = [0.15, 0.08, 7000.0, 4000.0, 3000.0]  # Resort Credit
    matrix[9, :] = [0.05, 0.02, 2000.0, 500.0, 1500.0]  # Late Checkout
    
    return matrix

@pytest.fixture
def mock_baseline_data():
    """Create test baseline data"""
    return {
        "revpar": 100.0,
        "goppar": 50.0,
        "total_revenue": 300000.0,
        "total_cost": 150000.0,
        "net_profit": 150000.0,
        "occupancy": 80.0,
        "staff_fte": {
            "housekeeping": 10,
            "fnb": 8
        }
    }

@pytest.fixture
def mock_room_types():
    """Create test room types"""
    return ["Standard", "Deluxe", "Suite", "Presidential"]

@pytest.fixture
def what_if_panel(monkeypatch):
    """Create a WhatIfPanel instance with mocked data"""
    panel = WhatIfPanel()
    
    # Mock the global variables
    import views.what_if
    monkeypatch.setattr(views.what_if, "ELASTICITY_MATRIX", mock_elasticity_matrix())
    monkeypatch.setattr(views.what_if, "BASELINE_DATA", mock_baseline_data())
    monkeypatch.setattr(views.what_if, "ROOM_TYPES", mock_room_types())
    
    return panel

def test_local_calculation_basic(what_if_panel):
    """Test basic calculation with default scenario"""
    scenario = {
        "room_rates": {"Standard": 0, "Deluxe": 0, "Suite": 0, "Presidential": 0},
        "target_occupancy": 80.0,  # Same as baseline
        "staffing": {"housekeeping": 0, "fnb": 0},
        "promotions": [],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # With no changes, deltas should be near zero
    assert abs(result["deltas"]["revpar"]) < 0.01
    assert abs(result["deltas"]["goppar"]) < 0.01
    assert abs(result["deltas"]["revenue"]) < 0.01
    assert abs(result["deltas"]["cost"]) < 0.01
    assert abs(result["deltas"]["profit"]) < 0.01

def test_price_increase_impact(what_if_panel):
    """Test impact of price increases"""
    scenario = {
        "room_rates": {"Standard": 10, "Deluxe": 20, "Suite": 30, "Presidential": 50},
        "target_occupancy": 80.0,
        "staffing": {"housekeeping": 0, "fnb": 0},
        "promotions": [],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # Price increases should positively impact revenue and profit
    assert result["deltas"]["revpar"] > 0
    assert result["deltas"]["revenue"] > 0
    assert result["deltas"]["profit"] > 0

def test_staffing_increase_impact(what_if_panel):
    """Test impact of staffing increases"""
    scenario = {
        "room_rates": {"Standard": 0, "Deluxe": 0, "Suite": 0, "Presidential": 0},
        "target_occupancy": 80.0,
        "staffing": {"housekeeping": 2, "fnb": 1},
        "promotions": [],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # Staffing increases should increase costs and decrease profit
    assert result["deltas"]["cost"] > 0
    assert result["deltas"]["profit"] < 0
    assert result["deltas_percent"]["profit"] < 0

def test_promotions_impact(what_if_panel):
    """Test impact of promotions"""
    scenario = {
        "room_rates": {"Standard": 0, "Deluxe": 0, "Suite": 0, "Presidential": 0},
        "target_occupancy": 80.0,
        "staffing": {"housekeeping": 0, "fnb": 0},
        "promotions": ["spa_discount", "breakfast"],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # Promotions should increase revenue, costs, and profit
    assert result["deltas"]["revenue"] > 0
    assert result["deltas"]["cost"] > 0
    assert result["deltas"]["profit"] > 0

def test_occupancy_increase_impact(what_if_panel):
    """Test impact of occupancy increases"""
    scenario = {
        "room_rates": {"Standard": 0, "Deluxe": 0, "Suite": 0, "Presidential": 0},
        "target_occupancy": 85.0,  # +5% from baseline
        "staffing": {"housekeeping": 0, "fnb": 0},
        "promotions": [],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # Higher occupancy should increase revenue and profit
    assert result["deltas"]["revpar"] > 0
    assert result["deltas"]["revenue"] > 0
    assert result["deltas"]["profit"] > 0
    
def test_combined_scenario(what_if_panel):
    """Test a combined scenario with multiple changes"""
    scenario = {
        "room_rates": {"Standard": 5, "Deluxe": 10, "Suite": 15, "Presidential": 25},
        "target_occupancy": 85.0,
        "staffing": {"housekeeping": 1, "fnb": 1},
        "promotions": ["spa_discount"],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    result = what_if_panel._local_calc(scenario)
    
    # Combined effects should be reflected in the waterfall chart
    waterfall = result["waterfall"]
    assert waterfall["price"] > 0
    assert waterfall["occupancy"] > 0
    assert waterfall["promo"] > 0
    assert waterfall["staff"] < 0
    
    # Final should equal baseline + sum of all impacts
    expected_final = (waterfall["baseline"] + waterfall["price"] + 
                      waterfall["occupancy"] + waterfall["promo"] + 
                      waterfall["staff"])
    assert abs(waterfall["final"] - expected_final) < 0.01
    
def test_calculation_performance(what_if_panel, benchmark):
    """Test the performance of the calculation"""
    scenario = {
        "room_rates": {"Standard": 5, "Deluxe": 10, "Suite": 15, "Presidential": 25},
        "target_occupancy": 85.0,
        "staffing": {"housekeeping": 1, "fnb": 1},
        "promotions": ["spa_discount", "breakfast"],
        "date_range": (date.today(), date.today() + timedelta(days=29))
    }
    
    # Benchmark the calculation function
    result = benchmark(what_if_panel._local_calc, scenario)
    
    # Assert the calculation completes and returns valid results
    assert result is not None
    assert "deltas" in result
    assert "waterfall" in result
    
    # The