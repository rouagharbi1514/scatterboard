# Hotel Dashboard Application

## Project Overview

The Hotel Dashboard is an interactive analytics application that visualizes and analyzes hotel performance metrics across multiple domains. The application provides a suite of dashboards for monitoring key performance indicators (KPIs) for hotel management.

## Data Connectors

The application supports multiple data sources:

- **Oracle OPERA Cloud (OHIP)**: Connect directly to Oracle Hospitality Integration Platform
- **Local PostgreSQL Database**: Connect to a self-hosted PMS database
- **CSV/Excel Files**: Import data from CSV or Excel files

## 3-Step Quick Start Guide

1. Set up your environment configuration:
   ```bash
   cp .env.example .env
   nano .env  # Edit configuration for your environment
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the data connector:
   ```bash
   python -m hotel_dashboard.connectors.runner
   ```

## Additional Runner Options

You can customize the data extraction:

```bash
# Pull 60 days of availability data
python -m hotel_dashboard.connectors.runner --days 60

# Enable verbose logging
python -m hotel_dashboard.connectors.runner --verbose
```

## Scheduling Data Collection

To schedule regular data collection, add the runner to your crontab:

```bash
# Example: Run data collection every day at 2 AM
0 2 * * * cd /path/to/hotel-dashboard && python -m hotel_dashboard.connectors.runner
```

## Testing Progress Report

We have successfully implemented unit tests for the core data handling functionality using pytest:

1. Created `helpers_test.py` to test the `get_df()` function in `data/helpers.py`
2. Implemented 5 test cases covering different aspects of the data preparation pipeline:
   - Empty dataframe handling and default columns
   - Column renaming from lowercase to proper case
   - Revenue calculation based on ADR and occupancy
   - Automatic BookingSource assignment
   - UpsellRevenue addition based on room revenue

These tests verify that our data processing pipeline correctly handles various input scenarios and consistently produces the expected output structure needed by the visualization modules.

### Test Results

All tests are now passing successfully:

## Project Functionality

### Core Features

The Hotel Dashboard application provides analytical insights across several key hotel management domains:

1. **Overview Dashboard**: General KPIs and hotel performance at a glance
2. **Revenue Analysis**:
   - ADR (Average Daily Rate) trends
   - RevPAR (Revenue Per Available Room) analysis
   - Revenue sources breakdown
3. **Profitability Metrics**:
   - Profit margins by room type
   - Fixed vs variable cost analysis
   - Revenue and cost trend visualization
4. **Room Cost Analysis**:
   - Cost per occupied room (CPOR) tracking
   - Cost category breakdown
   - Room margin analysis
5. **Operations Management**:
   - Food & Beverage performance
   - Operational efficiency metrics
   - Housekeeping performance
6. **Marketing Performance**:
   - Campaign ROI analysis
   - Channel performance
   - Booking source analytics
7. **Seasonality Analysis**:
   - Occupancy patterns
   - Seasonal pricing strategies
   - Demand forecasting

### Technical Implementation

The application is built with:
- Python for backend data processing
- PySide6 (Qt) for the frontend UI
- Pandas for data manipulation
- Matplotlib for visualization
- Pytest for testing

All dashboards are modular, allowing for easy extension and customization. Data transformations ensure that even incomplete data can be visualized effectively.

## Running the Application

### Prerequisites

- Python 3.10+ recommended
- Required Python packages: pandas, matplotlib, PySide6, pytest

### Installation

1. Clone the repository:
2. Install dependencies:

### Running the Dashboard

To start the main application:

### Running Tests

To run all tests:

To run specific test files:

## Development

The code is organized as follows:
- `hotel-dashboard/`: Main application directory
  - `data/`: Data handling and processing modules
  - `views/`: Dashboard view modules for different analytics domains
  - `assets/`: UI assets and resources
  - `tests/`: Test files

When contributing, please ensure all tests pass and follow the project's coding standards.
