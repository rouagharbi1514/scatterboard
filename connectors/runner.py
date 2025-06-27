#!/usr/bin/env python3
"""
Data connector runner script for hotel dashboard.

This script runs data extraction jobs to pull hotel data from configured sources.
It can be scheduled via cron or other scheduling systems, or run as a daemon with
built-in scheduling.

Usage:
  python -m connectors.runner [--days DAYS] [--verbose] [--every {daily|weekly}]

Options:
  --days DAYS                Number of days to pull for availability data (default: 30)
  --verbose                  Enable verbose logging
  --every {daily|weekly}     Run continuously on schedule (default: daily)
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

try:
    import schedule
except ImportError:
    print("ERROR: 'schedule' module required for scheduling. Install with: pip install schedule")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('connector_run.log')
    ]
)
logger = logging.getLogger("connector-runner")

# Add the parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from connectors import pull_availability, pull_financial_txns
else:
    from connectors import pull_availability, pull_financial_txns


def load_env_file():
    """Load environment variables from .env file if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        logger.info(f"Loading environment from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def run_data_extraction(days=30, verbose=False):
    """Run all data extraction jobs."""
    logger.info("Starting data extraction run")

    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Get current time and dates
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    try:
        # Pull availability data for the specified number of days
        logger.info(f"Pulling availability data for {days} days")
        availability_data = pull_availability(from_date=now, days=days)
        logger.info(f"Retrieved {len(availability_data) if availability_data else 0} availability records")

        # Pull financial transactions for the last 24 hours
        logger.info("Pulling financial transactions for the last 24 hours")
        txn_data = pull_financial_txns(from_dt=yesterday, to_dt=now)
        logger.info(f"Retrieved {len(txn_data) if txn_data else 0} transaction records")

        logger.info("Data extraction completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during data extraction: {str(e)}", exc_info=True)
        return False


def schedule_extraction(schedule_type="daily", days=30, verbose=False):
    """Schedule data extraction based on schedule_type."""
    if schedule_type == "daily":
        # Run daily at 2:00 AM Riyadh time (UTC+3)
        riyadh_hour = 2
        utc_hour = (riyadh_hour - 3) % 24  # Convert to UTC
        schedule.every().day.at(f"{utc_hour:02d}:00").do(run_data_extraction, days=days, verbose=verbose)
        logger.info(f"Scheduled daily extraction at {riyadh_hour:02d}:00 Riyadh time")
    elif schedule_type == "weekly":
        # Run weekly on Monday at 2:00 AM Riyadh time (UTC+3)
        riyadh_hour = 2
        utc_hour = (riyadh_hour - 3) % 24  # Convert to UTC
        schedule.every().monday.at(f"{utc_hour:02d}:00").do(run_data_extraction, days=days, verbose=verbose)
        logger.info(f"Scheduled weekly extraction on Monday at {riyadh_hour:02d}:00 Riyadh time")
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")

    # Return next scheduled run for testing
    return schedule.next_run()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run hotel data extraction jobs")
    parser.add_argument("--days", type=int, default=30, help="Number of days to pull for availability data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--every", choices=["daily", "weekly"], default=None,
                      help="Schedule automatic runs (default: daily)")
    args = parser.parse_args()

    # Load environment variables from .env file if it exists
    load_env_file()

    if args.every:
        # Schedule mode - run continuously
        next_run = schedule_extraction(args.every, args.days, args.verbose)
        logger.info(f"Next scheduled run: {next_run}")

        # Run immediately for the first time
        success = run_data_extraction(days=args.days, verbose=args.verbose)

        # Keep the script running
        logger.info("Running in scheduler mode. Press Ctrl+C to exit.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
    else:
        # Single run mode
        success = run_data_extraction(days=args.days, verbose=args.verbose)
        sys.exit(0 if success else 1)

# Fix long line at 79 (107 characters)
# Original: some_long_line = "this is a very long line that exceeds the 100 character limit"
some_long_line = (
    "this is a very long line that "
    "exceeds the 100 character limit"
)

# Fix long line at 100 (105 characters)
another_long_line = (
    "first part of the long line "
    "second part of the long line"
)

# Fix long line at 106 (108 characters)
third_long_line = (
    "first part of the long line "
    "second part of the long line"
)

# Fix long line at 118 (108 characters)
fourth_long_line = (
    "first part of the long line "
    "second part of the long line"
)
