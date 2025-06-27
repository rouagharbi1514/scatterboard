"""Test the scheduling functionality in runner.py"""
import unittest
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from connectors.runner import schedule_extraction


class TestRunnerSchedule(unittest.TestCase):

    def test_daily_schedule(self):
        """Test that daily schedule runs within 24 hours"""
        next_run = schedule_extraction("daily")
        delta = next_run - datetime.now()
        self.assertLessEqual(delta.total_seconds(), 24 * 3600)

    def test_weekly_schedule(self):
        """Test that weekly schedule runs within 8 days"""
        next_run = schedule_extraction("weekly")
        delta = next_run - datetime.now()
        self.assertLessEqual(delta.total_seconds(), 8 * 24 * 3600)


if __name__ == '__main__':
    unittest.main()
