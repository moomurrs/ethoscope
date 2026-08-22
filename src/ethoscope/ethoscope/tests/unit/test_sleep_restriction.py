"""
Unit tests for sleep restriction functionality.

Tests for DailyScheduler (legacy mAGOSleepRestriction tests removed).
"""

import json
import json
import os
import tempfile
import unittest

from ethoscope.utils.scheduler import DailyScheduleError, DailyScheduler


class TestDailyScheduler(unittest.TestCase):
    """Test cases for DailyScheduler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "test_state.json")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        os.rmdir(self.temp_dir)

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        scheduler = DailyScheduler(
            daily_duration_hours=8, interval_hours=24, daily_start_time="09:00:00"
        )

        self.assertEqual(scheduler._daily_duration_hours, 8)
        self.assertEqual(scheduler._interval_hours, 24)
        self.assertEqual(scheduler._daily_start_time, "09:00:00")
        self.assertEqual(scheduler._start_time_seconds, 9 * 3600)  # 9 AM in seconds

    def test_init_invalid_duration(self):
        """Test initialization with invalid duration hours."""
        with self.assertRaises(DailyScheduleError):
            DailyScheduler(daily_duration_hours=25)  # > 24 hours

        with self.assertRaises(DailyScheduleError):
            DailyScheduler(daily_duration_hours=0)  # <= 0 hours

    def test_init_invalid_interval(self):
        """Test initialization with invalid interval hours."""
        with self.assertRaises(DailyScheduleError):
            DailyScheduler(daily_duration_hours=8, interval_hours=200)  # > 168 hours

        with self.assertRaises(DailyScheduleError):
            DailyScheduler(
                daily_duration_hours=12, interval_hours=8
            )  # duration > interval

    def test_parse_time_string(self):
        """Test time string parsing."""
        scheduler = DailyScheduler(8, 24, "00:00:00")

        # Test valid times
        self.assertEqual(scheduler._parse_time_string("00:00:00"), 0)
        self.assertEqual(
            scheduler._parse_time_string("12:30:45"), 12 * 3600 + 30 * 60 + 45
        )
        self.assertEqual(
            scheduler._parse_time_string("23:59:59"), 23 * 3600 + 59 * 60 + 59
        )

        # Test invalid times
        with self.assertRaises(DailyScheduleError):
            scheduler._parse_time_string("25:00:00")  # Invalid hour
        with self.assertRaises(DailyScheduleError):
            scheduler._parse_time_string("12:60:00")  # Invalid minute
        with self.assertRaises(DailyScheduleError):
            scheduler._parse_time_string("invalid")  # Invalid format

    def test_is_active_period_daily_schedule(self):
        """Test daily schedule (24-hour intervals)."""
        # 8 hours active starting at 9 AM
        scheduler = DailyScheduler(8, 24, "09:00:00")

        # Create test timestamps for a specific day
        # Using January 1, 2024 (Monday) as reference
        base_date = 1704067200  # 2024-01-01 00:00:00 UTC

        # Test times
        test_8am = base_date + 8 * 3600  # 8:00 AM - should be inactive
        test_9am = base_date + 9 * 3600  # 9:00 AM - should be active (start)
        test_12pm = base_date + 12 * 3600  # 12:00 PM - should be active (middle)
        test_5pm = base_date + 17 * 3600  # 5:00 PM - should be active (end-1)
        test_6pm = base_date + 18 * 3600  # 6:00 PM - should be inactive (after end)

        self.assertFalse(scheduler.is_active_period(test_8am))
        self.assertTrue(scheduler.is_active_period(test_9am))
        self.assertTrue(scheduler.is_active_period(test_12pm))
        self.assertFalse(
            scheduler.is_active_period(test_5pm)
        )  # 5 PM = 17:00, end is at 17:00 (exclusive)
        self.assertFalse(scheduler.is_active_period(test_6pm))

    def test_is_active_period_twice_daily(self):
        """Test twice-daily schedule (12-hour intervals)."""
        # 4 hours active every 12 hours starting at 6 AM
        scheduler = DailyScheduler(4, 12, "06:00:00")

        base_date = 1704067200  # 2024-01-01 00:00:00 UTC

        # First period: 6 AM - 10 AM
        test_6am = base_date + 6 * 3600  # Should be active
        test_8am = base_date + 8 * 3600  # Should be active
        test_10am = base_date + 10 * 3600  # Should be inactive (end)
        test_12pm = base_date + 12 * 3600  # Should be inactive

        # Second period: 6 PM - 10 PM (18:00 - 22:00)
        test_6pm = base_date + 18 * 3600  # Should be active
        test_8pm = base_date + 20 * 3600  # Should be active
        test_10pm = base_date + 22 * 3600  # Should be inactive (end)

        self.assertTrue(scheduler.is_active_period(test_6am))
        self.assertTrue(scheduler.is_active_period(test_8am))
        self.assertFalse(scheduler.is_active_period(test_10am))
        self.assertFalse(scheduler.is_active_period(test_12pm))
        self.assertTrue(scheduler.is_active_period(test_6pm))
        self.assertTrue(scheduler.is_active_period(test_8pm))
        self.assertFalse(scheduler.is_active_period(test_10pm))

    def test_get_next_active_period(self):
        """Test getting next active period."""
        scheduler = DailyScheduler(8, 24, "09:00:00")

        base_date = 1704067200  # 2024-01-01 00:00:00 UTC
        test_8am = base_date + 8 * 3600  # Before active period

        next_start, next_end = scheduler.get_next_active_period(test_8am)

        expected_start = base_date + 9 * 3600  # 9 AM same day
        expected_end = base_date + 17 * 3600  # 5 PM same day

        self.assertEqual(next_start, expected_start)
        self.assertEqual(next_end, expected_end)

    def test_get_time_until_next_period(self):
        """Test getting time until next active period."""
        scheduler = DailyScheduler(8, 24, "09:00:00")

        base_date = 1704067200  # 2024-01-01 00:00:00 UTC
        test_8am = base_date + 8 * 3600  # 1 hour before active period

        time_until = scheduler.get_time_until_next_period(test_8am)
        self.assertEqual(time_until, 3600)  # 1 hour in seconds

    def test_get_remaining_active_time(self):
        """Test getting remaining time in active period."""
        scheduler = DailyScheduler(8, 24, "09:00:00")

        base_date = 1704067200  # 2024-01-01 00:00:00 UTC
        test_12pm = base_date + 12 * 3600  # Middle of active period (9 AM - 5 PM)

        remaining = scheduler.get_remaining_active_time(test_12pm)
        self.assertEqual(remaining, 5 * 3600)  # 5 hours remaining until 5 PM

        # Test inactive period
        test_8pm = base_date + 20 * 3600  # Outside active period
        remaining = scheduler.get_remaining_active_time(test_8pm)
        self.assertEqual(remaining, 0)

    def test_state_persistence(self):
        """Test state file persistence."""
        scheduler = DailyScheduler(
            daily_duration_hours=8,
            interval_hours=24,
            daily_start_time="09:00:00",
            state_file_path=self.state_file,
        )

        # Trigger state creation by checking active period
        base_date = 1704067200 + 9 * 3600  # 2024-01-01 09:00:00 UTC (active)
        scheduler.is_active_period(base_date)

        # Check that state file was created
        self.assertTrue(os.path.exists(self.state_file))

        # Load and verify state content
        with open(self.state_file) as f:
            state = json.load(f)

        self.assertIsInstance(state, dict)
        # Should have at least one period entry
        period_keys = [k for k in state.keys() if k.startswith("period_")]
        self.assertGreater(len(period_keys), 0)

    def test_get_schedule_info(self):
        """Test getting comprehensive schedule information."""
        scheduler = DailyScheduler(8, 24, "09:00:00")

        info = scheduler.get_schedule_info()

        # Check required fields
        required_fields = [
            "daily_duration_hours",
            "interval_hours",
            "daily_start_time",
            "currently_active",
            "next_period_start",
            "next_period_end",
        ]

        for field in required_fields:
            self.assertIn(field, info)

        self.assertEqual(info["daily_duration_hours"], 8)
        self.assertEqual(info["interval_hours"], 24)
        self.assertEqual(info["daily_start_time"], "09:00:00")


if __name__ == "__main__":
    unittest.main()
