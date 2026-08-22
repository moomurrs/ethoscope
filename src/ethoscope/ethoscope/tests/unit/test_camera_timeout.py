#!/usr/bin/env python3
"""
Unit tests for camera initialization watchdog.

Tests the process-level failsafe that prevents ethoscope from hanging
indefinitely during camera initialization. Picamera2-only, single-attempt
fail-fast — no legacy ``picamera`` fallback retry logic.
"""

import signal
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from ethoscope.control.tracking import ControlThread


def _make_bare_thread(initial_status="initialising"):
    thread = object.__new__(ControlThread)
    thread._info = {"status": initial_status, "error": None, "time": 0}
    thread._tmp_dir = tempfile.mkdtemp(prefix="ethoscope_watchdog_test_")
    thread._monit = None
    thread._metadata = None
    thread._metadata_cache = None
    thread._tracking_start_time = None
    thread._monit_args = ()
    thread._monit_kwargs = {}
    thread._last_info_t_stamp = 0
    thread._last_info_frame_idx = 0
    thread._last_img_write_time = 0
    thread._drawer = None
    thread._default_monitor_info = {"last_positions": None, "last_time_stamp": 0}
    return thread


class TestCameraTimeoutMechanisms:
    """Test suite for camera initialization watchdog (real ControlThread)."""

    def test_timeout_handler_sets_error_and_kills_when_initialising(self):
        thread = _make_bare_thread("initialising")
        with (
            patch("ethoscope.control.tracking.time.sleep") as mock_sleep,
            patch("ethoscope.control.tracking.os.kill") as mock_kill,
            patch("ethoscope.control.tracking.os.getpid", return_value=1234),
        ):
            thread._initialization_timeout_handler()

        mock_sleep.assert_called_once_with(120)
        assert thread._info["status"] == "error"
        assert "Initialization timeout" in thread._info["error"]
        assert "Process terminated after 2 minutes" in thread._info["error"]
        mock_kill.assert_called_once_with(1234, signal.SIGKILL)

    def test_timeout_handler_ignores_when_not_initialising(self):
        for status in ("running", "stopped", "error"):
            thread = _make_bare_thread(status)
            with (
                patch("ethoscope.control.tracking.time.sleep"),
                patch("ethoscope.control.tracking.os.kill") as mock_kill,
            ):
                thread._initialization_timeout_handler()
            assert thread._info["status"] == status
            mock_kill.assert_not_called()

    def test_timeout_handler_sleeps_120_seconds(self):
        thread = _make_bare_thread("initialising")
        with (
            patch("ethoscope.control.tracking.time.sleep") as mock_sleep,
            patch("ethoscope.control.tracking.os.kill"),
            patch("ethoscope.control.tracking.os.getpid", return_value=1),
        ):
            thread._initialization_timeout_handler()
        mock_sleep.assert_called_once_with(120)

    def test_run_starts_watchdog_daemon_thread(self):
        thread = _make_bare_thread("stopped")
        # Minimal stubs to avoid full ControlThread.run execution
        # We patch _initialization_timeout_handler to avoid sleep/kill
        with (
            patch("ethoscope.control.tracking.threading.Thread") as mock_thread_cls,
            patch.object(thread, "_initialization_timeout_handler"),
            patch.object(thread, "_set_tracking_from_scratch", return_value=None),
        ):
            mock_instance = Mock()
            mock_thread_cls.return_value = mock_instance
            thread.run()
            # Verify watchdog thread was created with daemon=True
            mock_thread_cls.assert_called()
            kwargs = (
                mock_thread_cls.call_args[1] if mock_thread_cls.call_args[1] else {}
            )
            args = mock_thread_cls.call_args[0] if mock_thread_cls.call_args[0] else ()
            # Check daemon=True and target is handler
            assert kwargs.get("daemon") is True
            assert kwargs.get("target") == thread._initialization_timeout_handler
            mock_instance.start.assert_called_once()

    def test_timeout_handler_updates_time(self):
        thread = _make_bare_thread("initialising")
        before = time.time()
        with (
            patch("ethoscope.control.tracking.time.sleep"),
            patch("ethoscope.control.tracking.os.kill"),
            patch("ethoscope.control.tracking.os.getpid", return_value=1),
        ):
            thread._initialization_timeout_handler()
        assert thread._info["time"] >= before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
