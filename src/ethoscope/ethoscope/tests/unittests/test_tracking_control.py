"""
Unit tests for control/tracking.py.

Covers ExperimentalInformation and the ControlThread control plane:
option parsing, status/`info` reporting, light schedule file IO, target
detection, monitor setup (SQLite path), and graceful stop.

ControlThread.__init__ talks to real hardware/system services, so tests
build bare instances with ``object.__new__`` and only set the attributes
each method under test needs.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from ethoscope.control.tracking import ControlThread, ExperimentalInformation
from ethoscope.core.monitor import Monitor
from ethoscope.utils.debug import EthoscopeException

# ===========================================================================
# ExperimentalInformation
# ===========================================================================


class TestExperimentalInformation:
    def test_defaults(self):
        info = ExperimentalInformation()
        assert info.info_dic == {
            "name": "",
            "location": "",
            "code": "",
            "sensor": "",
            "lights_on": "",
            "lights_off": "",
            "light_period_minutes": 1440,
            "light_cycle_anchor": "",
        }

    def test_with_values(self):
        info = ExperimentalInformation(
            name="alice", location="inc1", code="EXP-42", sensor="http://sensor"
        )
        assert info.info_dic["name"] == "alice"
        assert info.info_dic["location"] == "inc1"
        assert info.info_dic["code"] == "EXP-42"
        assert info.info_dic["sensor"] == "http://sensor"

    def test_code_with_special_characters_raises(self):
        with pytest.raises(Exception, match="special characters"):
            ExperimentalInformation(code="bad code!")

    def test_code_allows_letters_digits_dash(self):
        info = ExperimentalInformation(code="A1-b2")
        assert info.info_dic["code"] == "A1-b2"


# ===========================================================================
# ControlThread helpers
# ===========================================================================

_GENERATED_THREADS = []


@pytest.fixture(autouse=True)
def _normalize_generated_threads():
    """Make generated ControlThread instances safe to garbage collect.

    ControlThread.__del__ calls stop() + shutil.rmtree(self._tmp_dir), so any
    leftover attribute (Mock monitor, missing keys) raises inside __del__ and
    triggers PytestUnraisableExceptionWarning. Reset state after each test.
    """
    yield
    while _GENERATED_THREADS:
        thread = _GENERATED_THREADS.pop()
        if isinstance(thread._info, dict):
            thread._info["status"] = "stopped"
            thread._info["error"] = None
            thread._info["time"] = 0
        else:
            thread._info = {"status": "stopped", "time": 0}
        thread._monit = None
        thread._metadata_cache = None
        thread._tracking_start_time = None
        thread._drawer = None
        thread._last_info_t_stamp = 0
        thread._last_info_frame_idx = 0
        thread._last_img_write_time = 0


def _make_control_thread():
    """Create a bare ControlThread with minimal working state.

    The instance must be safe to garbage collect: ControlThread.__del__
    calls stop() and shutil.rmtree(self._tmp_dir), so we provide a
    stopped status and a throwaway temp dir.
    """
    thread = object.__new__(ControlThread)
    thread._info = {"status": "stopped", "time": 0}
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
    thread._tmp_dir = tempfile.mkdtemp(prefix="ethoscope_ct_test_")
    _GENERATED_THREADS.append(thread)
    return thread


class TestControlThreadUserOptions:
    def test_user_options_returns_curated_classes(self):
        with patch(
            "ethoscope.utils.pi.isExperimental", return_value=False
        ), patch("ethoscope.utils.pi.isMachinePI", return_value=False):
            out = ControlThread.user_options()
            assert "interactor" in out
            assert "roi_builder" in out
            assert "result_writer" in out
            # each option lists class descriptions
            for desc in out["interactor"]:
                assert "name" in desc

    def test_user_options_experimental_includes_hidden(self):
        with patch(
            "ethoscope.utils.pi.isExperimental", return_value=True
        ), patch("ethoscope.utils.pi.isMachinePI", return_value=True):
            out = ControlThread.user_options()
            assert "camera" in out  # hidden options exposed for experimental


class TestControlThreadParseOptions:
    def test_parse_one_user_option_present(self):
        thread = _make_control_thread()
        data = {"interactor": {"name": "DefaultStimulator", "arguments": {"a": 1}}}
        cls, kwargs = thread._parse_one_user_option("interactor", data)
        assert cls is not None
        assert kwargs == {"a": 1}

    def test_parse_one_user_option_missing_field(self):
        thread = _make_control_thread()
        cls, kwargs = thread._parse_one_user_option("interactor", {})
        assert cls is None
        assert kwargs == {}

    def test_parse_user_options_none_data(self):
        thread = _make_control_thread()
        thread._option_dict = {"a": {"class": None, "kwargs": None, "possible_classes": [dict]}}
        thread._parse_user_options(None)  # should not raise

    def test_parse_user_options_sets_class(self):
        thread = _make_control_thread()
        thread._option_dict = {
            "interactor": {
                "possible_classes": [dict],
                "class": None,
                "kwargs": None,
            }
        }
        thread._parse_user_options(
            {"interactor": {"name": "dict", "arguments": {"x": 2}}}
        )
        assert thread._option_dict["interactor"]["class"] is dict
        assert thread._option_dict["interactor"]["kwargs"] == {"x": 2}

    def test_parse_user_options_missing_field_uses_default(self):
        thread = _make_control_thread()
        thread._option_dict = {
            "interactor": {
                "possible_classes": [dict],
                "class": None,
                "kwargs": None,
            }
        }
        thread._parse_user_options({})
        assert thread._option_dict["interactor"]["class"] is dict
        assert thread._option_dict["interactor"]["kwargs"] == {}


class TestControlThreadInfoAndStatus:
    def test_info_returns_dict(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped", "time": 0}
        assert thread.info == thread._info

    def test_info_recovers_corrupted_dict(self):
        thread = _make_control_thread()
        thread._info = "corrupted"
        result = thread.info
        assert isinstance(result, dict)
        assert result["error"] == "info corruption detected and recovered"

    def test_hw_info(self):
        thread = _make_control_thread()
        with (
            patch("ethoscope.utils.pi.pi_version", return_value="4"),
            patch("ethoscope.utils.pi.getPiCameraVersion", return_value="v3"),
            patch("ethoscope.utils.pi.get_SD_CARD_AGE", return_value=42),
            patch(
                "ethoscope.utils.pi.get_partition_info",
                return_value={"Use%": "50%"},
            ),
            patch("ethoscope.utils.pi.get_SD_CARD_NAME", return_value="SD1"),
        ):
            info = thread.hw_info
        assert info["pi_version"] == "4"
        assert info["camera"] == "v3"
        assert info["SD_CARD_AGE"] == 42
        assert info["SD_CARD_NAME"] == "SD1"

    def test_create_backup_filename(self):
        import datetime as _dt

        thread = _make_control_thread()
        ts = 1500000000
        thread._info = {"id": "M001", "time": ts}
        name = thread._create_backup_filename()
        expected_ts = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")
        assert name == f"{expected_ts}_M001.db"

    def test_was_interrupted_no_cache(self):
        thread = _make_control_thread()
        assert thread.was_interrupted is False

    def test_was_interrupted_graceful_stop(self, tmp_path):
        thread = _make_control_thread()
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({"stopped_gracefully": True}))
        mock_cache = Mock()
        mock_cache.list_cache_files.return_value = [{"path": str(cache_file)}]
        thread._metadata_cache = mock_cache
        assert thread.was_interrupted is False

    def test_was_interrupted_abrupt_stop(self, tmp_path):
        thread = _make_control_thread()
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({"stopped_gracefully": False}))
        mock_cache = Mock()
        mock_cache.list_cache_files.return_value = [{"path": str(cache_file)}]
        thread._metadata_cache = mock_cache
        assert thread.was_interrupted is True

    def test_was_interrupted_cache_error(self):
        thread = _make_control_thread()
        mock_cache = Mock()
        mock_cache.list_cache_files.side_effect = Exception("boom")
        thread._metadata_cache = mock_cache
        assert thread.was_interrupted is False


class TestControlThreadUpdateInfo:
    def test_update_info_no_monitor(self):
        thread = _make_control_thread()
        thread._info = {"monitor_info": None}
        thread._update_info()  # no-op when _monit is None

    def test_update_info_computes_fps(self):
        thread = _make_control_thread()
        thread._info = {"monitor_info": {}, "database_info": None}
        thread._monit = Mock()
        thread._monit.last_time_stamp = 1000
        thread._monit.last_frame_idx = 10
        thread._metadata_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {"db_status": "ok"}

        thread._update_info()
        assert thread._info["monitor_info"]["last_time_stamp"] == 1000
        assert thread._info["database_info"] == {"db_status": "ok"}

    def test_update_info_draws_frame_once_per_second(self):
        thread = _make_control_thread()
        thread._info = {
            "monitor_info": {},
            "database_info": None,
            "last_drawn_img": "/tmp/frame.jpg",
        }
        thread._monit = Mock()
        thread._monit.last_time_stamp = 1
        thread._monit.last_frame_idx = 0
        thread._metadata_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {}
        thread._drawer = Mock()
        thread._drawer.last_drawn_frame = np.zeros((10, 10, 3), np.uint8)
        thread._last_img_write_time = 0

        with patch(
            "ethoscope.control.tracking.cv2.imwrite"
        ) as mock_imwrite:
            thread._update_info()
        mock_imwrite.assert_called_once()
        assert thread._last_img_write_time > 0

    def test_update_info_result_writer_backup_filename(self):
        thread = _make_control_thread()
        thread._info = {"monitor_info": {}, "database_info": None}
        thread._monit = Mock()
        thread._monit.last_time_stamp = 1
        thread._monit.last_frame_idx = 0
        thread._monit._result_writer = Mock()
        thread._monit._result_writer.get_backup_filename.return_value = "b.db"
        thread._metadata_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {}

        thread._update_info()
        assert thread._info["backup_filename"] == "b.db"


# ===========================================================================
# Light schedule
# ===========================================================================


class TestLightSchedule:
    def _setup(self, tmp_path):
        thread = _make_control_thread()
        schedule_file = tmp_path / "light_schedule.json"
        patch.object(
            ControlThread, "LIGHT_SCHEDULE_FILE", str(schedule_file)
        ).start()
        self._patch = patch.object(
            ControlThread, "LIGHT_SCHEDULE_FILE", str(schedule_file)
        )
        self._patch.start()
        self._schedule_file = schedule_file
        return thread, schedule_file

    def teardown_method(self):
        if hasattr(self, "_patch"):
            self._patch.stop()

    def test_write_light_schedule_active(self, tmp_path):
        thread, schedule_file = self._setup(tmp_path)
        thread._info = {
            "experimental_info": {
                "lights_on": "06:00",
                "lights_off": "18:00",
                "light_period_minutes": "720",
                "light_cycle_anchor": "1700000000",
            }
        }
        thread._write_light_schedule()
        schedule = json.loads(schedule_file.read_text())
        assert schedule["active"] is True
        assert schedule["lights_on"] == "06:00"
        assert schedule["period_minutes"] == 720
        assert schedule["anchor"] == 1700000000

    def test_write_light_schedule_inactive_and_bad_values(self, tmp_path):
        thread, schedule_file = self._setup(tmp_path)
        thread._info = {
            "experimental_info": {
                "lights_on": "",
                "lights_off": "",
                "light_period_minutes": "not-a-number",
                "light_cycle_anchor": "bogus",
            }
        }
        thread._write_light_schedule()
        schedule = json.loads(schedule_file.read_text())
        assert schedule["active"] is False
        assert schedule["period_minutes"] == 1440
        assert schedule["anchor"] is None

    def test_write_light_schedule_zero_period_falls_back(self, tmp_path):
        thread, schedule_file = self._setup(tmp_path)
        thread._info = {
            "experimental_info": {
                "lights_on": "06:00",
                "lights_off": "18:00",
                "light_period_minutes": 0,
            }
        }
        thread._write_light_schedule()
        schedule = json.loads(schedule_file.read_text())
        assert schedule["period_minutes"] == 1440

    def test_clear_light_schedule(self, tmp_path):
        thread, schedule_file = self._setup(tmp_path)
        thread._clear_light_schedule()
        schedule = json.loads(schedule_file.read_text())
        assert schedule["active"] is False
        assert schedule["lights_on"] == ""

    def test_force_release_lights_no_hardware(self, tmp_path):
        thread = _make_control_thread()
        with patch(
            "ethoscope.utils.pi.has_light_hardware", return_value=False
        ):
            thread._force_lights_on_for_targets()
            thread._release_lights_after_targets()

    def test_force_lights_on_success(self, tmp_path):
        thread = _make_control_thread()
        mock_client = Mock()
        with (
            patch("ethoscope.utils.pi.has_light_hardware", return_value=True),
            patch(
                "ethoscope.hardware.interfaces.light_daemon.LightDaemonClient",
                return_value=mock_client,
            ),
        ):
            thread._force_lights_on_for_targets()
        mock_client.force_on.assert_called_once()


class TestControlThreadTimeoutHandler:
    def test_initialization_timeout_kills_process(self):
        thread = _make_control_thread()
        thread._info = {"status": "initialising", "error": None, "time": 0}
        with patch(
            "ethoscope.control.tracking.time.sleep"
        ), patch(
            "ethoscope.control.tracking.os.kill"
        ) as mock_kill:
            thread._initialization_timeout_handler()
        assert thread._info["status"] == "error"
        mock_kill.assert_called_once()

    def test_timeout_handler_ignores_running(self):
        thread = _make_control_thread()
        thread._info = {"status": "running"}
        with patch(
            "ethoscope.control.tracking.time.sleep"
        ), patch(
            "ethoscope.control.tracking.os.kill"
        ) as mock_kill:
            thread._initialization_timeout_handler()
        mock_kill.assert_not_called()


# ===========================================================================
# Target detection
# ===========================================================================


class TestTargetDetection:
    def test_detect_and_store_targets_success(self):
        thread = _make_control_thread()
        thread._info = {"experimental_info": {}}
        roi_builder = Mock()
        roi_builder.build.return_value = ([(1, 2), (3, 4)], ["roi1"])
        thread._option_dict = {"roi_builder": {"class": Mock(return_value=roi_builder), "kwargs": {}}}

        pts, rois = thread._detect_and_store_targets(Mock())
        assert pts == [(1, 2), (3, 4)]
        assert rois == ["roi1"]
        assert thread._info["experimental_info"]["target_coordinates"] == [
            [1.0, 2.0],
            [3.0, 4.0],
        ]

    def test_detect_and_store_targets_failure_returns_none(self):
        thread = _make_control_thread()
        thread._info = {"experimental_info": {}}
        roi_builder = Mock()
        roi_builder.build.return_value = (None, None)
        thread._option_dict = {
            "roi_builder": {"class": Mock(return_value=roi_builder), "kwargs": {}}
        }
        with patch.object(thread, "_save_roi_debug_image") as mock_save:
            result = thread._detect_and_store_targets(Mock())
        assert result == (None, None)
        mock_save.assert_called_once()

    def test_detect_and_store_targets_exception_returns_none(self):
        thread = _make_control_thread()
        thread._info = {"experimental_info": {}}
        roi_builder = Mock()
        roi_builder.build.side_effect = EthoscopeException("bad arena")
        thread._option_dict = {
            "roi_builder": {"class": Mock(return_value=roi_builder), "kwargs": {}}
        }
        with patch.object(thread, "_save_roi_debug_image") as mock_save:
            result = thread._detect_and_store_targets(Mock())
        assert result == (None, None)
        mock_save.assert_called_once()

    def test_save_roi_debug_image_grayscale(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped", "dbg_img": "/tmp/debug.png"}
        cam = Mock()
        frame = np.zeros((10, 10), np.uint8)
        cam.__iter__ = Mock(return_value=iter([(0, frame)]))
        with patch(
            "ethoscope.control.tracking.cv2.imwrite"
        ) as mock_write, patch(
            "ethoscope.control.tracking.cv2.cvtColor", return_value=frame
        ):
            thread._save_roi_debug_image(cam, "error message")
        mock_write.assert_called_once()

    def test_save_roi_debug_image_handles_exception(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped", "dbg_img": "/tmp/debug.png"}
        cam = Mock()
        cam.__iter__ = Mock(side_effect=Exception("no frames"))
        thread._save_roi_debug_image(cam, "boom")  # should not raise


# ===========================================================================
# Monitor setup
# ===========================================================================


class TestStartTracking:
    def test_start_tracking_runs_monitor(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped"}
        thread._monit_kwargs = {}
        thread._metadata = {"date_time": 12345}
        thread._metadata_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {"db_status": "ok"}

        stimulator = Mock()
        stimulator_class = Mock(return_value=stimulator)
        hardware_connection = Mock()
        monitor_instance = Mock()
        with patch(
            "ethoscope.control.tracking.Monitor", return_value=monitor_instance
        ) as mock_monitor:
            thread._start_tracking(
                camera=Mock(),
                result_writer=Mock(),
                rois=[Mock(), Mock()],
                reference_points=[(1, 1)],
                TrackerClass=Mock(),
                tracker_kwargs={},
                hardware_connection=hardware_connection,
                StimulatorClass=stimulator_class,
                stimulator_kwargs={},
            )

        assert stimulator_class.call_count == 2  # one per ROI
        mock_monitor.assert_called_once()
        monitor_instance.run.assert_called_once()
        assert thread._info["status"] == "running"
        assert thread._tracking_start_time == 12345

    def test_start_tracking_falls_back_to_default_stimulator(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped"}
        thread._monit_kwargs = {}
        thread._metadata_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {}

        stimulator_class = Mock(side_effect=Exception("no hardware"))
        monitor_instance = Mock()
        with (
            patch(
                "ethoscope.control.tracking.Monitor",
                return_value=monitor_instance,
            ),
            patch(
                "ethoscope.stimulators.stimulators.DefaultStimulator"
            ) as mock_default,
        ):
            thread._start_tracking(
                camera=Mock(),
                result_writer=Mock(),
                rois=[Mock()],
                reference_points=[],
                TrackerClass=Mock(),
                tracker_kwargs={},
                hardware_connection=Mock(),
                StimulatorClass=stimulator_class,
                stimulator_kwargs={},
            )
        mock_default.assert_called_once()


class TestSetTrackingFromScratch:
    def _option_dict(self, roi_builder_class=None):
        interactor_class = Mock()
        interactor_class.__dict__["_HardwareInterfaceClass"] = Mock()

        return {
            "camera": {"class": Mock(), "kwargs": {}},
            "interactor": {
                "class": interactor_class,
                "kwargs": {},
            },
            "tracker": {"class": Mock(), "kwargs": {}},
            "result_writer": {"class": Mock(), "kwargs": {}},
            "drawer": {"class": Mock(return_value=Mock()), "kwargs": {}},
            "experimental_info": {"class": Mock(), "kwargs": {}},
            "roi_builder": {
                "class": roi_builder_class or Mock(),
                "kwargs": {},
            },
        }

    def test_detection_failure_returns_none(self):
        thread = _make_control_thread()
        cam = Mock()
        cam._close = Mock()
        camera_class = Mock(return_value=cam)
        thread._option_dict = self._option_dict()
        thread._option_dict["camera"]["class"] = camera_class

        with (
            patch.object(
                thread, "_detect_and_store_targets", return_value=(None, None)
            ),
            patch.object(thread, "_force_lights_on_for_targets"),
            patch.object(thread, "_release_lights_after_targets"),
            patch("ethoscope.control.tracking.time.sleep"),
        ):
            result = thread._set_tracking_from_scratch()

        assert result is None
        cam._close.assert_called_once()

    def test_sqlite_path_success(self):
        thread = _make_control_thread()
        thread._info = {
            "status": "initialising",
            "id": "M001",
            "name": "etho1",
            "version": {"id": "v1"},
            "time": time.time(),
            "experimental_info": {"name": "u", "location": "l", "sensor": ""},
            "backup_filename": "2025-01-01_00-00-00_M001.db",
        }
        cam = Mock()
        cam.start_time = 1234
        cam.width = 960
        cam.height = 720
        camera_class = Mock(return_value=cam)

        stimulator_class = Mock()
        stimulator_class.__dict__["_HardwareInterfaceClass"] = Mock()

        result_writer = Mock()
        result_writer.get_backup_filename.return_value = "w.db"
        result_writer_class = Mock(return_value=result_writer)
        result_writer_class._database_type = "SQLite3"

        exp_info_instance = Mock()
        exp_info_instance.info_dic = {"name": "u", "location": "l", "sensor": ""}
        exp_info_class = Mock(return_value=exp_info_instance)

        mock_metadata_cache = Mock()
        mock_metadata_cache.get_database_info.return_value = {"db_status": "ok"}
        thread._cache_dir = "/tmp"

        thread._option_dict = self._option_dict()
        thread._option_dict["camera"]["class"] = camera_class
        thread._option_dict["interactor"]["class"] = stimulator_class
        thread._option_dict["result_writer"]["class"] = result_writer_class
        thread._option_dict["result_writer"]["kwargs"] = {}
        thread._option_dict["interactor"]["kwargs"] = {}
        thread._option_dict["experimental_info"]["class"] = exp_info_class

        with (
            patch.object(
                thread,
                "_detect_and_store_targets",
                return_value=([(1, 2), (3, 4), (5, 6)], [Mock()]),
            ),
            patch.object(thread, "_force_lights_on_for_targets"),
            patch.object(thread, "_release_lights_after_targets"),
            patch.object(thread, "_write_light_schedule"),
            patch.object(
                type(thread),
                "hw_info",
                new_callable=PropertyMock,
                return_value={"kernel": "x"},
            ),
            patch(
                "ethoscope.control.tracking.create_metadata_cache",
                return_value=mock_metadata_cache,
            ),
            patch("ethoscope.control.tracking.HardwareConnection"),
            patch("ethoscope.control.tracking.os.makedirs"),
        ):
            result = thread._set_tracking_from_scratch()

        assert result is not None
        cam, rw, rois, refs, tracker, kwargs, hc, stim, stim_kwargs, offset = result
        assert rw is result_writer
        assert thread._metadata_cache is mock_metadata_cache
        # backup filename refreshed from the result writer
        assert thread._info["backup_filename"] == "w.db"
        assert thread._info["status"] == "initialising"


class TestControlThreadStop:
    def test_stop_running_tracking(self):
        thread = _make_control_thread()
        thread._info = {
            "status": "running",
            "time": 0,
            "experimental_info": {"run_id": "abc123", "name": "u"},
            "backup_filename": "b.db",
        }
        thread._default_monitor_info = {"last_positions": None}
        thread._tracking_start_time = 100
        thread._metadata_cache = Mock()
        thread._metadata_cache.finalize_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {"db_status": "ok"}
        monitor = Mock()
        monitor.last_frame_idx = 0
        monitor.last_time_stamp = 0
        monitor._result_writer = None
        thread._monit = monitor

        with patch.object(thread, "_clear_light_schedule"):
            thread.stop()

        assert thread._info["status"] == "stopped"
        assert thread._info["error"] is None
        monitor.stop.assert_called_once()
        assert thread._monit is None
        thread._metadata_cache.finalize_cache.assert_called_once()
        assert thread._info["database_info"] == {"db_status": "ok"}
        # run_id preserved for the next experiment (under "current")
        assert (
            thread._info["experimental_info"]["current"]["run_id"] == "abc123"
        )

    def test_stop_with_error_and_backup(self):
        thread = _make_control_thread()
        thread._info = {
            "status": "running",
            "time": 0,
            "experimental_info": {"run_id": "abc"},
            "backup_filename": "b.db",
        }
        thread._default_monitor_info = {}
        thread._tracking_start_time = 100
        thread._metadata_cache = Mock()
        thread._metadata_cache.finalize_cache = Mock()
        thread._metadata_cache.get_database_info.return_value = {}
        monitor = Mock()
        monitor.last_frame_idx = 0
        monitor.last_time_stamp = 0
        monitor._result_writer = None
        thread._monit = monitor

        with patch.object(thread, "_clear_light_schedule"):
            thread.stop(error="boom")

        assert thread._info["status"] == "stopped"
        assert thread._info["error"] == "boom"
        # previous experiment info is archived under experimental_info.previous
        assert thread._info["experimental_info"]["previous"]["backup_filename"] == "b.db"
        thread._metadata_cache.finalize_cache.assert_called_once_with(
            thread._tracking_start_time, graceful=False, stop_reason="error"
        )

    def test_stop_when_not_running_is_noop(self):
        thread = _make_control_thread()
        thread._info = {"status": "stopped", "time": 0}
        with patch.object(thread, "_clear_light_schedule") as mock_clear:
            thread.stop()
        mock_clear.assert_not_called()
        assert thread._info["status"] == "stopped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
