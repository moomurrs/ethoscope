"""
Unit tests for control/record.py.

Covers the video recording / streaming control plane:
  * timedStop countdown parsing
  * cameraCaptureThread (camera setup, chunk naming, preview, local
    recording and streaming loops) with a mocked camera
  * GeneralVideoRecorder and its presets (Standard/HD) + Streamer
  * ControlThreadVideoRecording lifecycle (option parsing, status, stop)
"""

import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

import ethoscope.control.record as record_mod
from ethoscope.control.record import (
    ControlThreadVideoRecording,
    GeneralVideoRecorder,
    HDVideoRecorder,
    StandardVideoRecorder,
    Streamer,
    cameraCaptureThread,
    timedStop,
)
from ethoscope.utils.debug import EthoscopeException

# ===========================================================================
# timedStop
# ===========================================================================


class TestTimedStop:
    def test_zero_timer(self):
        ts = timedStop(timer="00:00:00")
        assert ts.countdown == 0
        assert ts.autostop is False

    def test_conversion(self):
        ts = timedStop(timer="01:02:03")
        assert ts.countdown == 93780  # 1d + 2h + 3m
        assert ts.autostop is True

    def test_multiple_days(self):
        ts = timedStop(timer="02:00:00")
        assert ts.countdown == 172800

    def test_bad_format(self):
        with pytest.raises(ValueError, match="DD:HH:MM"):
            timedStop(timer="12:00")

    def test_hours_out_of_range(self):
        with pytest.raises(ValueError, match="countdown format"):
            timedStop(timer="00:24:00")

    def test_minutes_out_of_range(self):
        with pytest.raises(ValueError, match="countdown format"):
            timedStop(timer="00:00:60")


# ===========================================================================
# cameraCaptureThread
# ===========================================================================


class _FakeCamera:
    """Duck-typed camera with the attributes cameraCaptureThread needs."""

    def __init__(self, is_pi=True, frames=None):
        self.hardware_recording = is_pi
        self.fps = 15.0
        self.width = 640
        self.height = 480
        self._frames = frames or []

    def __iter__(self):
        yield from self._frames
        # Signal end-of-stream so the run loop terminates.
        getattr(self, "on_stream_end", lambda: None)()

    def _close(self):
        pass


class _BoundedCamera(_FakeCamera):
    """Camera that flips a shared stop flag when its frames run out."""

    def __init__(self, holder, frames=None, is_pi=True):
        super().__init__(is_pi=is_pi, frames=frames)
        self._holder = holder

    def __iter__(self):
        for i, frame in enumerate(self._frames):
            yield float(i) / self.fps, frame
        self._holder.stop_camera_activity = True


class TestCameraCaptureThread:
    def test_init_creates_camera(self):
        camera_class = Mock(return_value=_FakeCamera())
        thread = cameraCaptureThread(
            camera_class,
            {},
            "/tmp/preview.jpg",
            "/tmp/video",
            width=640,
            height=480,
            fps=15,
            bitrate=1000,
            quality=20,
        )
        camera_class.assert_called_once()
        assert thread._resolution == (640, 480)
        assert thread._local_recording is False  # Pi camera records itself
        thread.camera._close()

    def test_init_local_recording_for_v4l2(self):
        camera_class = Mock(return_value=_FakeCamera(is_pi=False))
        thread = cameraCaptureThread(
            camera_class,
            {},
            "/tmp/preview.jpg",
            "/tmp/video",
            width=640,
            height=480,
            fps=15,
            bitrate=1000,
            quality=20,
            record_video=True,
        )
        assert thread._local_recording is True
        thread.camera._close()

    def test_init_no_camera_hardware_raises(self):
        def raising(*args, **kwargs):
            raise EthoscopeException("Camera hardware not available")

        with pytest.raises(EthoscopeException, match="Recording disabled"):
            cameraCaptureThread(
                Mock(side_effect=raising),
                {},
                "/tmp/preview.jpg",
                "/tmp/video",
                width=640,
                height=480,
                fps=15,
                bitrate=1000,
                quality=20,
            )

    def test_init_other_camera_error_re_raised(self):
        def raising(*args, **kwargs):
            raise EthoscopeException("sensor exploded")

        with pytest.raises(EthoscopeException, match="sensor exploded"):
            cameraCaptureThread(
                Mock(side_effect=raising),
                {},
                "/tmp/preview.jpg",
                "/tmp/video",
                width=640,
                height=480,
                fps=15,
                bitrate=1000,
                quality=20,
            )

    def test_get_video_chunk_filename(self):
        camera = _FakeCamera()
        camera.fps = 25.0
        thread = object.__new__(cameraCaptureThread)
        thread._resolution = (960, 720)
        thread.camera = camera
        thread._video_prefix = "/tmp/chunk"
        thread.video_file_index = 0

        name = thread._get_video_chunk_filename()
        assert name == "/tmp/chunk_960x720@25.0_00001.h264"
        assert thread.video_file_index == 1

    def test_create_recording_folder(self, tmp_path):
        video_dir = tmp_path / "nested" / "dir"
        thread = object.__new__(cameraCaptureThread)
        thread._video_prefix = str(video_dir / "chunk")
        thread._create_recording_folder()
        assert video_dir.exists()

    def test_save_preview_frame(self, tmp_path):
        camera = _FakeCamera()
        camera.fps = 15.0
        thread = object.__new__(cameraCaptureThread)
        thread.camera = camera
        img_path = str(tmp_path / "preview.jpg")
        thread._img_path = img_path

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        thread._save_preview_frame(frame, "PI Recording")
        assert os.path.exists(img_path)

    def test_run_local_recording(self):
        thread = object.__new__(cameraCaptureThread)
        thread._frames = None
        thread.camera = _BoundedCamera(holder=thread)
        thread.camera._frames = [np.zeros((480, 640), np.uint8)] * 3
        thread.camera.hardware_recording = False
        thread._resolution = (640, 480)
        thread._img_path = "/tmp/preview.jpg"
        thread._local_recording = True
        thread._record_video = True
        thread._stream = False
        thread._video_prefix = "/tmp/video"
        thread.stop_camera_activity = False
        thread.video_file_index = 0
        thread.preview_time = time.time() - 60  # previews due immediately
        thread.start_time = time.time() - 301  # chunk timer already elapsed
        thread._fps = 15
        thread._bitrate = 1000
        thread._quality = 20

        writer = Mock()
        writer.isOpened.return_value = True

        with (
            patch.object(
                thread, "_get_video_chunk_filename", return_value="/tmp/out.h264"
            ),
            patch.object(record_mod.cv2, "VideoWriter", return_value=writer),
            patch.object(record_mod.cv2, "VideoWriter_fourcc"),
        ):
            thread.run()

        assert writer.write.called
        writer.release.assert_called()
        assert thread.stop_camera_activity is True

    def test_run_stream(self):
        thread = object.__new__(cameraCaptureThread)
        thread.camera = _BoundedCamera(holder=thread)
        thread.camera._frames = [
            np.zeros((480, 640, 3), np.uint8),
            np.zeros((480, 640, 3), np.uint8),
        ]
        thread.camera.hardware_recording = True
        thread._img_path = "/tmp/preview.jpg"
        thread._local_recording = False
        thread._record_video = False
        thread._stream = True
        thread._video_prefix = None
        thread.stop_camera_activity = False
        thread.preview_time = time.time()

        client_socket = Mock()
        client_socket.sendall = Mock()
        server_socket = Mock()
        server_socket.accept.return_value = (client_socket, "addr")

        with (
            patch.object(
                record_mod.socket,
                "socket",
                return_value=server_socket,
            ),
            patch.object(record_mod.cv2, "imencode", return_value=(True, b"jpeg")),
        ):
            thread.run()

        assert server_socket.bind.called
        assert server_socket.listen.called
        assert client_socket.sendall.called
        client_socket.close.assert_called_once()
        server_socket.close.assert_called_once()


# ===========================================================================
# Recorder presets
# ===========================================================================


class TestRecorders:
    def test_general_recorder_uses_capture_thread(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = GeneralVideoRecorder(
            camera_class,
            {},
            img_path="/tmp/preview.jpg",
            video_prefix="/tmp/video",
            width=1280,
            height=960,
            fps=15,
        )
        assert isinstance(recorder._p, cameraCaptureThread)
        recorder._p.camera._close()

    def test_standard_recorder_preset(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = StandardVideoRecorder(
            camera_class, {}, video_prefix="/tmp/video", img_path="/tmp/preview.jpg"
        )
        assert recorder._p._resolution == (1280, 960)
        assert recorder._p._fps == 15
        assert recorder.status == "recording"
        recorder._p.camera._close()

    def test_hd_recorder_preset(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = HDVideoRecorder(
            camera_class, {}, video_prefix="/tmp/video", img_path="/tmp/preview.jpg"
        )
        assert recorder._p._resolution == (1920, 1088)
        assert recorder._p._fps == 15
        recorder._p.camera._close()

    def test_streamer_preset(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = Streamer(
            camera_class, {}, video_prefix="/tmp/video", img_path="/tmp/preview.jpg"
        )
        assert recorder._p._stream is True
        assert recorder._p._record_video is False
        assert recorder._p._resolution == (960, 720)
        recorder._p.camera._close()

    def test_start_and_stop_recorder(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = GeneralVideoRecorder(
            camera_class,
            {},
            img_path="/tmp/preview.jpg",
            video_prefix="/tmp/video",
        )
        with patch.object(recorder._p, "start") as mock_start:
            recorder.start_recording()
        mock_start.assert_called_once()

        with patch.object(recorder._p, "join") as mock_join:
            recorder.stop()
        assert recorder._p.stop_camera_activity is True
        mock_join.assert_called_once()

    def test_stop_closes_stream_connection(self):
        camera_class = Mock(return_value=_FakeCamera())
        recorder = Streamer(
            camera_class, {}, video_prefix="/tmp/video", img_path="/tmp/preview.jpg"
        )
        mock_conn = Mock()
        recorder._p.connection = mock_conn
        with patch.object(recorder._p, "join"):
            recorder.stop()
        mock_conn.close.assert_called_once()
        recorder._p.camera._close()


# ===========================================================================
# ControlThreadVideoRecording
# ===========================================================================


class TestControlThreadVideoRecording:
    def _bare_thread(self):
        thread = object.__new__(ControlThreadVideoRecording)
        thread._info = {
            "status": "stopped",
            "time": 0,
            "last_drawn_img": "/tmp/last_img.jpg",
            "dbg_img": "/tmp/dbg.png",
            "log_file": "/tmp/ethoscope.log",
        }
        thread._recorder = None
        thread._machine_id = "M001"
        thread._device_name = "etho1"
        thread._video_root_dir = "/tmp"
        thread._tmp_dir = tempfile.mkdtemp(prefix="ethoscope_rec_test_")
        thread._last_info_t_stamp = 0
        thread._last_info_frame_idx = 0
        return thread

    def test_controltype(self):
        thread = self._bare_thread()
        assert thread.controltype == "recording"

    def test_update_info_no_recorder(self):
        thread = self._bare_thread()
        thread._update_info()  # no-op without recorder

    def test_parse_one_user_option_present(self):
        thread = self._bare_thread()
        data = {"recorder": {"name": "StandardVideoRecorder", "arguments": {"a": 1}}}
        cls, kwargs = thread._parse_one_user_option("recorder", data)
        assert cls is StandardVideoRecorder
        assert kwargs == {"a": 1}

    def test_parse_one_user_option_missing(self):
        thread = self._bare_thread()
        cls, kwargs = thread._parse_one_user_option("recorder", {})
        assert cls is None
        assert kwargs == {}

    def test_stop_cleans_up_recorder(self):
        thread = self._bare_thread()
        thread._info = {
            "status": "recording",
            "time": 0,
            "experimental_info": {"code": "X"},
        }
        recorder = Mock()
        thread._recorder = recorder

        with patch.object(thread, "_clear_light_schedule"):
            thread.stop()

        assert thread._info["status"] == "stopped"
        recorder.stop.assert_called_once()
        assert thread._recorder is None
        assert thread._info["error"] is None

    def test_stop_with_error(self):
        thread = self._bare_thread()
        thread._info = {"status": "recording", "time": 0, "experimental_info": {}}
        thread._recorder = None
        with patch.object(thread, "_clear_light_schedule"):
            thread.stop(error="boom")
        assert thread._info["status"] == "stopped"
        assert thread._info["error"] == "boom"

    def test_run_full_recording(self):
        thread = self._bare_thread()
        thread._info = {
            "status": "stopped",
            "time": time.time(),
            "experimental_info": {"code": "EXP1"},
            "last_drawn_img": "/tmp/last_img.jpg",
        }
        exp_info = Mock()
        exp_info.info_dic = {"code": "EXP1"}
        recorder_instance = Mock()
        recorder_instance.status = "recording"
        thread._option_dict = {
            "experimental_info": {"class": Mock(return_value=exp_info), "kwargs": {}},
            "recorder": {
                "class": Mock(return_value=recorder_instance),
                "kwargs": {},
            },
            "time_control": {"class": timedStop, "kwargs": {}},
            "camera": {"class": Mock(), "kwargs": {}},
        }

        with (
            patch.object(thread, "_write_light_schedule") as mock_write_light,
            patch.object(thread, "stop") as mock_stop,
        ):
            thread.run()

        mock_write_light.assert_called_once()
        mock_stop.assert_not_called()  # autostop timer is 0
        assert thread._recorder is not None
        assert thread._info["status"] == "recording"

    def test_run_no_camera_hardware(self):
        thread = self._bare_thread()
        thread._info = {
            "status": "stopped",
            "time": time.time(),
            "experimental_info": {"code": "EXP1"},
        }
        exp_info = Mock()
        exp_info.info_dic = {"code": "EXP1"}

        def raising_recorder(*args, **kwargs):
            raise EthoscopeException("Camera hardware not available")

        thread._option_dict = {
            "experimental_info": {"class": Mock(return_value=exp_info), "kwargs": {}},
            "recorder": {"class": Mock(side_effect=raising_recorder), "kwargs": {}},
            "time_control": {"class": timedStop, "kwargs": {}},
            "camera": {"class": Mock(), "kwargs": {}},
        }

        with (
            patch.object(thread, "_write_light_schedule"),
            patch.object(thread, "stop") as mock_stop,
        ):
            thread.run()

        mock_stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
