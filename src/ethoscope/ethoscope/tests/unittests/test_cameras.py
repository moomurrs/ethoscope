"""
Unit tests for hardware/input/cameras.py.

Tests the camera abstraction layer without real hardware:
  * BaseCamera frame iteration/dropping contract
  * MovieVirtualCamera (real mp4 playback via OpenCV)
  * V4L2Camera (mocked capture device)
  * PiFrameGrabber (picamera2) failure signalling (no camera hardware)
  * OurPiCameraAsync lifecycle helpers (state, queue, cleanup)

Camera hardware paths (PiFrameGrabber recording, OurPiCameraAsync real
initialization) are exercised up to the point where they must talk to
picamera2, which is stubbed out by the root conftest to signal "no camera".
"""

import os
import queue
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

import ethoscope.hardware.input.cameras as cameras
from ethoscope.hardware.input.cameras import (
    BaseCamera,
    MovieVirtualCamera,
    OurPiCameraAsync,
    PiFrameGrabber,
    V4L2Camera,
)
from ethoscope.utils.debug import EthoscopeException

TEST_VIDEO = str(
    Path(__file__).parent.parent
    / "static_files"
    / "videos"
    / "arena_10x2_sortTubes.mp4"
)


# ===========================================================================
# BaseCamera
# ===========================================================================


class _IterCamera(BaseCamera):
    """Minimal BaseCamera subclass that yields synthetic frames."""

    def __init__(self, frames, drop_each=1, max_duration=None, opened=True):
        super().__init__(drop_each=drop_each, max_duration=max_duration)
        self._frames = list(frames)
        self._opened = opened
        self._resolution = (10, 10)

    def is_last_frame(self):
        return self._frame_idx >= len(self._frames)

    def _next_image(self):
        if self._frame_idx >= len(self._frames):
            return None
        return self._frames[self._frame_idx]

    def _time_stamp(self):
        return float(self._frame_idx) / 30.0

    def is_opened(self):
        return self._opened

    def restart(self):
        self._frame_idx = 0


class TestBaseCamera:
    def test_init_stores_drop_each_and_max_duration(self):
        cam = BaseCamera(drop_each=3, max_duration=12.5)
        assert cam._drop_each == 3
        assert cam._max_duration == 12.5

    def test_exit_closes_camera(self):
        cam = _IterCamera([])
        with patch.object(cam, "_close") as mock_close:
            cam.__exit__()
        mock_close.assert_called_once()

    def test_base_close_is_noop(self):
        cam = BaseCamera()
        cam._close()  # should not raise

    def test_abstract_methods_raise(self):
        cam = BaseCamera()
        for call in (
            lambda: cam.is_last_frame(),
            lambda: cam._next_image(),
            lambda: cam._time_stamp(),
            lambda: cam.is_opened(),
            lambda: cam.restart(),
        ):
            with pytest.raises(NotImplementedError):
                call()

    def test_resolution_width_height_properties(self):
        cam = object.__new__(BaseCamera)
        cam._resolution = (640, 480)
        assert cam.resolution == (640, 480)
        assert cam.width == 640
        assert cam.height == 480

    def test_next_time_image_increments_frame_idx(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)])
        t, img = cam._next_time_image()
        assert cam._frame_idx == 1
        assert isinstance(t, float)
        assert img is not None

    def test_iter_yields_frames_with_ms_timestamps(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 3)
        out = list(cam)
        assert len(out) == 3
        for t_ms, frame in out:
            assert isinstance(t_ms, int)
            assert isinstance(frame, np.ndarray)
        # frame indices advance on each underlying read
        assert cam._frame_idx == 3

    def test_iter_drops_frames_per_drop_each(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 4, drop_each=2)
        out = list(cam)
        # frames at frame_idx 2 and 4 are yielded (index % 2 == 0)
        assert len(out) == 2

    def test_iter_respects_max_duration(self):
        # t = frame_idx / 30; stop when t > 0.05 -> ~2 frames
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 20, max_duration=0.05)
        out = list(cam)
        assert 0 < len(out) < 20

    def test_iter_stops_when_frame_is_none(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8), None])
        out = list(cam)
        assert len(out) == 1

    def test_iter_raises_when_closed_before_first_frame(self):
        cam = _IterCamera([], opened=False)
        with pytest.raises(EthoscopeException):
            list(cam)


# ===========================================================================
# MovieVirtualCamera
# ===========================================================================


@pytest.fixture
def video_path():
    if not os.path.exists(TEST_VIDEO):
        pytest.skip("test video not available")
    return TEST_VIDEO


class TestMovieVirtualCamera:
    def test_init_reads_video_metadata(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam._resolution == (1280, 960)
        assert cam._total_n_frames == 1200
        assert cam._has_end_of_file is True
        assert cam.path == video_path
        assert cam.start_time == 0
        assert cam.canbepickled is False
        assert cam.isPiCamera is True

    def test_init_wall_clock_start_time(self, video_path):
        before = time.time()
        cam = MovieVirtualCamera(video_path, use_wall_clock=True)
        assert before <= cam.start_time <= time.time()
        cam._close()

    def test_init_missing_path_raises(self, tmp_path):
        with pytest.raises(EthoscopeException):
            MovieVirtualCamera(str(tmp_path / "missing.mp4"))

    def test_init_non_string_path_raises(self):
        with pytest.raises(EthoscopeException):
            MovieVirtualCamera(12345)

    def test_is_opened(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam.is_opened() is True
        cam._close()

    def test_restart_reopens(self, video_path):
        cam = MovieVirtualCamera(video_path)
        cam._frame_idx = 100
        cam.restart()
        assert cam._frame_idx == 0
        assert cam.is_opened() is True
        cam._close()

    def test_next_image_returns_grayscale(self, video_path):
        cam = MovieVirtualCamera(video_path)
        frame = cam._next_image()
        assert frame.ndim == 2
        assert frame.shape == (960, 1280)
        cam._close()

    def test_next_image_at_end_returns_none(self, video_path):
        cam = MovieVirtualCamera(video_path)
        cam.capture.set(cv2.CAP_PROP_POS_FRAMES, 1200)
        assert cam._next_image() is None
        cam._close()

    def test_time_stamp_file_based(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam._time_stamp() == 0.0
        cam._close()

    def test_time_stamp_wall_clock(self, video_path):
        cam = MovieVirtualCamera(video_path, use_wall_clock=True)
        assert cam._time_stamp() >= 0.0
        cam._close()

    def test_is_last_frame(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam.is_last_frame() is False
        cam._frame_idx = cam._total_n_frames
        assert cam.is_last_frame() is True
        cam._close()

    def test_close_releases_capture(self, video_path):
        cam = MovieVirtualCamera(video_path)
        cam._close()
        assert cam.capture.isOpened() is False

    def test_iteration_with_max_duration(self, video_path):
        cam = MovieVirtualCamera(video_path, max_duration=1.0)
        frames = list(cam)
        assert len(frames) > 0
        for t_ms, frame in frames:
            assert isinstance(t_ms, int)
            assert frame.ndim == 2
        cam._close()

    def test_iteration_drop_each(self, video_path):
        full = MovieVirtualCamera(video_path, max_duration=1.0)
        n_full = len(list(full))
        full._close()
        dropped = MovieVirtualCamera(video_path, max_duration=1.0, drop_each=4)
        n_dropped = len(list(dropped))
        dropped._close()
        assert 0 < n_dropped <= n_full


# ===========================================================================
# V4L2Camera
# ===========================================================================


def _make_v4l2_capture(frame=None, opened=True):
    capture = Mock()
    capture.isOpened.return_value = opened
    capture.read.return_value = (True, frame)
    capture.retrieve.return_value = True
    return capture


class TestV4L2Camera:
    def _init_camera(self, frame, target_resolution=(960, 720), target_fps=25):
        capture = _make_v4l2_capture(frame=frame)
        with (
            patch.object(cameras, "cv2") as mock_cv2,
            patch.object(cameras.time, "sleep"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
        ):
            mock_cv2.VideoCapture.return_value = capture
            cam = V4L2Camera(
                device=0,
                target_fps=target_fps,
                target_resolution=target_resolution,
            )
        return cam, capture

    def test_init_success(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, capture = self._init_camera(frame)
        assert cam._resolution == (960, 720)
        assert cam.fps == 25
        assert cam.isPiCamera is False
        assert capture.set.called
        cam._close()

    def test_init_rejects_non_integer_fps(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        with (
            patch.object(cameras, "cv2") as mock_cv2,
            patch.object(cameras.time, "sleep"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
        ):
            mock_cv2.VideoCapture.return_value = _make_v4l2_capture(frame=frame)
            with pytest.raises(EthoscopeException, match="FPS must be an integer"):
                V4L2Camera(target_fps=25.5)

    def test_init_rejects_fps_below_two(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        with (
            patch.object(cameras, "cv2") as mock_cv2,
            patch.object(cameras.time, "sleep"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
        ):
            mock_cv2.VideoCapture.return_value = _make_v4l2_capture(frame=frame)
            with pytest.raises(EthoscopeException, match="FPS must be at least 2"):
                V4L2Camera(target_fps=1)

    def test_init_raises_when_first_frame_missing(self):
        with (
            patch.object(cameras, "cv2") as mock_cv2,
            patch.object(cameras.time, "sleep"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
        ):
            mock_cv2.VideoCapture.return_value = _make_v4l2_capture(frame=None)
            with pytest.raises(EthoscopeException, match="Got None instead"):
                V4L2Camera(target_fps=25, target_resolution=(960, 720))

    def test_restart_resets_state(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, _ = self._init_camera(frame)
        cam._frame_idx = 42
        cam.restart()
        assert cam._frame_idx == 0
        assert cam._start_time <= time.time()
        cam._close()

    def test_is_opened_and_last_frame(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, capture = self._init_camera(frame)
        assert cam.is_opened() is True
        assert cam.is_last_frame() is False
        capture.isOpened.return_value = False
        assert cam.is_opened() is False
        cam._close()

    def test_time_stamp_and_start_time(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, _ = self._init_camera(frame)
        assert cam._time_stamp() >= 0.0
        assert cam.start_time <= time.time()
        cam._close()

    def test_next_image_converts_bgr_to_gray(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, capture = self._init_camera(frame)
        cam._frame = frame.copy()
        result = cam._next_image()
        assert result.ndim == 2
        cam._close()

    def test_close_releases_capture(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, capture = self._init_camera(frame)
        cam._close()
        capture.release.assert_called_once()


# ===========================================================================
# PiFrameGrabber (picamera2)
# ===========================================================================


class TestPiFrameGrabber:
    def test_save_camera_info_writes_file(self, tmp_path):
        out = tmp_path / "info"
        grabber = object.__new__(PiFrameGrabber)
        PiFrameGrabber._save_camera_info(
            grabber, {"Model": "imx708", "Num": 0}, save_path=str(out)
        )
        content = out.read_text()
        assert "imx708" in content
        assert "IFD0.Model" in content  # compatibility double-key

    def test_get_video_chunk_filename(self):
        grabber = object.__new__(PiFrameGrabber)
        grabber._video_prefix = "/tmp/chunk"
        grabber._target_resolution = (960, 720)
        grabber.video_quality = 20
        grabber._file_index = 0
        grabber._last_computed_filename = ""

        name = grabber._get_video_chunk_filename(fps=25)
        assert name == "/tmp/chunk_960x720@25fps-20q_00001.h264"
        assert grabber._file_index == 1

        assert grabber._get_video_chunk_filename(current=True) == name

    def _make_grabber(self):
        with patch.object(cameras.pi, "get_gain_setting", return_value=1.0):
            grabber = PiFrameGrabber(
                target_fps=10,
                target_resolution=(640, 480),
                queue=queue.Queue(),
                stop_queue=queue.Queue(),
            )
        return grabber

    def test_run_puts_none_with_automatic_tuning(self):
        grabber = self._make_grabber()
        with patch.object(cameras.pi, "get_noir_setting", return_value=False):
            grabber.run()
        assert grabber._queue.get() is None

    def test_run_puts_none_with_noir_tuning(self):
        grabber = self._make_grabber()
        with patch.object(cameras.pi, "get_noir_setting", return_value=True):
            grabber.run()
        assert grabber._queue.get() is None

    def test_run_puts_none_when_picamera2_is_none(self):
        grabber = self._make_grabber()
        with patch.object(cameras, "Picamera2", None):
            grabber.run()
        assert grabber._queue.get() is None

    def test_run_non_camera_exception_does_not_put_none(self):
        # Exception without camera keywords should take warning branch, not put None.
        # The stub Picamera2 raises RuntimeError("picamera2 is not available ...") which contains "camera",
        # so we need to force a non-camera error by making get_noir_setting raise ValueError.
        grabber = self._make_grabber()
        with patch.object(
            cameras.pi, "get_noir_setting", side_effect=ValueError("Some other error")
        ):
            grabber.run()
        # For non-camera errors the queue should have no None; task_done leaves queue empty except the guarded call.
        # run() does not put None in this branch, so get(timeout=0.1) should raise Empty.
        with pytest.raises(queue.Empty):
            grabber._queue.get(timeout=0.1)

    def test_save_camera_info_without_model(self, tmp_path):
        out = tmp_path / "info2"
        grabber = object.__new__(PiFrameGrabber)
        original = {"Num": 0, "Location": 2}
        PiFrameGrabber._save_camera_info(grabber, dict(original), save_path=str(out))
        content = out.read_text()
        assert "Num" in content
        # When Model absent, IFD0.Model must NOT be injected
        assert "IFD0.Model" not in content

    def test_save_camera_info_mutates_input_with_model(self, tmp_path):
        out = tmp_path / "info3"
        grabber = object.__new__(PiFrameGrabber)
        data = {"Model": "imx708", "Num": 0}
        PiFrameGrabber._save_camera_info(grabber, data, save_path=str(out))
        assert data["IFD0.Model"] == "imx708"

    def test_get_video_chunk_filename_sequential_and_fps_none_and_ext(self):
        grabber = object.__new__(PiFrameGrabber)
        grabber._video_prefix = "/tmp/chunk"
        grabber._target_resolution = (960, 720)
        grabber.video_quality = 20
        grabber._file_index = 0
        grabber._last_computed_filename = ""

        first = grabber._get_video_chunk_filename(fps=None)
        assert first == "/tmp/chunk_960x720@0fps-20q_00001.h264"
        second = grabber._get_video_chunk_filename(fps=25)
        assert second == "/tmp/chunk_960x720@25fps-20q_00002.h264"
        assert grabber._file_index == 2
        # current=True returns last without increment
        assert grabber._get_video_chunk_filename(current=True) == second
        assert grabber._file_index == 2
        # custom extension
        grabber._file_index = 0
        grabber._last_computed_filename = ""
        mp4 = grabber._get_video_chunk_filename(fps=10, ext="mp4")
        assert mp4.endswith(".mp4")

    def test_run_success_puts_frames_and_calls_cleanup(self):
        # Mock Picamera2 to simulate successful capture of one frame
        grabber = self._make_grabber()
        grabber._queue = queue.Queue()
        grabber._stop_queue = queue.Queue()
        # pre-signal stop after one frame: make empty side_effect [True, False]
        # but we need capture_array to be called once

        mock_capture = Mock()
        mock_capture.__enter__ = Mock(return_value=mock_capture)
        mock_capture.__exit__ = Mock(return_value=False)
        mock_capture.global_camera_info.return_value = [{"Model": "imx708", "Num": 0}]
        mock_capture.create_video_configuration.return_value = Mock()
        mock_capture.configure = Mock()
        mock_capture.start = Mock()
        mock_capture.stop = Mock()
        # YUV420 frame: height 480 + 240 (U/V) = 720 rows, width 640
        full_frame = np.zeros((720, 640), dtype=np.uint8)
        mock_capture.capture_array.return_value = full_frame
        mock_capture.camera_controls = Mock()
        mock_capture.camera_controls.get.return_value = "Unknown"

        mock_picamera2 = Mock(return_value=mock_capture)
        mock_picamera2.load_tuning_file = Mock(return_value=Mock())
        mock_picamera2.set_logging = Mock()

        # stop queue behavior: first check empty True (enter loop), second False (exit)
        # Simulate by patching _stop_queue.empty to return True once then False
        call_count = {"n": 0}

        def empty_side_effect():
            call_count["n"] += 1
            return call_count["n"] == 1

        with (
            patch.object(cameras, "Picamera2", mock_picamera2),
            patch.object(cameras.pi, "get_noir_setting", return_value=False),
            patch.object(cameras, "MappedArray"),  # not used in non-record path
            patch.object(cameras.PiFrameGrabber, "_save_camera_info"),
            patch.object(OurPiCameraAsync, "_perform_camera_cleanup") as mock_cleanup,
        ):
            grabber._stop_queue.empty = empty_side_effect
            grabber._stop_queue.get = Mock()
            grabber._stop_queue.task_done = Mock()
            grabber.run()
            # After run, queue should have one frame (Y slice)
            assert not grabber._queue.empty()
            frame = grabber._queue.get()
            assert frame.shape == (480, 640)  # Y slice
            mock_capture.start.assert_called_once()
            mock_capture.stop.assert_called_once()
            mock_capture.create_video_configuration.assert_called_once()
            mock_cleanup.assert_called_once_with(delay=0)

    def test_run_success_with_record_video(self):
        grabber = self._make_grabber()
        grabber._target_resolution = (640, 480)
        grabber._queue = queue.Queue()
        grabber._stop_queue = queue.Queue()
        grabber._record_video = True
        grabber._video_prefix = "/tmp/rec"

        mock_capture = Mock()
        mock_capture.__enter__ = Mock(return_value=mock_capture)
        mock_capture.__exit__ = Mock(return_value=False)
        mock_capture.global_camera_info.return_value = [{"Model": "imx708"}]
        mock_capture.create_video_configuration.return_value = Mock()
        mock_capture.configure = Mock()
        mock_capture.start = Mock()
        mock_capture.stop = Mock()
        mock_capture.start_encoder = Mock()
        mock_capture.stop_encoder = Mock()
        mock_capture.capture_request = Mock()
        mock_capture.camera_controls = Mock()
        mock_capture.camera_controls.get.return_value = "Unknown"
        # request for preview
        mock_request = Mock()
        mock_frame = Mock()
        mock_frame.array = np.zeros((480, 640), dtype=np.uint8)
        mock_capture.capture_request.return_value = mock_request

        mock_picamera2 = Mock(return_value=mock_capture)
        mock_picamera2.load_tuning_file = Mock(return_value=Mock())
        mock_picamera2.set_logging = Mock()

        mock_encoder = Mock()
        mock_encoder_cls = Mock(return_value=mock_encoder)

        # Need to patch picamera2.encoders.H264Encoder and MappedArray
        # Patch at cameras module level for H264Encoder import inside run()
        import sys
        import types

        enc_module = types.ModuleType("picamera2.encoders")
        enc_module.H264Encoder = mock_encoder_cls
        sys.modules["picamera2.encoders"] = enc_module

        # Use MappedArray mock as context manager
        mock_mapped = Mock()
        mock_mapped.__enter__ = Mock(return_value=mock_frame)
        mock_mapped.__exit__ = Mock(return_value=False)

        call_count = {"n": 0}

        def empty_side_effect():
            call_count["n"] += 1
            # Run two iterations to ensure preview refresh triggers, then exit
            return call_count["n"] <= 2

        # Make preview refresh trigger immediately
        grabber._PREVIEW_REFRESH_TIME = 0

        with (
            patch.object(cameras, "Picamera2", mock_picamera2),
            patch.object(cameras.pi, "get_noir_setting", return_value=False),
            patch.object(cameras, "MappedArray", return_value=mock_mapped),
            patch.object(cameras.PiFrameGrabber, "_save_camera_info"),
            patch.object(OurPiCameraAsync, "_perform_camera_cleanup"),
        ):
            grabber._stop_queue.empty = empty_side_effect
            grabber._stop_queue.get = Mock()
            grabber._stop_queue.task_done = Mock()
            # run will execute preview loop then exit
            grabber.run()
            mock_capture.start_encoder.assert_called()
            mock_capture.stop_encoder.assert_called()
            mock_capture.capture_request.assert_called()


# ===========================================================================
# OurPiCameraAsync
# ===========================================================================


class TestOurPiCameraAsync:
    def _bare_camera(self):
        cam = object.__new__(OurPiCameraAsync)
        cam._frame_idx = 0
        cam._start_time = time.time()
        cam._args = ()
        cam._kwargs = {}
        return cam

    def test_perform_camera_cleanup(self):
        with patch.object(cameras, "time"):
            OurPiCameraAsync._perform_camera_cleanup(delay=0)
        # should not raise even with the picamera2 stub installed

    def test_restart_resets_state(self):
        cam = self._bare_camera()
        cam._frame_idx = 7
        cam.restart()
        assert cam._frame_idx == 0

    def test_is_opened_and_last_frame(self):
        cam = self._bare_camera()
        assert cam.is_opened() is True
        assert cam.is_last_frame() is False

    def test_time_stamp(self):
        cam = self._bare_camera()
        cam._start_time = time.time() - 5
        assert cam._time_stamp() == pytest.approx(5.0, abs=1.0)

    def test_start_time_property(self):
        cam = self._bare_camera()
        assert cam.start_time == cam._start_time

    def test_next_image_returns_queue_frame(self):
        cam = self._bare_camera()
        cam._queue = Mock()
        frame = np.zeros((480, 640), np.uint8)
        cam._queue.get.return_value = frame
        assert cam._next_image() is frame
        cam._queue.get.assert_called_once()

    def test_next_image_raises_on_queue_timeout(self):
        cam = self._bare_camera()
        cam._queue = Mock()
        cam._queue.get.side_effect = queue.Empty("timeout")
        with pytest.raises(EthoscopeException):
            cam._next_image()

    def test_getstate(self):
        cam = object.__new__(OurPiCameraAsync)
        cam._args = (1,)
        cam._kwargs = {"a": 2}
        cam._frame_idx = 3
        cam._start_time = 42.0
        state = cam.__getstate__()
        assert state["args"] == (1,)
        assert state["kwargs"] == {"a": 2}
        assert state["frame_idx"] == 3

    def test_cleanup_frame_grabber_joins(self):
        cam = self._bare_camera()
        cam._stop_queue = queue.Queue()
        cam._queue = queue.Queue()
        cam._p = Mock()
        cam._p.is_alive.return_value = True
        cam._cleanup_frame_grabber()
        cam._p.join.assert_called_once()
        assert not cam._stop_queue.empty()

    def test_init_success_with_mocked_first_frame(self):
        frame = np.zeros((720, 960), np.uint8)
        mock_queue = Mock()
        mock_queue.get.return_value = frame

        grabber_instance = Mock()
        grabber_class = Mock(return_value=grabber_instance)

        with (
            patch.object(cameras.queue, "Queue") as mock_queue_cls,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber", grabber_class),
        ):
            mock_queue_cls.side_effect = lambda maxsize=0: mock_queue
            cam = OurPiCameraAsync(target_fps=10, target_resolution=(960, 720))

        assert cam._resolution == (960, 720)
        grabber_class.assert_called_once()
        grabber_instance.start.assert_called_once()
        cam._close()

    def test_init_fails_when_first_frame_none(self):
        mock_queue = Mock()
        mock_queue.get.return_value = None
        mock_stop = Mock()
        mock_stop.empty.return_value = True
        mock_queue.empty.return_value = True
        grabber_instance = Mock()
        grabber_class = Mock(return_value=grabber_instance)

        # mock queue module to return our mocks per call
        def queue_side_effect(maxsize=0):
            # first call is _queue, second is _stop_queue
            queue_side_effect.calls += 1
            return mock_queue if queue_side_effect.calls == 1 else mock_stop

        queue_side_effect.calls = 0

        with (
            patch.object(cameras.queue, "Queue") as mock_queue_cls,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber", grabber_class),
            patch.object(OurPiCameraAsync, "_cleanup_frame_grabber") as mock_cleanup,
        ):
            mock_queue_cls.side_effect = queue_side_effect
            with pytest.raises(
                EthoscopeException, match="Camera hardware not available"
            ):
                OurPiCameraAsync(target_fps=10, target_resolution=(640, 480))

        grabber_class.assert_called_once()
        mock_cleanup.assert_called_once_with(force_global_cleanup=True)

    def test_init_fails_on_queue_empty_timeout(self):
        mock_queue = Mock()
        mock_queue.get.side_effect = queue.Empty("timeout")
        mock_queue.empty.return_value = True
        mock_stop = Mock()
        mock_stop.empty.return_value = True
        grabber_instance = Mock()
        grabber_class = Mock(return_value=grabber_instance)

        def queue_side_effect(maxsize=0):
            queue_side_effect.calls += 1
            return mock_queue if queue_side_effect.calls == 1 else mock_stop

        queue_side_effect.calls = 0

        with (
            patch.object(cameras.queue, "Queue") as mock_queue_cls,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber", grabber_class),
            patch.object(OurPiCameraAsync, "_cleanup_frame_grabber") as mock_cleanup,
        ):
            mock_queue_cls.side_effect = queue_side_effect
            with pytest.raises(
                EthoscopeException, match="Camera initialization timeout"
            ):
                OurPiCameraAsync(target_fps=10, target_resolution=(640, 480))

        grabber_class.assert_called_once()
        mock_cleanup.assert_called_once_with(force_global_cleanup=True)

    def test_init_fails_on_corrupted_frame(self):
        # shape with len < 2 should raise "corrupted"
        frame = np.zeros((5,), dtype=np.uint8)  # 1-dim
        mock_queue = Mock()
        mock_queue.get.return_value = frame
        grabber_instance = Mock()
        grabber_class = Mock(return_value=grabber_instance)

        with (
            patch.object(cameras.queue, "Queue") as mock_queue_cls,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber", grabber_class),
        ):
            mock_queue_cls.side_effect = lambda maxsize=0: mock_queue
            with pytest.raises(EthoscopeException, match="corrupted"):
                OurPiCameraAsync(target_fps=10, target_resolution=(640, 480))

        grabber_class.assert_called_once()

    def test_init_single_attempt_guarantee(self):
        # Even when queue returns None, ensure no second PiFrameGrabber instantiation
        mock_queue = Mock()
        mock_queue.get.return_value = None
        mock_queue.empty.return_value = True
        mock_stop = Mock()
        grabber_instance = Mock()
        grabber_class = Mock(return_value=grabber_instance)

        def queue_side_effect(maxsize=0):
            queue_side_effect.calls += 1
            return mock_queue if queue_side_effect.calls == 1 else mock_stop

        queue_side_effect.calls = 0

        with (
            patch.object(cameras.queue, "Queue") as mock_queue_cls,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber", grabber_class),
            patch.object(OurPiCameraAsync, "_cleanup_frame_grabber"),
        ):
            mock_queue_cls.side_effect = queue_side_effect
            with pytest.raises(EthoscopeException):
                OurPiCameraAsync(target_fps=10, target_resolution=(640, 480))
            assert grabber_class.call_count == 1

    def test_init_rejects_non_integer_fps(self):
        with (
            patch.object(cameras, "queue"),
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
        ):
            with pytest.raises(EthoscopeException, match="FPS must be an integer"):
                OurPiCameraAsync(target_fps=25.5, target_resolution=(640, 480))

    def test_setstate_resets_start_time(self):
        cam = object.__new__(OurPiCameraAsync)
        # patch __init__ to avoid real hardware, then test __setstate__ logic partially
        with patch.object(OurPiCameraAsync, "__init__", lambda self, *a, **k: None):
            state = {"args": (), "kwargs": {}, "frame_idx": 5, "start_time": 1234.0}
            # need to set _frame_idx before call, but __setstate__ will call __init__
            # so patch time.time to control
            with patch.object(cameras.time, "time", return_value=9999.0):
                # For __setstate__ we need _frame_idx already? Actually __setstate__ does self.__init__ then sets _frame_idx
                # Provide minimal attributes
                cam._frame_idx = 0
                OurPiCameraAsync.__setstate__(cam, state)
                assert cam._frame_idx == 5
                assert cam._start_time == 9999.0
