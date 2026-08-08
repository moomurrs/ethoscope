"""
Unit tests for hardware/input/cameras.py.

Tests the camera abstraction layer without real hardware:
  * BaseCamera frame iteration/dropping contract
  * MovieVirtualCamera (real mp4 playback via OpenCV)
  * V4L2Camera (mocked capture device)
  * PiFrameGrabber / PiFrameGrabber2 failure signalling (no camera hardware)
  * OurPiCameraAsync lifecycle helpers (state, queue, cleanup)

Camera hardware paths (PiFrameGrabber2 recording, OurPiCameraAsync real
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
    PiFrameGrabber2,
    V4L2Camera,
)
from ethoscope.utils.debug import EthoscopeException

TEST_VIDEO = str(
    Path(__file__).parent.parent / "static_files" / "videos" / "arena_10x2_sortTubes.mp4"
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
        cam = _IterCamera(
            [np.zeros((4, 4), np.uint8)] * 20, max_duration=0.05
        )
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
# PiFrameGrabber (legacy picamera)
# ===========================================================================


class TestPiFrameGrabber:
    def test_save_camera_info_writes_file(self, tmp_path):
        out = tmp_path / "info"
        grabber = object.__new__(PiFrameGrabber)
        PiFrameGrabber._save_camera_info(
            grabber, {"Model": "imx219", "Num": 0}, save_path=str(out)
        )
        content = out.read_text()
        assert "imx219" in content
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

    def test_run_puts_none_when_picamera_missing(self):
        # picamera is not installed in the test environment, so run() must
        # signal "no camera" by putting None on the queue.
        grabber = object.__new__(PiFrameGrabber)
        grabber._queue = queue.Queue()
        grabber._stop_queue = queue.Queue()
        grabber._target_resolution = (640, 480)
        grabber._record_video = False

        grabber.run()
        assert grabber._queue.get() is None


# ===========================================================================
# PiFrameGrabber2 (picamera2)
# ===========================================================================


class TestPiFrameGrabber2:
    def _make_grabber(self):
        with patch.object(cameras.pi, "get_gain_setting", return_value=1.0):
            grabber = PiFrameGrabber2(
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
            patch.object(cameras, "queue") as mock_queue_module,
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "PiFrameGrabber2", grabber_class),
        ):
            mock_queue_module.Queue.side_effect = lambda maxsize=0: mock_queue
            cam = OurPiCameraAsync(target_fps=10, target_resolution=(960, 720))

        assert cam._resolution == (960, 720)
        grabber_class.assert_called_once()
        grabber_instance.start.assert_called_once()
        cam._close()
