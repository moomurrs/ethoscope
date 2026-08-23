"""
Unit tests for hardware/input/cameras.py.

Tests the camera abstraction layer without real hardware:
  * BaseCamera frame iteration/dropping contract and context manager
  * MovieVirtualCamera (real mp4 playback via OpenCV)
  * V4L2Camera (mocked capture device)
  * _save_camera_info persistence helper
  * Picamera2Driver (tuning selection and lifecycle, mocked picamera2)
  * VideoRecorder (chunk naming, preview frames, encoder rotation)
  * FrameProducer (frame pumping, drop policy, error signalling)
  * Picamera2Camera lifecycle helpers (state, queue, cleanup, pickling)

Camera hardware paths (real frame acquisition) are stubbed out by the root
conftest which signals "no camera" via a picamera2 stub.
"""

import queue
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from ethoscope.hardware.input import cameras
from ethoscope.hardware.input.cameras import (
    BaseCamera,
    CameraConfig,
    CameraError,
    FrameProducer,
    MovieVirtualCamera,
    Picamera2Camera,
    Picamera2Driver,
    V4L2Camera,
    VideoRecorder,
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

    def _close(self):
        pass


class TestBaseCamera:
    def test_init_stores_drop_each_and_max_duration(self):
        cam = _IterCamera([], drop_each=3, max_duration=12.5)
        assert cam._drop_each == 3  # noqa: PLR2004 - magic values in tests are intentional
        assert cam._max_duration == 12.5  # noqa: PLR2004 - magic values in tests are intentional

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseCamera()  # type: ignore[abstract]

    def test_exit_closes_camera(self):
        cam = _IterCamera([])
        with patch.object(cam, "_close") as mock_close:
            cam.__exit__(None, None, None)
        mock_close.assert_called_once()

    def test_context_manager_enter_and_exit(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)])
        with patch.object(cam, "_close") as mock_close, cam as entered:
            assert entered is cam
        mock_close.assert_called_once()

    def test_base_close_is_noop(self):
        cam = _IterCamera([])
        cam._close()  # should not raise

    def test_resolution_width_height_properties(self):
        cam = _IterCamera([])
        cam._resolution = (640, 480)
        assert cam.resolution == (640, 480)
        assert cam.width == 640  # noqa: PLR2004 - magic values in tests are intentional
        assert cam.height == 480  # noqa: PLR2004 - magic values in tests are intentional

    def test_next_time_image_increments_frame_idx(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)])
        t, img = cam._next_time_image()
        assert cam._frame_idx == 1
        assert isinstance(t, float)
        assert img is not None

    def test_iter_yields_frames_with_ms_timestamps(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 3)
        out = list(cam)
        assert len(out) == 3  # noqa: PLR2004 - magic values in tests are intentional
        for t_ms, frame in out:
            assert isinstance(t_ms, int)
            assert isinstance(frame, np.ndarray)
        # frame indices advance on each underlying read
        assert cam._frame_idx == 3  # noqa: PLR2004 - magic values in tests are intentional

    def test_iter_drops_frames_per_drop_each(self):
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 4, drop_each=2)
        out = list(cam)
        # frames at frame_idx 2 and 4 are yielded (index % 2 == 0)
        assert len(out) == 2  # noqa: PLR2004 - magic values in tests are intentional

    def test_iter_respects_max_duration(self):
        # t = frame_idx / 30; stop when t > 0.05 -> ~2 frames
        cam = _IterCamera([np.zeros((4, 4), np.uint8)] * 20, max_duration=0.05)
        out = list(cam)
        assert 0 < len(out) < 20  # noqa: PLR2004 - magic values in tests are intentional

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
    if not Path(TEST_VIDEO).exists():
        pytest.skip("test video not available")
    return TEST_VIDEO


class TestMovieVirtualCamera:
    def test_init_reads_video_metadata(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam._resolution == (1280, 960)
        assert cam._total_n_frames == 1200  # noqa: PLR2004 - magic values in tests are intentional
        assert cam._has_end_of_file is True
        assert cam.path == video_path
        assert cam.start_time == 0
        assert cam.hardware_recording is False

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
            MovieVirtualCamera(12345)  # type: ignore[arg-type]

    def test_is_opened(self, video_path):
        cam = MovieVirtualCamera(video_path)
        assert cam.is_opened() is True
        cam._close()

    def test_restart_reopens_without_leak(self, video_path):
        cam = MovieVirtualCamera(video_path)
        cam._frame_idx = 100
        old_capture = cam.capture
        cam.restart()
        assert cam._frame_idx == 0
        assert cam.is_opened() is True
        assert old_capture.isOpened() is False  # previous capture released
        cam._close()

    def test_next_image_returns_grayscale(self, video_path):
        cam = MovieVirtualCamera(video_path)
        frame = cam._next_image()
        assert frame is not None
        assert frame.ndim == 2  # noqa: PLR2004 - magic values in tests are intentional
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
        cam._frame_idx = int(cam._total_n_frames)
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
            assert frame.ndim == 2  # noqa: PLR2004 - magic values in tests are intentional
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
        assert cam.fps == 25  # noqa: PLR2004 - magic values in tests are intentional
        assert cam.hardware_recording is False
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
                V4L2Camera(target_fps=25.5)  # type: ignore[arg-type]

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
        cam, _ = self._init_camera(frame)
        cam._frame = frame.copy()
        result = cam._next_image()
        assert result is not None
        assert result.ndim == 2  # noqa: PLR2004 - magic values in tests are intentional
        cam._close()

    def test_close_releases_capture(self):
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        cam, capture = self._init_camera(frame)
        cam._close()
        capture.release.assert_called_once()


# ===========================================================================
# _save_camera_info
# ===========================================================================


class TestSaveCameraInfo:
    def test_writes_file_with_model(self, tmp_path):
        out = tmp_path / "info"
        cameras._save_camera_info({"Model": "imx708", "Num": 0}, save_path=str(out))
        content = out.read_text()
        assert "imx708" in content
        assert "IFD0.Model" in content  # compatibility double-key

    def test_without_model_does_not_inject_key(self, tmp_path):
        out = tmp_path / "info2"
        original = {"Num": 0, "Location": 2}
        cameras._save_camera_info(dict(original), save_path=str(out))
        content = out.read_text()
        assert "Num" in content
        assert "IFD0.Model" not in content

    def test_does_not_mutate_input_with_model(self, tmp_path):
        data = {"Model": "imx708", "Num": 0}
        cameras._save_camera_info(data, save_path=str(tmp_path / "info3"))
        assert "IFD0.Model" not in data  # input dict is left untouched


# ===========================================================================
# Picamera2Driver (picamera2)
# ===========================================================================


def _mock_capture():
    capture = Mock()
    capture.create_video_configuration.return_value = Mock()
    capture.camera_controls = Mock()
    capture.camera_controls.get.return_value = "Unknown"
    capture.global_camera_info.return_value = [{"Model": "imx708", "Num": 0}]
    return capture


def _driver_config(**overrides):
    params = {
        "target_fps": 10,
        "target_resolution": (640, 480),
        "gain": 1.0,
        "noir": False,
    }
    params.update(overrides)
    return CameraConfig(**params)


class TestPicamera2Driver:
    def test_open_with_automatic_tuning(self):
        capture = _mock_capture()
        mock_picamera2 = Mock(return_value=capture)
        driver = Picamera2Driver(_driver_config())
        with (
            patch.object(cameras, "Picamera2", mock_picamera2),
            patch.object(cameras, "_save_camera_info"),
        ):
            opened = driver.open()
        assert opened is capture
        mock_picamera2.assert_called_once_with()
        capture.configure.assert_called_once()
        capture.create_video_configuration.assert_called_once()
        driver.close()
        capture.stop.assert_called_once()
        capture.close.assert_called_once()

    def test_open_with_noir_tuning_uses_first_working_file(self):
        capture = _mock_capture()
        mock_picamera2 = Mock(return_value=capture)
        mock_picamera2.load_tuning_file.side_effect = [RuntimeError("missing"), Mock()]
        driver = Picamera2Driver(_driver_config(noir=True))
        with (
            patch.object(cameras, "Picamera2", mock_picamera2),
            patch.object(cameras, "_save_camera_info"),
        ):
            driver.open()
        # one tuning file failed, the second succeeded -> single Picamera2() call
        assert mock_picamera2.call_count == 1
        assert mock_picamera2.load_tuning_file.call_count == 2  # noqa: PLR2004

    def test_open_noir_falls_back_when_all_tuning_files_fail(self):
        capture = _mock_capture()
        mock_picamera2 = Mock(return_value=capture)
        mock_picamera2.load_tuning_file.side_effect = RuntimeError("missing")
        driver = Picamera2Driver(_driver_config(noir=True))
        with (
            patch.object(cameras, "Picamera2", mock_picamera2),
            patch.object(cameras, "_save_camera_info"),
        ):
            driver.open()
        # all tuning files failed -> single fallback Picamera2() instance
        assert mock_picamera2.call_count == 1
        assert mock_picamera2.load_tuning_file.call_count == len(
            Picamera2Driver.NOIR_TUNING_FILES
        )

    def test_close_is_idempotent(self):
        driver = Picamera2Driver(_driver_config())
        driver.close()  # should not raise without an open camera
        driver.close()


# ===========================================================================
# VideoRecorder
# ===========================================================================


def _recorder_config(target_resolution=(960, 720)):
    return CameraConfig(
        target_fps=10,
        target_resolution=target_resolution,
        video_prefix="/tmp/chunk",
        quality=20,
    )


class TestVideoRecorder:
    def test_chunk_filename(self):
        recorder = VideoRecorder(_recorder_config())
        name = recorder.chunk_filename(fps=25)
        assert name == "/tmp/chunk_960x720@25fps-20q_00001.h264"
        assert recorder._file_index == 1
        assert recorder.chunk_filename(current=True) == name

    def test_chunk_filename_sequential_and_ext(self):
        recorder = VideoRecorder(_recorder_config())
        first = recorder.chunk_filename()
        assert first == "/tmp/chunk_960x720@0fps-20q_00001.h264"
        second = recorder.chunk_filename(fps=25)
        assert second == "/tmp/chunk_960x720@25fps-20q_00002.h264"
        assert recorder._file_index == 2  # noqa: PLR2004 - magic values in tests are intentional
        assert recorder.chunk_filename(current=True) == second
        recorder._file_index = 0
        recorder._last_computed_filename = ""
        assert recorder.chunk_filename(fps=10, ext="mp4").endswith(".mp4")

    def test_start_preview_and_rotate(self):
        recorder = VideoRecorder(_recorder_config(target_resolution=(640, 480)))
        capture = Mock()
        mock_request = Mock()
        mock_frame = Mock()
        mock_frame.array = np.zeros((720, 640), dtype=np.uint8)  # YUV420 full frame
        mock_frame.__enter__ = Mock(return_value=mock_frame)
        mock_frame.__exit__ = Mock(return_value=False)
        capture.capture_request.return_value = mock_request

        enc_module = types.ModuleType("picamera2.encoders")
        enc_module.H264Encoder = Mock()  # type: ignore[attr-defined]
        sys.modules["picamera2.encoders"] = enc_module

        with patch.object(cameras, "MappedArray", return_value=mock_frame):
            recorder.start(capture, fps=10)
            capture.start_encoder.assert_called_once()
            recorder._refresh_interval = time.monotonic() - 60
            recorder._video_time = time.monotonic() - 400
            frame = recorder.preview_frame(capture, 480)
            assert frame is not None
            assert frame.shape == (480, 640)
            recorder.rotate_if_needed(capture, 10)
            capture.stop_encoder.assert_called()
            recorder.stop(capture)
            assert capture.stop_encoder.call_count == 2  # noqa: PLR2004 - magic values in tests are intentional

    def test_stop_without_start_is_noop(self):
        recorder = VideoRecorder(_recorder_config())
        recorder.stop(Mock())  # should not raise


# ===========================================================================
# FrameProducer
# ===========================================================================


class TestFrameProducer:
    def _make_producer(self, recorder=None):
        driver = Mock()
        frame_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        producer = FrameProducer(
            driver,
            frame_queue,
            stop_event,
            _driver_config(),
            recorder=recorder,
        )
        return producer, driver, frame_queue, stop_event

    def test_run_fast_fail_when_driver_open_fails(self):
        producer, driver, frame_queue, _ = self._make_producer()
        driver.open.side_effect = RuntimeError("no camera")
        producer.run()
        assert frame_queue.get() is None
        assert producer.error is not None
        driver.close.assert_not_called()

    def test_run_pumps_frames_and_closes_driver(self):
        producer, driver, frame_queue, stop_event = self._make_producer()
        capture = Mock()
        capture.capture_array.return_value = np.zeros((720, 640), np.uint8)
        driver.open.return_value = capture

        calls = {"n": 0}
        real_is_set = stop_event.is_set

        def is_set():
            calls["n"] += 1
            return calls["n"] >= 2 or real_is_set()  # noqa: PLR2004 - magic values in tests are intentional

        stop_event.is_set = is_set  # type: ignore[method-assign]
        producer.run()
        frame = frame_queue.get()
        assert frame.shape == (480, 640)  # Y plane slice
        assert producer.error is None
        capture.start.assert_called_once()
        driver.close.assert_called_once()

    def test_run_with_recorder_pushes_preview_frames(self):
        recorder = Mock()
        recorder.preview_frame.return_value = np.zeros((480, 640), np.uint8)
        producer, driver, frame_queue, stop_event = self._make_producer(
            recorder=recorder
        )
        capture = Mock()
        driver.open.return_value = capture

        calls = {"n": 0}
        real_is_set = stop_event.is_set

        def is_set():
            calls["n"] += 1
            return calls["n"] >= 3 or real_is_set()  # noqa: PLR2004 - magic values in tests are intentional

        stop_event.is_set = is_set  # type: ignore[method-assign]
        producer.run()
        recorder.start.assert_called_once()
        recorder.rotate_if_needed.assert_called()
        recorder.stop.assert_called_once()
        assert frame_queue.get() is not None
        capture.start.assert_called_once()
        driver.close.assert_called_once()


# ===========================================================================
# Picamera2Camera
# ===========================================================================


class TestPicamera2Camera:
    def _bare_camera(self):
        cam = object.__new__(Picamera2Camera)
        cam._frame_idx = 0
        cam._start_time = time.time()
        cam._start_monotonic = time.monotonic() - 5.0
        cam._queue = Mock()
        producer = Mock()
        producer.error = None
        cam._producer = producer
        return cam

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
        assert cam._time_stamp() == pytest.approx(5.0, abs=1.0)

    def test_start_time_property(self):
        cam = self._bare_camera()
        assert cam.start_time == cam._start_time

    def test_hardware_recording_flag(self):
        assert Picamera2Camera.hardware_recording is True
        assert V4L2Camera.hardware_recording is False

    def test_next_image_returns_queue_frame(self):
        cam = self._bare_camera()
        frame = np.zeros((480, 640), np.uint8)
        cam._queue.get.return_value = frame  # type: ignore[attr-defined]
        assert cam._next_image() is frame
        cam._queue.get.assert_called_once()  # type: ignore[attr-defined]

    def test_next_image_raises_on_queue_timeout(self):
        cam = self._bare_camera()
        cam._queue.get.side_effect = queue.Empty("timeout")  # type: ignore[attr-defined]
        with pytest.raises(EthoscopeException):
            cam._next_image()

    def test_next_image_raises_producer_error(self):
        cam = self._bare_camera()
        cam._queue.get.side_effect = queue.Empty("timeout")  # type: ignore[attr-defined]
        error = CameraError("sensor exploded")
        cam._producer.error = error
        with pytest.raises(CameraError, match="sensor exploded"):
            cam._next_image()

    def test_next_image_raises_on_none_frame(self):
        cam = self._bare_camera()
        cam._queue.get.return_value = None  # type: ignore[attr-defined]
        with pytest.raises(EthoscopeException):
            cam._next_image()

    def test_getstate(self):
        cam = object.__new__(Picamera2Camera)
        cam._init_kwargs = {"target_fps": 10, "target_resolution": (960, 720)}
        cam._frame_idx = 3
        state = cam.__getstate__()
        assert state["init_kwargs"]["target_fps"] == 10  # noqa: PLR2004 - magic values in tests are intentional
        assert state["frame_idx"] == 3  # noqa: PLR2004 - magic values in tests are intentional

    def test_setstate_resets_start_time(self):
        cam = object.__new__(Picamera2Camera)
        with (
            patch.object(Picamera2Camera, "__init__", lambda self, *a, **k: None),
            patch.object(cameras.time, "time", return_value=9999.0),
        ):
            cam._frame_idx = 0
            Picamera2Camera.__setstate__(cam, {"init_kwargs": {}, "frame_idx": 5})
            assert cam._frame_idx == 5  # noqa: PLR2004 - magic values in tests are intentional
            assert cam._start_time == 9999.0  # noqa: PLR2004 - magic values in tests are intentional

    def test_shutdown_producer_joins_and_drains(self):
        cam = object.__new__(Picamera2Camera)
        cam._stop_event = threading.Event()
        cam._queue = queue.Queue()
        cam._queue.put(np.zeros((4, 4), np.uint8))
        producer = Mock()
        cam._producer = producer
        cam._driver = Mock()
        cam._shutdown_producer()
        assert cam._stop_event.is_set()
        assert cam._queue.empty()
        producer.join.assert_called_once()
        cam._driver.close.assert_called_once()

    def _init_camera(self, first_frame, producer=None):
        mock_queue = Mock()
        mock_queue.get.return_value = first_frame
        mock_queue.empty.return_value = True
        producer = producer or Mock()
        producer.error = None
        return mock_queue, producer

    def test_init_success_with_mocked_first_frame(self):
        frame = np.zeros((720, 960), np.uint8)
        mock_queue, producer = self._init_camera(frame)
        with (
            patch.object(cameras.queue, "Queue", return_value=mock_queue),
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "FrameProducer", return_value=producer),
        ):
            cam = Picamera2Camera(target_fps=10, target_resolution=(960, 720))

        assert cam._resolution == (960, 720)
        producer.start.assert_called_once()
        cam._close()

    def test_init_fails_when_first_frame_none(self):
        mock_queue, producer = self._init_camera(None)
        producer.error = CameraError("Camera hardware not available. boom")
        with (
            patch.object(cameras.queue, "Queue", return_value=mock_queue),
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "FrameProducer", return_value=producer),
            patch.object(Picamera2Camera, "_shutdown_producer") as mock_shutdown,
            pytest.raises(CameraError, match="Camera hardware not available"),
        ):
            Picamera2Camera(target_fps=10, target_resolution=(640, 480))
        mock_shutdown.assert_called_once()

    def test_init_fails_on_queue_empty_timeout(self):
        mock_queue = Mock()
        mock_queue.get.side_effect = queue.Empty("timeout")
        mock_queue.empty.return_value = True
        producer = Mock()
        producer.error = None
        with (
            patch.object(cameras.queue, "Queue", return_value=mock_queue),
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "FrameProducer", return_value=producer),
            patch.object(Picamera2Camera, "_shutdown_producer"),
            pytest.raises(CameraError, match="Camera initialization timeout"),
        ):
            Picamera2Camera(target_fps=10, target_resolution=(640, 480))

    def test_init_fails_on_corrupted_frame(self):
        mock_queue = Mock()
        mock_queue.get.return_value = np.zeros((5,), dtype=np.uint8)
        mock_queue.empty.return_value = True
        producer = Mock()
        producer.error = None
        with (
            patch.object(cameras.queue, "Queue", return_value=mock_queue),
            patch.object(cameras, "time"),
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            patch.object(cameras, "FrameProducer", return_value=producer),
            patch.object(Picamera2Camera, "_shutdown_producer"),
            pytest.raises(CameraError, match="corrupted"),
        ):
            Picamera2Camera(target_fps=10, target_resolution=(640, 480))

    def test_init_rejects_non_integer_fps(self):
        with (
            patch.object(cameras.pi, "get_maxfps_setting", return_value=30),
            pytest.raises(CameraError, match="FPS must be an integer"),
        ):
            Picamera2Camera(
                target_fps=25.5,
                target_resolution=(640, 480),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
