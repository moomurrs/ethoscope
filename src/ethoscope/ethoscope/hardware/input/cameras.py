"""
Camera abstraction layer for the ethoscope device.

Frame sources (live cameras and video files) implement :class:`BaseCamera`,
which exposes a consistent iteration contract: ``for t_ms, frame in camera``
yields frame timestamps in milliseconds and grayscale :class:`numpy.ndarray`
frames.

The Raspberry Pi camera (Camera Module 3 NoIR, picamera2, Trixie-only) is
implemented as a composition of three focused components:

* :class:`Picamera2Driver` - owns the picamera2 instance: tuning selection,
  configuration and lifecycle.
* :class:`FrameProducer` - background thread that pumps frames into a bounded
  queue and captures acquisition errors.
* :class:`VideoRecorder` - chunked H264 recording on top of a picamera2
  instance, including periodic preview frames.

:class:`Picamera2Camera` wires these components together and exposes the
same :class:`BaseCamera` contract as the other sources.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import cv2
import numpy as np
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_MSEC,
)
from picamera2 import MappedArray, Picamera2

from ethoscope.utils import pi
from ethoscope.utils.debug import EthoscopeException

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from types import TracebackType
    from typing import Self

    from picamera2.encoders import H264Encoder

MIN_FPS = 2


class CameraError(EthoscopeException):
    """Raised when a camera cannot be opened, read from, or closed."""


@dataclass(frozen=True)
class CameraConfig:
    """Immutable bundle of validated camera acquisition settings."""

    target_fps: int
    target_resolution: tuple[int, int]
    drop_each: int = 1
    max_duration: float | None = None
    gain: float = 1.0
    noir: bool = False
    video_prefix: str | None = None
    record_video: bool = False
    quality: int = 20


def _resolve_fps(target_fps: int | None) -> int:
    """Validate ``target_fps`` against the machine's configured maximum.

    Falls back to the configured maximum FPS when ``target_fps`` is
    ``None``, and clamps over-ambitious requests down to that maximum.
    """
    max_fps = pi.get_maxfps_setting()
    if target_fps is None:
        target_fps = max_fps
    if not isinstance(target_fps, int):
        raise CameraError("FPS must be an integer number")  # noqa: TRY003
    if target_fps < MIN_FPS:
        raise CameraError("FPS must be at least 2")  # noqa: TRY003
    if target_fps > max_fps:
        logger.warning(
            f"Requested FPS {target_fps} exceeds maximum {max_fps}, using {max_fps}"
        )
        target_fps = max_fps
    return target_fps


def _save_camera_info(
    camera_info: Mapping[str, Any], save_path: str = "/etc/picamera-version"
) -> None:
    """Persist detected camera info to the filesystem for other services.

    ``camera_info`` is the first element of ``Picamera2.global_camera_info()``.
    The ``IFD0.Model`` compatibility key is mirrored for the legacy reader.
    """
    info = dict(camera_info)
    if "Model" in info:
        info["IFD0.Model"] = info["Model"]
    logger.info(f"Detected camera {info}")
    with Path(save_path).open("w") as outfile:
        print(info, file=outfile)


class BaseCamera(ABC):
    """Template class to generate and use video streams.

    Subclasses implement frame acquisition (:meth:`_next_image`), timing
    (:meth:`_time_stamp`) and lifecycle (:meth:`is_opened`,
    :meth:`is_last_frame`, :meth:`restart`). Iterating over a camera yields
    ``(timestamp_ms, frame)`` tuples.
    """

    hardware_recording = False

    def __init__(self, drop_each: int = 1, max_duration: float | None = None) -> None:
        """Configure frame dropping and duration limits.

        :param drop_each: keep only ``1/drop_each``'th frame
        :param max_duration: stop the video stream if ``t > max_duration`` (seconds)
        """
        self._drop_each = drop_each
        self._max_duration = max_duration
        self._frame_idx = 0
        self._resolution: tuple[int, int] = (0, 0)
        self._start_time = 0.0
        self._start_monotonic = 0.0
        self.fps = 0.0

    @abstractmethod
    def is_opened(self) -> bool:
        """Return whether the underlying capture is open."""

    @abstractmethod
    def is_last_frame(self) -> bool:
        """Return whether the source has been fully consumed."""

    @abstractmethod
    def _next_image(self) -> np.ndarray | None:
        """Return the next frame or ``None`` at the end of the stream."""

    @abstractmethod
    def _time_stamp(self) -> float:
        """Return the time (in seconds) of the next frame."""

    @abstractmethod
    def restart(self) -> None:
        """Restart the camera from the beginning (also resets time)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self._close()
        return False

    @abstractmethod
    def _close(self) -> None:
        """Release any resources held by the camera."""

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over consecutive ``(time_ms, frame)`` pairs.

        :return: the time (in ms) and a frame (numpy array).
        """
        at_least_one_frame = False
        while True:
            if self.is_last_frame() or not self.is_opened():
                if not at_least_one_frame:
                    raise CameraError(  # noqa: TRY003
                        "Camera could not read the first frame"
                    )
                break
            t, out = self._next_time_image()
            if out is None:
                break
            t_ms = int(1000 * t)
            at_least_one_frame = True
            if self._frame_idx % self._drop_each == 0:
                yield t_ms, out
            if self._max_duration is not None and t > self._max_duration:
                break

    @property
    def resolution(self) -> tuple[int, int]:
        """The resolution of the camera W x H."""
        return self._resolution

    @property
    def width(self) -> int:
        """The width of the returned frames."""
        return self._resolution[0]

    @property
    def height(self) -> int:
        """The height of the returned frames."""
        return self._resolution[1]

    @property
    def start_time(self) -> float:
        """Wall-clock time at which the camera started acquiring frames."""
        return self._start_time

    def _next_time_image(self) -> tuple[float, np.ndarray | None]:
        time = self._time_stamp()
        im = self._next_image()
        self._frame_idx += 1
        return time, im


class MovieVirtualCamera(BaseCamera):
    _description: ClassVar[dict[str, Any]] = {
        "overview": "Class to acquire frames from a video file.",
        "arguments": [
            {
                "type": "filepath",
                "name": "path",
                "description": (
                    "Will be looking for videos in /ethoscope_data/upload/video/"
                ),
                "default": "",
            },
        ],
    }

    def __init__(
        self,
        path: str,
        use_wall_clock: bool = False,
        drop_each: int = 1,
        max_duration: float | None = None,
    ) -> None:
        """Acquire frames from a video file.

        :param path: the path of the video file
        :param use_wall_clock: use real machine time instead of the video file
            timestamps (useful for prototyping)
        :param drop_each: keep only ``1/drop_each``'th frame
        :param max_duration: stop the stream after this many seconds
        """
        if not isinstance(path, str):
            raise CameraError("path to video must be a string")  # noqa: TRY003
        if not Path(path).exists():
            raise CameraError(f"'{path}' does not exist. No such file")  # noqa: TRY003

        self._path = path
        self._use_wall_clock = use_wall_clock

        self.capture = cv2.VideoCapture(path)
        w = self.capture.get(CAP_PROP_FRAME_WIDTH)
        h = self.capture.get(CAP_PROP_FRAME_HEIGHT)
        self._total_n_frames = self.capture.get(CAP_PROP_FRAME_COUNT)
        self._has_end_of_file = self._total_n_frames != 0.0

        super().__init__(drop_each=drop_each, max_duration=max_duration)
        self._resolution = (int(w), int(h))

        if self._use_wall_clock:
            self._start_time = time.time()
            self._start_monotonic = time.monotonic()
        else:
            self._start_time = 0.0
            self._start_monotonic = 0.0

    @property
    def path(self) -> str:
        return self._path

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def restart(self) -> None:
        self._close()
        self.capture = cv2.VideoCapture(self._path)
        if not self.capture.isOpened():
            raise CameraError(  # noqa: TRY003
                f"Could not reopen video file '{self._path}'"
            )
        self._frame_idx = 0
        if self._use_wall_clock:
            self._start_time = time.time()
            self._start_monotonic = time.monotonic()
        else:
            self._start_time = 0.0
            self._start_monotonic = 0.0

    def _next_image(self) -> np.ndarray | None:
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _time_stamp(self) -> float:
        if self._use_wall_clock:
            return time.monotonic() - self._start_monotonic
        return self.capture.get(CAP_PROP_POS_MSEC) / 1e3

    def is_last_frame(self) -> bool:
        return self._has_end_of_file and self._frame_idx >= self._total_n_frames

    def _close(self) -> None:
        self.capture.release()


class V4L2Camera(BaseCamera):
    _description: ClassVar[dict[str, Any]] = {
        "overview": (
            "Class to acquire frames from the V4L2 default interface (e.g. a webcam)."
        ),
        "arguments": [
            {
                "type": "number",
                "min": 0,
                "max": 4,
                "step": 1,
                "name": "device",
                "description": "The device to be open",
                "default": 0,
            },
        ],
    }

    def __init__(
        self,
        device: int = 0,
        target_fps: int | None = None,
        target_resolution: tuple[int, int] = (960, 720),
        drop_each: int = 1,
        max_duration: float | None = None,
    ) -> None:
        """Acquire a stream from a Video for Linux compatible device.

        :param device: the index of the device, or its path
        :param target_fps: the desired number of frames per second
        :param target_resolution: the desired resolution (W x H)
        :param drop_each: keep only ``1/drop_each``'th frame
        :param max_duration: stop the stream after this many seconds
        """
        self._target_fps = _resolve_fps(target_fps)

        self.capture = cv2.VideoCapture(device)
        self._warm_up()

        w, h = target_resolution
        if w < 0 or h < 0:
            self.capture.set(CAP_PROP_FRAME_WIDTH, 99999)
            self.capture.set(CAP_PROP_FRAME_HEIGHT, 99999)
        else:
            self.capture.set(CAP_PROP_FRAME_WIDTH, w)
            self.capture.set(CAP_PROP_FRAME_HEIGHT, h)
        self.capture.set(CAP_PROP_FPS, self._target_fps)

        time.sleep(1)
        _, first_frame = self.capture.read()
        if first_frame is None:
            raise CameraError(  # noqa: TRY003
                "Error whist retrieving video frame. Got None instead. "
                "Camera not plugged?"
            )
        if len(first_frame.shape) < 2:  # noqa: PLR2004 - grayscale frames are 2D
            raise CameraError(  # noqa: TRY003
                "Camera image is corrupted (less than 2 dimensions)"
            )

        self._frame = first_frame

        super().__init__(drop_each=drop_each, max_duration=max_duration)
        self._resolution = (first_frame.shape[1], first_frame.shape[0])
        self.fps = float(self._target_fps)
        self._start_time = time.time()
        self._start_monotonic = time.monotonic()

        if self._resolution != target_resolution:
            if w > 0 and h > 0:
                logger.warning(
                    f'Target resolution "{target_resolution}" could NOT be achieved. '
                    f'Effective resolution is "{self._resolution}"'
                )
            else:
                logger.info(f'Maximal effective resolution is "{self._resolution}"')

    def _warm_up(self) -> None:
        logger.info(f"{self!s} is warming up")
        time.sleep(2)

    def restart(self) -> None:
        self._frame_idx = 0
        self._start_time = time.time()
        self._start_monotonic = time.monotonic()

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def is_last_frame(self) -> bool:
        return False

    def _time_stamp(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _close(self) -> None:
        self.capture.release()

    def _next_image(self) -> np.ndarray | None:
        """Return the next frame, pacing acquisition towards the target FPS."""
        if self._frame_idx > 0:
            expected_time = self._start_monotonic + self._frame_idx / self._target_fps
            now = time.monotonic()
            self.fps = self._frame_idx / (now - self._start_monotonic)
            to_sleep = expected_time - now

            if to_sleep < 0:
                if self._frame_idx % 5000 == 0:
                    logger.warning(
                        f"The target FPS ({self._target_fps:f}) could not be reached. "
                        f"Effective FPS is about "
                        f"{self._frame_idx / (now - self._start_monotonic):f}"
                    )
                self.capture.grab()

            # drop frames until we go above the expected time
            while now < expected_time:
                self.capture.grab()
                now = time.monotonic()
        else:
            self.capture.grab()

        self.capture.retrieve(self._frame)
        if len(self._frame.shape) == 3:  # noqa: PLR2004 - BGR frames are 3D
            return cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        return self._frame


class Picamera2Driver:
    """Owns a picamera2 instance: tuning, configuration and lifecycle.

    The driver does not start acquisition itself; :class:`FrameProducer`
    calls :meth:`open` (which returns a configured, stopped camera) and
    :meth:`close` around its capture loop.
    """

    NOIR_TUNING_FILES: ClassVar[list[str]] = [
        "/usr/share/libcamera/ipa/rpi/vc4/imx708_noir.json",
        "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json",
    ]
    SENSOR_SIZE: ClassVar[tuple[int, int]] = (2304, 1296)

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: Picamera2 | None = None

    @property
    def target_fps(self) -> int:
        return self._config.target_fps

    def open(self) -> Picamera2:
        """Create, tune and configure a picamera2 instance."""
        Picamera2.set_logging(logging.ERROR)
        if self._config.noir:
            capture = self._open_with_noir_tuning()
        else:
            logger.info(
                "Creating Picamera2 instance with automatic tuning detection "
                "for dynamic light adaptation"
            )
            capture = Picamera2()

        target_w, target_h = self._config.target_resolution
        sensor_w, sensor_h = self.SENSOR_SIZE
        logger.info(
            f"Target resolution: {target_w}x{target_h}, "
            f"Sensor mode: {sensor_w}x{sensor_h} (full FoV), "
            f"fps: {self._config.target_fps}"
        )

        # Prioritise exposure over gain to minimise noise artefacts that
        # interfere with background subtraction tracking algorithms.
        camera_controls = {
            "FrameRate": self._config.target_fps,
            "ExposureTime": 45000,
            "HdrMode": 0,
            "AnalogueGain": self._config.gain,
            "AwbEnable": False,
            "AfMode": 0,
            "LensPosition": 8.5,
            "AeEnable": False,
        }

        config = capture.create_video_configuration(
            main={"size": (target_w, target_h), "format": "YUV420"},
            sensor={"output_size": (sensor_w, sensor_h)},
            buffer_count=2,
            controls=camera_controls,
        )
        capture.configure(config)
        self._capture = capture
        self._log_camera_status(capture)
        self._save_camera_info_if_available(capture)
        return capture

    def _open_with_noir_tuning(self) -> Picamera2:
        logger.info(
            "Creating Picamera2 instance with forced NoIR tuning for "
            "IR pass-through filter"
        )
        for tuning_file in self.NOIR_TUNING_FILES:
            try:
                capture = Picamera2(tuning=Picamera2.load_tuning_file(tuning_file))
            except Exception as e:  # noqa: BLE001 - library faults are arbitrary
                logger.debug(f"Failed to load tuning file {tuning_file}: {e}")
                continue
            logger.info(f"Successfully loaded NoIR tuning file: {tuning_file}")
            return capture
        logger.warning(
            "Failed to load any NoIR tuning file, falling back to automatic detection"
        )
        return Picamera2()

    def _log_camera_status(self, capture: Picamera2) -> None:
        try:
            exposure_time = capture.camera_controls.get("ExposureTime", "Unknown")
            analogue_gain = capture.camera_controls.get("AnalogueGain", "Unknown")
            logger.info(
                f"Camera control status - ExposureTime: {exposure_time}, "
                f"AnalogueGain: {analogue_gain}"
            )
        except Exception as e:  # noqa: BLE001 - library faults are arbitrary
            logger.warning(f"Could not check auto-exposure status: {e}")

    def _save_camera_info_if_available(self, capture: Picamera2) -> None:
        try:
            camera_info = capture.global_camera_info()
        except Exception as e:  # noqa: BLE001 - library faults are arbitrary
            logger.warning(f"Could not get camera info: {e}")
        else:
            if camera_info:
                _save_camera_info(camera_info[0])

    def close(self) -> None:
        """Stop and close the underlying picamera2 instance, if any."""
        if self._capture is None:
            return
        capture, self._capture = self._capture, None
        with contextlib.suppress(Exception):
            capture.stop()
        with contextlib.suppress(Exception):
            capture.close()


class VideoRecorder:
    """Chunked H264 recording helper used by :class:`FrameProducer`.

    Recordings are split into time-bounded chunks so a corrupted chunk never
    destroys an entire experiment, and a preview frame is pushed periodically
    so consumers keep receiving frames while recording.
    """

    CHUNK_DURATION = 300
    PREVIEW_REFRESH = 5

    def __init__(self, config: CameraConfig) -> None:
        self._video_prefix = config.video_prefix or ""
        self._target_resolution = config.target_resolution
        self.video_quality = config.quality
        self._file_index = 0
        self._last_computed_filename = ""
        self._encoder: H264Encoder | None = None
        self._video_time = 0.0
        self._refresh_interval = 0.0
        w, h = self._target_resolution
        self._preview_buffer = np.empty((h, w), dtype=np.uint8)

    def chunk_filename(
        self, fps: int | None = None, ext: str = "h264", current: bool = False
    ) -> str:
        """Return the filename of the next (or current) video chunk."""
        if current:
            return self._last_computed_filename
        self._file_index += 1
        w, h = self._target_resolution
        video_info = f"{w}x{h}@{fps or 0}fps-{self.video_quality}q"
        name = f"{self._video_prefix}_{video_info}_{self._file_index:05d}.{ext}"
        self._last_computed_filename = name
        return name

    def start(self, capture: Picamera2, fps: int) -> None:
        """Attach an H264 encoder to ``capture`` and start the first chunk."""
        from picamera2.encoders import (  # noqa: PLC0415 - importable only on device
            H264Encoder,
        )

        self._encoder = H264Encoder(bitrate=10000000)
        capture.start_encoder(self._encoder, self.chunk_filename(fps))
        self._video_time = time.monotonic()
        self._refresh_interval = time.monotonic()

    def rotate_if_needed(self, capture: Picamera2, fps: int) -> None:
        """Rotate to a new H264 chunk when the current one is old enough."""
        if time.monotonic() - self._video_time < self.CHUNK_DURATION:
            return
        logger.info("Splitting video recording into a new H264 chunk.")
        capture.stop_encoder()
        capture.start_encoder(self._encoder, self.chunk_filename(fps))
        self._video_time = time.monotonic()

    def preview_frame(self, capture: Picamera2, target_h: int) -> np.ndarray | None:
        """Return a fresh preview frame if one is due, otherwise ``None``."""
        if time.monotonic() - self._refresh_interval < self.PREVIEW_REFRESH:
            return None
        request = capture.capture_request()
        try:
            with MappedArray(request, "main") as frame:
                if frame.array is None:
                    return None
                np.copyto(self._preview_buffer, frame.array[:target_h, :])
        finally:
            request.release()
        self._refresh_interval = time.monotonic()
        return self._preview_buffer

    def stop(self, capture: Picamera2) -> None:
        """Detach the encoder from ``capture`` if one is attached."""
        if self._encoder is None:
            return
        capture.stop_encoder()
        self._encoder = None


class FrameProducer(threading.Thread):
    """Pump frames from a :class:`Picamera2Driver` on a background thread.

    Grayscale frames (the Y plane of the YUV420 output) are pushed into a
    bounded queue. If the consumer falls behind, frames are dropped rather
    than blocking the camera. Acquisition failures are captured in
    :attr:`error`; when the producer dies before delivering any frame, a
    ``None`` sentinel is pushed so the consumer can fail fast during
    initialisation.
    """

    def __init__(
        self,
        driver: Picamera2Driver,
        frame_queue: queue.Queue[np.ndarray | None],
        stop_event: threading.Event,
        config: CameraConfig,
        recorder: VideoRecorder | None = None,
    ) -> None:
        super().__init__(name="FrameProducer", daemon=True)
        self._driver = driver
        self._queue = frame_queue
        self._stop_event = stop_event
        self._config = config
        self._recorder = recorder
        self.error: CameraError | None = None

    def _put_frame(self, frame: np.ndarray) -> None:
        # drop the frame if the consumer is slow
        with contextlib.suppress(queue.Full):
            self._queue.put(frame, timeout=0.5)

    def run(self) -> None:
        try:
            capture = self._driver.open()
        except Exception as e:  # noqa: BLE001 - hardware faults are arbitrary
            self.error = CameraError(f"Camera hardware not available: {e}")
            self._queue.put(None)
            return

        delivered_frame = False
        try:
            target_h = self._config.target_resolution[1]
            capture.start()
            if self._recorder is not None:
                self._recorder.start(capture, self._driver.target_fps)
                while not self._stop_event.is_set():
                    self._recorder.rotate_if_needed(capture, self._driver.target_fps)
                    frame = self._recorder.preview_frame(capture, target_h)
                    if frame is not None:
                        self._put_frame(frame)
                        delivered_frame = True
            else:
                while not self._stop_event.is_set():
                    frame = capture.capture_array("main")
                    self._put_frame(frame[:target_h, :])
                    delivered_frame = True
        except Exception as e:
            logger.exception("Frame acquisition failed")
            self.error = CameraError(f"Frame acquisition failed: {e}")
            if not delivered_frame:
                self._queue.put(None)
        finally:
            if self._recorder is not None:
                self._recorder.stop(capture)
            self._driver.close()
            logger.warning("Camera frame producer stopped acquisition cleanly.")


class Picamera2Camera(BaseCamera):
    _description: ClassVar[dict[str, Any]] = {
        "overview": (
            "Default class to acquire frames from the raspberry pi camera "
            "asynchronously."
        ),
        "arguments": [],
    }

    hardware_recording = True

    def __init__(  # noqa: PLR0913, PLR0917 - public API dictated by the web interface
        self,
        target_fps: int | None = None,
        target_resolution: tuple[int, int] = (1280, 960),
        drop_each: int = 1,
        max_duration: float | None = None,
        video_prefix: str | None = None,
        record_video: bool = False,
        quality: int = 20,
        gain: float | None = None,
        noir: bool | None = None,
        **kwargs: object,
    ) -> None:
        """Acquire frames from the Raspberry Pi camera asynchronously.

        Frames are greyscale images captured on a background thread.

        :param target_fps: the desired number of frames per second
        :param target_resolution: the desired resolution (W x H)
        :param video_prefix: base path of the recorded video chunks
        :param record_video: record H264 video chunks while acquiring
        :param quality: H264 encoder quality (10 high, 40 low)
        :param gain: fixed analogue gain (``None`` uses the machine setting)
        :param noir: force NoIR tuning (``None`` uses the machine setting)
        """
        if kwargs:
            logger.warning(
                f"Picamera2Camera: ignoring unknown arguments: {sorted(kwargs)}"
            )

        self._init_kwargs = {
            "target_fps": target_fps,
            "target_resolution": target_resolution,
            "drop_each": drop_each,
            "max_duration": max_duration,
            "video_prefix": video_prefix,
            "record_video": record_video,
            "quality": quality,
            "gain": gain,
            "noir": noir,
        }

        fps = _resolve_fps(target_fps)
        self._record_video = video_prefix is not None and record_video
        config = CameraConfig(
            target_fps=fps,
            target_resolution=target_resolution,
            drop_each=drop_each,
            max_duration=max_duration,
            gain=gain if gain is not None else pi.get_gain_setting(),
            noir=noir if noir is not None else pi.get_noir_setting(),
            video_prefix=video_prefix,
            record_video=self._record_video,
            quality=quality,
        )

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._driver = Picamera2Driver(config)
        self._recorder = VideoRecorder(config) if self._record_video else None
        self._producer = FrameProducer(
            self._driver, self._queue, self._stop_event, config, recorder=self._recorder
        )
        self._producer.start()

        try:
            first_frame = self._queue.get(timeout=30)
        except queue.Empty as e:
            self._shutdown_producer()
            raise CameraError(  # noqa: TRY003
                "Camera initialization timeout: No frames received within 30 seconds. "
                "This may indicate a camera hardware issue or picamera2 "
                "compatibility problem."
            ) from e

        if first_frame is None:
            self._shutdown_producer()
            raise self._producer.error or CameraError(
                "Camera hardware not available. Video tracking and recording "
                "are disabled."
            )

        if len(first_frame.shape) < 2:  # noqa: PLR2004 - grayscale frames are 2D
            self._shutdown_producer()
            raise CameraError(  # noqa: TRY003
                "The camera image is corrupted (less that 2 dimensions)"
            )

        w, h = target_resolution
        super().__init__(drop_each=drop_each, max_duration=max_duration)
        self._resolution = (first_frame.shape[1], first_frame.shape[0])
        self._start_time = time.time()
        self._start_monotonic = time.monotonic()

        if self._resolution != target_resolution:
            if w > 0 and h > 0:
                logger.warning(
                    f'Target resolution "{target_resolution}" could NOT be achieved. '
                    f'Effective resolution is "{self._resolution}"'
                )
            else:
                logger.info(f'Maximal effective resolution is "{self._resolution}"')
        logger.info("Camera initialised")

    def _shutdown_producer(self) -> None:
        """Signal the producer to stop and release camera resources."""
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._producer.join(5)
        self._driver.close()
        gc.collect()

    def restart(self) -> None:
        self._frame_idx = 0
        self._start_time = time.time()
        self._start_monotonic = time.monotonic()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "init_kwargs": self._init_kwargs,
            "frame_idx": self._frame_idx,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        init_kwargs = cast("dict[str, Any]", state["init_kwargs"])
        self.__init__(**init_kwargs)
        self._frame_idx = int(state["frame_idx"])
        # Use the current time when resuming after a reboot, not the old
        # pickled time, so SQLite databases use a fresh timestamp.
        self._start_time = time.time()
        self._start_monotonic = time.monotonic()

    def is_opened(self) -> bool:
        return True

    def is_last_frame(self) -> bool:
        return False

    def _time_stamp(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _close(self) -> None:
        logger.info("Requesting grabbing process to stop!")
        self._shutdown_producer()

    def _next_image(self) -> np.ndarray | None:
        elapsed = self._time_stamp()
        self.fps = self._frame_idx / elapsed if elapsed > 0.0 else 0.0
        try:
            frame = self._queue.get(timeout=30)
        except queue.Empty:
            if self._producer.error is not None:
                raise self._producer.error from None
            raise CameraError(  # noqa: TRY003
                "Could not get frame from camera: producer stalled"
            ) from None
        if frame is None:
            raise self._producer.error or CameraError(
                "Camera acquisition terminated unexpectedly"
            )
        return frame


if __name__ == "__main__":
    camera = Picamera2Camera(target_fps=10, target_resolution=(960, 720))
    with camera:
        for t_ms, frame in camera:
            logger.info(f"{t_ms} {frame.shape}")
