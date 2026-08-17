"""Drawers annotate tracked frames and can record or display them.

This module provides the drawer classes used by
:class:`~ethoscope.core.monitor.Monitor` to visualise tracking results.
"""
# author: quentin
# refactor: moomurrs

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, override

import cv2
import numpy as np
from cv2 import LINE_AA

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Any, Self

    from ethoscope.core.roi import ROI
    from ethoscope.core.tracking_unit import TrackingUnit


_LOGGER = logging.getLogger(__name__)

type PositionMap = dict[int, list[dict[str, Any]]]
type ReferencePoints = Sequence[tuple[float, float]] | None
type BGR = tuple[int, int, int]

_BLACK: BGR = (0, 0, 0)
_WHITE: BGR = (255, 255, 255)
_RED: BGR = (0, 0, 255)
_BLUE: BGR = (255, 0, 0)
_GREEN: BGR = (0, 255, 0)
_YELLOW: BGR = (0, 255, 255)

_BGR_NDIM = 3
_EDGE_MARGIN = 15
_INDICATOR_INSET_X = 20
_INDICATOR_LIFT = 6
_INDICATOR_RADIUS = 12
_LABEL_FONT_SCALE = 1.2
_LABEL_OFFSET_X = 10
_LABEL_OFFSET_Y = 40
_MIN_ROI_EXTENT = 50
_DEFAULT_ROI_EXTENT = 100
_PREVIEW_INTERVAL_SECONDS = 1.0


class BaseDrawer:
    """Template class to annotate and save processed frames.

    Subclasses implement :meth:`_annotate_frame` to define how frames are
    annotated. Annotated frames can be written to a video file and/or
    displayed in a live window. Instances can be used as context managers,
    which guarantees the output video and display window are released::

        with DefaultDrawer(video_out="out.avi") as drawer:
            drawer.draw(frame, positions, tracking_units)
    """

    def __init__(
        self,
        video_out: str | None = None,
        draw_frames: bool = True,
        video_out_fourcc: str = "DIVX",
        video_out_fps: float = 25,
    ) -> None:
        """Initialise the drawer.

        Args:
            video_out: Path to the output video file (.avi); no video is
                written when ``None``.
            draw_frames: Whether frames should be displayed on the screen
                (a new window will be created).
            video_out_fourcc: When setting ``video_out``, the codec used to
                save the output video (see `fourcc
                <http://www.fourcc.org/codecs.php>`_).
            video_out_fps: When setting ``video_out``, the output fps,
                typically the same as the input fps.
        """
        self._video_out: str | None = video_out
        self._draw_frames: bool = draw_frames
        self._live_window_name: str = f"ethoscope_{os.getpid()}"
        self._video_out_fourcc: str = video_out_fourcc
        self._video_out_fps: float = video_out_fps
        self._video_writer: cv2.VideoWriter | None = None
        self._last_drawn_frame: np.ndarray | None = None
        self._last_preview_time: float | None = None

        if draw_frames:
            cv2.namedWindow(self._live_window_name, cv2.WINDOW_AUTOSIZE)

    def _annotate_frame(
        self,
        img: np.ndarray,
        positions: PositionMap,
        tracking_units: list[TrackingUnit],
        reference_points: ReferencePoints = None,
    ) -> None:
        """Define how frames should be annotated.

        The ``img`` array, which is passed by reference, is meant to be
        modified by this method.

        Args:
            img: The BGR frame that was just processed.
            positions: Positions resulting from analysis of the frame, keyed
                by ROI index.
            tracking_units: The tracking units corresponding to the positions.
            reference_points: Optional reference points to mark.
        """
        raise NotImplementedError

    @property
    def last_drawn_frame(self) -> np.ndarray | None:
        """The most recently drawn (annotated) BGR frame, if any."""
        return self._last_drawn_frame

    def draw(
        self,
        img: np.ndarray,
        positions: PositionMap,
        tracking_units: list[TrackingUnit],
        reference_points: ReferencePoints = None,
    ) -> None:
        """Annotate a frame and optionally display and record it.

        The input frame is copied into a reusable BGR buffer (pre-allocated
        once per frame size/dtype) which is then annotated in place.
        Preview-only rendering is throttled to once per second; display and
        video output remain full rate.

        Args:
            img: The frame that was just processed (grayscale or BGR).
            positions: Positions resulting from analysis of the frame,
                keyed by ROI index.
            tracking_units: The tracking units corresponding to the positions.
            reference_points: Optional reference points to mark.
        """
        preview_time: float | None = None
        if not self._draw_frames and self._video_out is None:
            preview_time = time.monotonic()
            if (
                self._last_preview_time is not None
                and preview_time - self._last_preview_time
                < _PREVIEW_INTERVAL_SECONDS
            ):
                return

        buffer = self._last_drawn_frame
        if (
            buffer is None
            or buffer.shape[:2] != img.shape[:2]
            or buffer.dtype != img.dtype
        ):
            buffer = np.empty((*img.shape[:2], 3), dtype=img.dtype)
            self._last_drawn_frame = buffer

        if img.ndim == _BGR_NDIM:
            buffer[:] = img
        else:
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR, dst=buffer)

        self._annotate_frame(buffer, positions, tracking_units, reference_points)

        if self._draw_frames:
            cv2.imshow(self._live_window_name, buffer)
            cv2.waitKey(1)

        self._write_video_frame(buffer)

        if preview_time is not None:
            self._last_preview_time = preview_time

    def _write_video_frame(self, frame: np.ndarray) -> None:
        """Lazily open the output video and append a frame to it."""
        if self._video_out is None:
            return
        if self._video_writer is None:
            self._video_writer = cv2.VideoWriter(
                self._video_out,
                cv2.VideoWriter.fourcc(*self._video_out_fourcc),
                self._video_out_fps,
                (frame.shape[1], frame.shape[0]),
            )
        self._video_writer.write(frame)

    def close(self) -> None:
        """Release the output video and close the display window.

        Safe to call multiple times.
        """
        if self._draw_frames:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self._draw_frames = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class NullDrawer(BaseDrawer):
    """A drawer that does nothing.

    No video writing, no annotation, no display on the screen.
    """

    def __init__(self) -> None:
        """Initialise a drawer that performs no drawing or recording."""
        super().__init__(draw_frames=False)

    @override
    def _annotate_frame(
        self,
        img: np.ndarray,
        positions: PositionMap,
        tracking_units: list[TrackingUnit],
        reference_points: ReferencePoints = None,
    ) -> None:
        """Do nothing."""


class DefaultDrawer(BaseDrawer):
    """The default drawer.

    It draws ellipses on the detected objects and polygons around ROIs.
    When an "interaction" (see
    :class:`~ethoscope.stimulators.stimulators.BaseInteractor`) happens
    within a ROI, the ellipse is blue, red otherwise. Stimulator state is
    shown as an indicator in the top-right corner of each ROI.
    """

    def __init__(
        self,
        video_out: str | None = None,
        draw_frames: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the default drawer.

        Args:
            video_out: Path to the output video file (.avi).
            draw_frames: Whether frames should be displayed on the screen
                (a new window will be created).
            **kwargs: Forwarded to :class:`BaseDrawer`.
        """
        super().__init__(video_out=video_out, draw_frames=draw_frames, **kwargs)
        self._annotate_logged = False
        self._last_stimulator_states: dict[int, str] = {}
        self._logged_stimulator_rois: set[int] = set()
        self._method_checks_logged: set[str] = set()

    @override
    def _annotate_frame(
        self,
        img: np.ndarray | None,
        positions: PositionMap,
        tracking_units: list[TrackingUnit],
        reference_points: ReferencePoints = None,
    ) -> None:
        """Annotate frames with information about ROIs and tracked objects."""
        if img is None:
            return

        if not self._annotate_logged:
            _LOGGER.info(
                "DefaultDrawer._annotate_frame first call with"
                f" {len(tracking_units)} tracking units"
            )
            self._annotate_logged = True

        self._draw_reference_points(img, reference_points)

        for track_u in tracking_units:
            roi = track_u.roi
            self._draw_roi_label(img, roi)
            self._draw_roi_contour(img, roi)
            self._draw_stimulator_state(img, track_u)
            self._draw_positions(img, positions, roi)

    @staticmethod
    def _draw_reference_points(
        img: np.ndarray, reference_points: ReferencePoints
    ) -> None:
        """Draw a green cross marker at each reference point."""
        if reference_points is None:
            return
        for p in reference_points:
            cv2.drawMarker(
                img,
                (int(p[0]), int(p[1])),
                color=_GREEN,
                markerType=cv2.MARKER_CROSS,
                thickness=2,
            )

    @staticmethod
    def _draw_roi_label(img: np.ndarray, roi: ROI) -> None:
        """Draw the ROI index at its top-left corner, with a black outline."""
        x, y = roi.offset
        text_x = int(x + _LABEL_OFFSET_X)
        text_y = int(y + _LABEL_OFFSET_Y)
        roi_text = str(roi.idx)
        cv2.putText(
            img,
            roi_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            _LABEL_FONT_SCALE,
            _BLACK,
            4,
        )
        cv2.putText(
            img,
            roi_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            _LABEL_FONT_SCALE,
            _WHITE,
            2,
        )

    @staticmethod
    def _draw_roi_contour(img: np.ndarray, roi: ROI) -> None:
        """Draw the ROI polygon outline, black over green."""
        cv2.drawContours(img, [roi.polygon], -1, _BLACK, 3, LINE_AA)
        cv2.drawContours(img, [roi.polygon], -1, _GREEN, 1, LINE_AA)

    def _draw_positions(
        self, img: np.ndarray, positions: PositionMap, roi: ROI
    ) -> None:
        """Draw an ellipse around each detected object in the ROI."""
        pos_list = positions.get(roi.idx)
        if not pos_list:
            return
        for pos in pos_list:
            colour = _BLUE if pos.get("has_interacted", False) else _RED
            cv2.ellipse(
                img,
                ((pos["x"], pos["y"]), (pos["w"], pos["h"]), pos["phi"]),
                colour,
                1,
                LINE_AA,
            )

    def _draw_stimulator_state(self, img: np.ndarray, track_u: TrackingUnit) -> None:
        """Draw the stimulator state indicator for a single tracking unit."""
        stimulator = track_u.stimulator
        roi = track_u.roi

        if stimulator is None:
            _LOGGER.debug(f"ROI {roi.idx}: No stimulator assigned")
            return

        stimulator_type = type(stimulator).__name__
        if roi.idx not in self._logged_stimulator_rois:
            _LOGGER.info(f"ROI {roi.idx}: Using stimulator type = {stimulator_type}")
            self._logged_stimulator_rois.add(roi.idx)

        has_method = hasattr(stimulator, "get_stimulator_state")
        if stimulator_type not in self._method_checks_logged:
            _LOGGER.info(
                f"Stimulator {stimulator_type} has get_stimulator_state(): {has_method}"
            )
            self._method_checks_logged.add(stimulator_type)

        if not has_method:
            self._draw_unsupported_indicator(img, roi, stimulator_type)
            return

        try:
            state = stimulator.get_stimulator_state()
        except Exception:
            _LOGGER.exception(f"ROI {roi.idx}: Error calling get_stimulator_state()")
            self._draw_stimulator_indicator(img, roi, "error")
            return

        self._log_stimulator_state(roi, stimulator_type, state)
        self._draw_stimulator_indicator(img, roi, state)

    def _log_stimulator_state(self, roi: ROI, stimulator_type: str, state: str) -> None:
        """Log stimulator state transitions once per change."""
        if self._last_stimulator_states.get(roi.idx) == state:
            return
        _LOGGER.info(f"ROI {roi.idx}: {stimulator_type} state = {state}")
        self._last_stimulator_states[roi.idx] = state

    def _draw_unsupported_indicator(
        self, img: np.ndarray, roi: ROI, stimulator_type: str
    ) -> None:
        """Warn about and mark stimulators without state reporting."""
        if stimulator_type == "DefaultStimulator":
            return
        _LOGGER.warning(
            f"ROI {roi.idx}: {stimulator_type} doesn't support"
            " get_stimulator_state() - consider updating"
        )
        x, y = roi.offset
        center_x = max(_EDGE_MARGIN, min(img.shape[1] - _EDGE_MARGIN, int(x + 30)))
        center_y = max(_EDGE_MARGIN, min(img.shape[0] - _EDGE_MARGIN, int(y + 25)))
        cv2.circle(img, (center_x, center_y), 10, _YELLOW, -1)
        cv2.putText(
            img,
            "?",
            (center_x - 3, center_y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            _BLACK,
            1,
        )

    def _draw_stimulator_indicator(
        self, img: np.ndarray, roi: ROI, stimulator_state: str
    ) -> None:
        """Draw the stimulator state indicator in the corner of a ROI.

        Args:
            img: Frame to draw on (modified in place).
            roi: The ROI whose corner hosts the indicator.
            stimulator_state: "inactive" (empty circle), "scheduled" (white
                fill), "stimulating" (blue fill); any other value is drawn
                as a red warning indicator.
        """
        x, y = roi.offset
        roi_width, _ = self._roi_extent(roi)

        label_y = int(y + _LABEL_OFFSET_Y) - _INDICATOR_LIFT
        center_x = max(
            _EDGE_MARGIN,
            min(
                img.shape[1] - _EDGE_MARGIN,
                int(x + roi_width - _INDICATOR_INSET_X),
            ),
        )
        center_y = max(_EDGE_MARGIN, min(img.shape[0] - _EDGE_MARGIN, label_y))
        center = (center_x, center_y)

        match stimulator_state:
            case "inactive":
                cv2.circle(img, center, _INDICATOR_RADIUS, _BLACK, 2)
            case "scheduled":
                cv2.circle(img, center, _INDICATOR_RADIUS, _WHITE, -1)
                cv2.circle(img, center, _INDICATOR_RADIUS, _BLACK, 1)
            case "stimulating":
                cv2.circle(img, center, _INDICATOR_RADIUS, _BLUE, -1)
                cv2.circle(img, center, _INDICATOR_RADIUS, _BLACK, 1)
            case _:
                cv2.circle(img, center, _INDICATOR_RADIUS, _RED, -1)
                cv2.circle(img, center, _INDICATOR_RADIUS, _BLACK, 1)
                _LOGGER.warning(
                    f"Drew UNKNOWN state indicator ({stimulator_state}) at {center}"
                )

    @staticmethod
    def _roi_extent(roi: ROI) -> tuple[int, int]:
        """Return the (width, height) span of the ROI polygon.

        Values are clamped to a minimum of 50px and fall back to 100px when
        the polygon is empty or malformed.
        """
        try:
            points = np.asarray(roi.polygon).reshape(-1, 2)
        except (TypeError, ValueError):
            return _DEFAULT_ROI_EXTENT, _DEFAULT_ROI_EXTENT
        if points.size == 0:
            return _DEFAULT_ROI_EXTENT, _DEFAULT_ROI_EXTENT
        x_coords, y_coords = points[:, 0], points[:, 1]
        return (
            max(int(x_coords.max() - x_coords.min()), _MIN_ROI_EXTENT),
            max(int(y_coords.max() - y_coords.min()), _MIN_ROI_EXTENT),
        )
