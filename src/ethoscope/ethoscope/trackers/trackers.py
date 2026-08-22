# author: quentin

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from ethoscope.core.variables import IsInferredVariable
from ethoscope.utils.description import DescribedObject

if TYPE_CHECKING:
    from ethoscope.core.data_point import DataPoint
    from ethoscope.core.roi import ROI

_LOGGER: Final = logging.getLogger(__name__)

_DEFAULT_MAX_HISTORY_LENGTH_MS: Final[int] = 250_000
_DEFAULT_MAX_INFER_MS: Final[int] = 30_000
_MIN_HISTORY_FOR_PRUNE: Final[int] = 2

type TrackerImage = NDArray[np.uint8]


class NoPositionError(Exception):
    """Raised to abort tracking and fall back to position inference.

    When raised inside :meth:`BaseTracker._find_position`, the tracker
    returns the last known position annotated as inferred, when available.
    """


class BaseTracker(DescribedObject):
    """Template tracker that locates animals inside a single ROI.

    Concrete subclasses must implement :meth:`_find_position`. The public
    :meth:`track` method orchestrates ROI cropping, position inference,
    annotation with :class:`~ethoscope.core.variables.IsInferredVariable`,
    and bounded history management.

    This class holds no external resources (sockets, files, handles) and
    therefore requires no explicit close or context-manager protocol.
    Instances are typically owned by :class:`~ethoscope.core.tracking_unit.TrackingUnit`
    for the lifetime of an experiment and are reclaimed by the garbage
    collector when the unit is discarded.

    Attributes:
        _max_history_length: Maximum time span (ms) of stored positions.
    """

    def __init__(self, roi: ROI, data: object | None = None) -> None:
        """Initialize the tracker.

        Args:
            roi: Region of interest the tracker operates on.
            data: Optional dataset for pre-trained algorithms.
        """
        self._positions: deque[list[DataPoint]] = deque()
        self._times: deque[int] = deque()
        self._data: object | None = data
        self._roi: ROI = roi
        self._last_non_inferred_time: int = 0
        self._last_time_point: int = 0
        self._max_history_length: int = _DEFAULT_MAX_HISTORY_LENGTH_MS

    def track(self, t: int, img: TrackerImage) -> list[DataPoint]:
        """Locate the animal at time ``t`` within ``img``.

        Args:
            t: Timestamp in milliseconds.
            img: Full frame image.

        Returns:
            List of :class:`~ethoscope.core.data_point.DataPoint` for this
            ROI. An empty list means no animal was located and no inference
            was possible.

        Raises:
            TypeError: If :meth:`_find_position` returns a non-list value.
        """
        sub_img, mask = self._roi.apply(img)
        self._last_time_point = t

        try:
            raw_points = self._find_position(sub_img, mask, t)
            points = self._validate_points(raw_points)
            if not points:
                return []
            self._last_non_inferred_time = t
            self._annotate(points, inferred=False)
        except NoPositionError:
            if not self._positions:
                return []
            points = self._infer_position(t)
            if not points:
                return []
            self._annotate(points, inferred=True)

        self._append_history(points, t)
        return points

    def _infer_position(
        self, t: int, max_time: int = _DEFAULT_MAX_INFER_MS
    ) -> list[DataPoint]:
        """Return the last known position if still within ``max_time``.

        Args:
            t: Current timestamp in milliseconds.
            max_time: Maximum age (ms) before inference is abandoned.

        Returns:
            The last stored position list, or an empty list when inference
            is not possible.
        """
        if not self._times:
            return []
        if t - self._last_non_inferred_time > max_time:
            return []
        return self._positions[-1]

    @property
    def positions(self) -> deque[list[DataPoint]]:
        """Last positions, bounded to ``_max_history_length`` ms.

        Returns:
            Deque of position lists retained for the configured history window.
        """
        return self._positions

    def xy_pos(self, i: int) -> DataPoint:
        """Return the first data point at history index ``i``.

        Args:
            i: Index into the history deque (supports negative indexing).

        Returns:
            The first :class:`~ethoscope.core.data_point.DataPoint` at
            position ``i``.

        Raises:
            IndexError: If the history is empty or ``i`` is out of range.
        """
        return self._positions[i][0]

    @property
    def last_time_point(self) -> int:
        """Last timestamp passed to :meth:`track`.

        This is updated even when position is inferred or no animal is found.

        Returns:
            Timestamp in milliseconds.
        """
        return self._last_time_point

    @property
    def times(self) -> deque[int]:
        """Timestamps corresponding to :attr:`positions`.

        Returns:
            Deque of timestamps (ms) aligned with :attr:`positions`.
        """
        return self._times

    def _find_position(
        self, img: TrackerImage, mask: NDArray[np.uint8], t: int
    ) -> list[DataPoint]:
        """Locate animals in the ROI crop.

        Args:
            img: Cropped ROI image.
            mask: Binary ROI mask.
            t: Timestamp in milliseconds.

        Returns:
            List of detected data points.

        Raises:
            NoPositionError: If no position can be determined.
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError

    def _validate_points(self, points: object) -> list[DataPoint]:
        """Validate that tracker output is a list.

        Args:
            points: Raw return value from :meth:`_find_position`.

        Returns:
            The validated list of data points.

        Raises:
            TypeError: If ``points`` is not a list.
        """
        if not isinstance(points, list):
            raise TypeError(  # noqa: TRY003
                "tracking algorithms are expected to return a LIST of DataPoints"
            )
        return points  # type: ignore[return-value]

    def _annotate(self, points: list[DataPoint], inferred: bool) -> None:
        """Annotate each data point with its inference flag.

        Args:
            points: Data points to annotate in place.
            inferred: Whether the position was inferred.
        """
        variable = IsInferredVariable(inferred)
        for point in points:
            point.append(variable)

    def _append_history(self, points: list[DataPoint], t: int) -> None:
        """Append a result and prune history beyond the time window.

        Args:
            points: Validated (and annotated) data points.
            t: Timestamp in milliseconds.
        """
        self._positions.append(points)
        self._times.append(t)
        self._prune_history()

    def _prune_history(self) -> None:
        """Drop oldest entry when history exceeds the time window."""
        if len(self._times) > _MIN_HISTORY_FOR_PRUNE and (
            self._times[-1] - self._times[0]
        ) > self._max_history_length:
            self._positions.popleft()
            self._times.popleft()
            _LOGGER.debug(
                "Pruned tracker history: window %d ms exceeds %d ms",
                self._times[-1] - self._times[0] if self._times else 0,
                self._max_history_length,
            )
