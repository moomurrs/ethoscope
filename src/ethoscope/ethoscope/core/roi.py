"""Region-of-interest (ROI) primitives for the tracking pipeline.

An :class:`ROI` wraps a single filled polygon and provides the binary mask,
bounding rectangle, and cropping helpers used to isolate individual arenas
(or tubes) from acquired frames.

Masks and polygons are exposed as **read-only views** — attempting to mutate
them raises ``ValueError`` — so that callers operating at frame rate cannot
accidentally corrupt ROI invariants. :meth:`ROI.bounding_rect` is deprecated
in favour of the :attr:`ROI.rectangle` property and now emits
``FutureWarning`` (visible by default).
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, ClassVar, Final

import cv2
import numpy as np

from ethoscope.utils.debug import EthoscopeException

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Number of dimensions of a flat ``(N, 2)`` polygon array.
_FLAT_FORM_NDIM: Final = 2
#: Number of dimensions of a contour-form ``(N, 1, 2)`` polygon array.
_CONTOUR_FORM_NDIM: Final = 3
#: Number of coordinates per polygon point.
_POINT_NCOORDS: Final = 2
#: Minimum number of dimensions of a croppable image.
_MIN_IMAGE_NDIM: Final = 2
#: Minimum number of points for a non-degenerate polygon.
_MIN_POLYGON_POINTS: Final = 3

#: Upright bounding box formatted as ``(x, y, w, h)``.
type Rectangle = tuple[int, int, int, int]


def _as_readonly(
    view: NDArray[np.uint8] | NDArray[np.int32],
) -> NDArray[np.uint8] | NDArray[np.int32]:
    """Return *view* with ``writeable=False`` so mutation raises.

    The returned array shares memory with *view* — no copy is made.
    """
    view.flags.writeable = False
    return view  # type: ignore[return-value]


def _normalize_polygon(polygon: ArrayLike) -> NDArray[np.int32]:
    """Convert *polygon* to the OpenCV contour form ``(N, 1, 2)`` of ``int32``.

    Floating-point inputs are rounded to the nearest integer (``rint``) before
    coercion — this keeps the manual-template path (which historically used
    ``float32``) working without silently truncating towards zero.

    Args:
        polygon: Flat ``(N, 2)`` or contour-form ``(N, 1, 2)`` point array.

    Returns:
        An owned ``(N, 1, 2)`` ``int32`` copy of *polygon*.

    Raises:
        ValueError: If the polygon is empty, degenerate, or has an invalid
            shape.
    """
    poly = np.asarray(polygon)
    if poly.size == 0:
        msg = "ROI polygon must contain at least one point"
        raise ValueError(msg)
    if poly.ndim < _FLAT_FORM_NDIM:
        msg = f"Invalid polygon shape {poly.shape}; expected (N, 2) or (N, 1, 2)"
        raise ValueError(msg)
    # Coerce to int32: round floats, direct cast for ints.
    if poly.dtype.kind == "f":
        poly = np.rint(poly).astype(np.int32, copy=False)
    else:
        poly = poly.astype(np.int32, copy=False)
    # Ensure owned copy — caller must not be able to mutate internal state
    # through the original array.
    poly = np.array(poly, copy=True)
    if poly.ndim == _FLAT_FORM_NDIM:
        poly = poly.reshape((poly.shape[0], 1, poly.shape[1]))
    # Handle (1, N, 2) form produced by TargetGridROIBuilder (ct reshaped as (1,4,2))
    if (
        poly.ndim == _CONTOUR_FORM_NDIM
        and poly.shape[0] == 1
        and poly.shape[1] >= _MIN_POLYGON_POINTS
        and poly.shape[2] == _POINT_NCOORDS
    ):
        poly = poly[0].reshape((poly.shape[1], 1, _POINT_NCOORDS))
    if poly.ndim != _CONTOUR_FORM_NDIM or poly.shape[-1] != _POINT_NCOORDS:
        msg = f"Invalid polygon shape {poly.shape}; expected (N, 2) or (N, 1, 2)"
        raise ValueError(msg)
    if poly.shape[0] < _MIN_POLYGON_POINTS:
        msg = f"ROI polygon must have at least 3 points, got {poly.shape[0]}"
        raise ValueError(msg)
    return poly  # type: ignore[return-value]


def _build_mask(polygon: NDArray[np.int32]) -> tuple[NDArray[np.uint8], Rectangle]:
    """Rasterise *polygon* into a local binary mask.

    Args:
        polygon: Contour-form ``(N, 1, 2)`` ``int32`` polygon.

    Returns:
        Tuple ``(mask, rectangle)`` where *mask* is the filled ``uint8``
        mask local to the ROI and *rectangle* is its ``(x, y, w, h)``
        bounding box in parent-frame coordinates.

    Raises:
        ValueError: If OpenCV fails to compute the bounding rectangle
            (e.g. degenerate polygon).
    """
    try:
        x, y, w, h = cv2.boundingRect(polygon)
    except cv2.error as exc:
        msg = (
            f"Failed to compute bounding rectangle for polygon shape "
            f"{polygon.shape}: {exc}"
        )
        raise ValueError(msg) from exc
    mask: NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [polygon], 0, (255,), -1, offset=(-x, -y))
    return mask, (int(x), int(y), int(w), int(h))


def _clamp_to_image(
    rect: Rectangle,
    img_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clamp an ``(x, y, w, h)`` rectangle to image boundaries.

    Args:
        rect: Rectangle in parent-frame coordinates.
        img_hw: Image ``(height, width)``.

    Returns:
        Clamped corner coordinates ``(x1, y1, x2, y2)``.
    """
    x, y, w, h = rect
    img_h, img_w = img_hw
    return max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)


class ROI:
    """A region of interest (ROI) defined by a single polygon.

    Internally, ROIs are single polygons: they cannot have holes. The
    polygon is rasterised into an opaque binary mask used to exclude
    off-target pixels (i.e. cross-ROI information) during tracking.

    Instances are plain value objects holding no external resources, so
    they require neither closing nor explicit disposal. :attr:`mask` and
    :attr:`polygon` are exposed as read-only views.

    ``__slots__`` is used to reduce per-instance overhead (trackers hold
    many ROIs) and to guard against accidental attribute creation.
    ``__weakref__`` is included so ROIs remain weak-referenceable when
    needed by caches or diagnostics.
    """

    __slots__: ClassVar[tuple[str, ...]] = (
        "__weakref__",
        "_hierarchy",
        "_idx",
        "_mask",
        "_polygon",
        "_rectangle",
        "_regions",
        "_value",
    )

    def __init__(
        self,
        polygon: ArrayLike,
        idx: int,
        value: int | None = None,
        _orientation: object | None = None,
        regions: NDArray[np.int32] | Sequence[NDArray[np.int32]] | None = None,
        hierarchy: NDArray[np.int32] | None = None,
    ) -> None:
        """Initialize a region of interest.

        Args:
            polygon: An array of points, either flat ``(N, 2)`` or
                contour-form ``(N, 1, 2)``. Floats are rounded to nearest
                ``int32``.
            idx: The index of this ROI.
            value: An optional value saved for this ROI (e.g. to define left
                and right sides). Defaults to ``idx``.
            _orientation: Unused placeholder retained for positional
                compatibility; orientation support is not implemented yet.
            regions: Optional sub-region contours within the ROI. When given,
                they are stored as-is together with *hierarchy*.
            hierarchy: The contour hierarchy matching *regions*.

        Raises:
            ValueError: If ``regions`` and ``hierarchy`` are not both set or
                both ``None``, or if *polygon* is degenerate.
        """
        if (regions is None) ^ (hierarchy is None):
            msg = "regions and hierarchy must be both set or both None"
            raise ValueError(msg)
        self._polygon: NDArray[np.int32] = _normalize_polygon(polygon)
        self._mask, self._rectangle = _build_mask(self._polygon)
        self._idx = idx
        self._value = idx if value is None else value

        if regions is None:
            self._regions: NDArray[np.int32] | Sequence[NDArray[np.int32]] = (
                self._polygon
            )
            self._hierarchy: NDArray[np.int32] | None = None
        else:
            self._regions = regions  # type: ignore[assignment]
            self._hierarchy = hierarchy

    def __repr__(self) -> str:
        """Return a concise debug representation of this ROI."""
        return f"{type(self).__name__}(idx={self._idx}, rectangle={self._rectangle})"

    @property
    def idx(self) -> int:
        """The index of this ROI."""
        return self._idx

    def bounding_rect(self) -> Rectangle:
        """Return the upright bounding rectangle ``(x, y, w, h)``.

        .. deprecated::
            Use the :attr:`rectangle` property instead. This method
            previously raised :class:`NotImplementedError`; it now delegates
            to :attr:`rectangle` for backward compatibility and emits
            ``FutureWarning`` (visible by default).

        Returns:
            The bounding rectangle as ``(x, y, w, h)``.
        """
        warnings.warn(
            "bounding_rect() is deprecated; use the 'rectangle' property instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self._rectangle

    @property
    def mask(self) -> NDArray[np.uint8]:
        """The mask as a single-channel, ``uint8`` read-only view."""
        return _as_readonly(self._mask.view())  # type: ignore[return-value]

    @property
    def offset(self) -> tuple[int, int]:
        """The ``(x, y)`` offset of the ROI relative to its parent frame."""
        return self._rectangle[0], self._rectangle[1]

    @property
    def polygon(self) -> NDArray[np.int32]:
        """The internal contour-form ``(N, 1, 2)`` read-only polygon."""

        return _as_readonly(self._polygon.view())  # type: ignore[return-value]

    @property
    def longest_axis(self) -> float:
        """The value of the longest axis (either width or height)."""
        _, _, w, h = self._rectangle
        return float(max(w, h))

    @property
    def rectangle(self) -> Rectangle:
        """The upright bounding rectangle ``(x, y, w, h)``.

        ``x`` and ``y`` are the coordinates of the top-left corner in
        parent-frame coordinates.
        """
        return self._rectangle

    def get_feature_dict(self) -> dict[str, int]:
        """Return the geometric features of this ROI.

        Returns:
            A dictionary with the following fields: ``"x"``, ``"y"``, ``"w"``,
            ``"h"``, ``"value"``, and ``"idx"``.
        """
        x, y, w, h = self._rectangle
        return {"x": x, "y": y, "w": w, "h": h, "value": self._value, "idx": self.idx}

    def set_value(self, new_val: int) -> None:
        """Assign a new value to this ROI.

        Args:
            new_val: The new semantic value (e.g. a side identifier).
        """
        self._value = new_val

    @property
    def value(self) -> int:
        """The value of this ROI (defaults to :attr:`idx`)."""
        return self._value

    def apply(
        self, img: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Cut an image where the ROI is defined.

        The crop window is clamped to the image boundaries, and the mask is
        cropped by the complementary offset so that image and mask pixels
        stay aligned even for ROIs reaching past the frame edges.

        Args:
            img: An image, typically one or three channels of ``uint8``.
                A read-only view is returned for the mask — mutating it
                raises ``ValueError``.

        Returns:
            Tuple ``(cropped_image, mask)`` with identical spatial
            dimensions. The mask is a read-only view; mutating it raises.

        Raises:
            EthoscopeException: If the ROI does not intersect *img*, if the
                image has fewer than two dimensions, or if mask and image
                dimensions diverge unexpectedly.
        """
        if img.ndim < _MIN_IMAGE_NDIM:
            msg = f"Expected an image with at least 2 dimensions, got {img.ndim}"
            raise EthoscopeException(msg, img)

        x, y, w, h = self._rectangle
        x1, y1, x2, y2 = _clamp_to_image(self._rectangle, (img.shape[0], img.shape[1]))
        width, height = x2 - x1, y2 - y1

        if width <= 0 or height <= 0:
            msg = f"ROI {self.get_feature_dict()} does not intersect the image"
            raise EthoscopeException(msg, img)

        if width != w or height != h:
            _LOGGER.debug(
                "ROI %d clipped to image bounds (%dx%d -> %dx%d)",
                self._idx,
                w,
                h,
                width,
                height,
            )

        out: NDArray[np.uint8] = img[y1:y2, x1:x2]
        # The mask window skips exactly the columns/rows clipped off-frame.
        mask = self._crop_mask(x1 - x, y1 - y, width, height)

        if mask.shape != out.shape[:2]:
            msg = (
                f"Failed to crop ROI {self.get_feature_dict()}: mask shape "
                f"{mask.shape} does not match image shape {out.shape[:2]}"
            )
            raise EthoscopeException(msg, img)

        return out, mask

    def _crop_mask(
        self, dx: int, dy: int, width: int, height: int
    ) -> NDArray[np.uint8]:
        """Crop the ROI mask to a ``(width, height)`` window at offset ``(dx, dy)``.

        Args:
            dx: Horizontal offset into the mask (non-negative after clamping).
            dy: Vertical offset into the mask (non-negative after clamping).
            width: Requested window width.
            height: Requested window height.

        Returns:
            The cropped mask as a read-only view into :attr:`mask`.
        """
        mask_h, mask_w = self._mask.shape
        x_end = min(dx + width, mask_w)
        y_end = min(dy + height, mask_h)
        view: NDArray[np.uint8] = self._mask[dy:y_end, dx:x_end]
        return _as_readonly(view)  # type: ignore[return-value]

    @property
    def regions(self) -> NDArray[np.int32] | Sequence[NDArray[np.int32]]:
        """The sub-regions of this ROI (defaults to :attr:`polygon`)."""
        return self._regions  # type: ignore[return-value]

    @property
    def hierarchy(self) -> NDArray[np.int32] | None:
        """The hierarchy of :attr:`regions`, or ``None`` when absent."""
        return self._hierarchy
