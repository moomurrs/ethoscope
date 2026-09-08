"""
Image quality diagnostics for the Ethoscope.

This module computes per-frame image quality metrics (focus, noise, contrast)
and logs them to the systemd journal at a fixed interval. It is designed to
help diagnose per-device image issues (illumination, focus calibration,
flicker) that can degrade tracking quality.

The metrics are intentionally cheap and sampled at a low rate (by default one
sample every 5 seconds) so that they can run alongside tracking on a
Raspberry Pi without affecting the frame rate.

Usage: enabled from the node UI with the "log_image_diagnostics" boolean in
the experimental information section. When enabled, one structured record per
interval is written to the journal with the syslog identifier
"ethoscope-image-diagnostics", viewable with:

    journalctl -t ethoscope-image-diagnostics -f

Metrics:
    1.  tenengrad            - mean squared Sobel gradient magnitude (focus)
    2.  laplacian_variance   - variance of the Laplacian (focus)
    3.  brenner              - mean squared horizontal 2px gradient (focus)
    4.  fft_hf_ratio         - fraction of spectral energy above a frequency
                               cutoff (focus; blur is a low-pass filter)
    5.  edge_density         - fraction of Canny edge pixels (detail/noise)
    6.  noise_floor          - std within the flattest background patch,
                               excluding ROI patches (sensor noise estimate)
    7.  mean_brightness and brightness_drift (flicker/illumination)
    8.  fly_width_median_px / fly_height_median_px (detected silhouette size)
    9.  fly_contrast_median  - |fly patch mean - surrounding ring mean|
"""

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from ethoscope.core.roi import ROI

SYSLOG_IDENTIFIER = "ethoscope-image-diagnostics"
DEFAULT_INTERVAL_MS = 5000.0

# The FFT metric is computed on a downscaled copy to bound the CPU cost.
_FFT_MAX_DIM = 512
# Frequencies (in cycles/pixel) above this cutoff count as high frequency.
# Nyquist is 0.5, so 0.25 selects the upper half of the spectrum.
_FFT_HF_CUTOFF_CYCLES_PER_PX = 0.25
# Canny thresholds, kept identical to
# ethoscope.roi_builders.target_detection_diagnostics for consistency.
_CANNY_LOW = 50
_CANNY_HIGH = 150
# Patch size used by the noise floor estimator.
_NOISE_PATCH_SIZE_PX = 16
# Radii (px) of the fly patch and the surrounding ring used for the
# fly-vs-background contrast metric. Sized for the default 1280x960
# resolution where a fly silhouette is roughly 15-20 px long.
_FLY_INNER_RADIUS_PX = 10
_FLY_OUTER_RADIUS_PX = 24


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Return a single channel view/copy of the input image."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def tenengrad(img: np.ndarray) -> float:
    """
    Mean squared Sobel gradient magnitude (Tenengrad focus measure).

    Sharp images concentrate edges and score higher than blurred ones.
    """
    grey = _to_gray(img)
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx * gx + gy * gy
    return float(np.mean(grad_sq))


def laplacian_variance(img: np.ndarray) -> float:
    """
    Variance of the Laplacian. Classic single-number focus measure.
    """
    grey = _to_gray(img)
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    return float(np.var(lap))


def brenner(img: np.ndarray) -> float:
    """
    Brenner focus measure: mean squared difference between pixels two
    columns apart. Cheap and robust for vertical edge content.
    """
    grey = _to_gray(img)
    if grey.shape[1] < 3:
        return 0.0
    data = grey.astype(np.float32)
    diff = data[:, 2:] - data[:, :-2]
    return float(np.mean(diff * diff))


def fft_hf_ratio(img: np.ndarray) -> float:
    """
    Fraction of spectral energy above the high frequency cutoff.

    Defocus acts as a low-pass filter, so this ratio drops monotonically
    with blur. Computed on a downscaled copy for speed.
    """
    grey = _to_gray(img)
    h, w = grey.shape[:2]
    scale = _FFT_MAX_DIM / float(max(h, w))
    if scale < 1.0:
        new_w = max(8, round(w * scale))
        new_h = max(8, round(h * scale))
        grey = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_AREA)

    data = grey.astype(np.float32)
    rows_in, cols_in = data.shape
    spectrum = np.fft.rfft2(data)
    power = np.abs(spectrum)
    np.multiply(power, power, out=power)

    fy = np.fft.fftfreq(rows_in)[:, None]
    fx = np.fft.rfftfreq(cols_in)[None, :]
    freq = np.hypot(fy, fx)

    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    hf = float(power[freq > _FFT_HF_CUTOFF_CYCLES_PER_PX].sum())
    return hf / total


def edge_density(img: np.ndarray) -> float:
    """
    Fraction of pixels flagged as edges by Canny(50, 150).

    Both focus and sensor noise raise this metric, so always interpret it
    together with the noise floor.
    """
    grey = _to_gray(img)
    edges = cv2.Canny(grey, _CANNY_LOW, _CANNY_HIGH)
    if edges.size == 0:
        return 0.0
    return float(np.count_nonzero(edges) / edges.size)


def _intersects_any_roi(
    x: int, y: int, size: int, roi_boxes: list[tuple[int, int, int, int]]
) -> bool:
    """Whether the patch at (x, y) with the given square size hits any box."""
    for bx, by, bw, bh in roi_boxes:
        if x < bx + bw and bx < x + size and y < by + bh and by < y + size:
            return True
    return False


def noise_floor(
    img: np.ndarray,
    roi_boxes: list[tuple[int, int, int, int]] | None = None,
    patch_size: int = _NOISE_PATCH_SIZE_PX,
) -> float:
    """
    Estimate the sensor noise floor as the std of the flattest image patch.

    Patches overlapping any ROI bounding box (x, y, w, h) are excluded so
    that tube walls, flies and arena structures do not bias the estimate.
    If every patch overlaps an ROI, the median patch std is returned.
    """
    grey = _to_gray(img)
    h, w = grey.shape[:2]
    if h < patch_size or w < patch_size:
        return float(np.std(grey.astype(np.float32)))

    best: float | None = None
    all_stds: list[float] = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = grey[y : y + patch_size, x : x + patch_size]
            std = float(np.std(patch.astype(np.float32)))
            all_stds.append(std)
            if roi_boxes and _intersects_any_roi(x, y, patch_size, roi_boxes):
                continue
            if best is None or std < best:
                best = std

    if best is None:
        return float(np.median(all_stds))
    return best


def fly_contrast(
    img: np.ndarray,
    x: float,
    y: float,
    inner_radius: int = _FLY_INNER_RADIUS_PX,
    outer_radius: int = _FLY_OUTER_RADIUS_PX,
) -> float | None:
    """
    |mean(fly patch) - mean(surrounding ring)| around an absolute position.

    Returns None when the window does not fit inside the frame.
    """
    grey = _to_gray(img).astype(np.float32)
    h, w = grey.shape[:2]
    xi = round(float(x))
    yi = round(float(y))
    if (
        xi - outer_radius < 0
        or yi - outer_radius < 0
        or xi + outer_radius >= w
        or yi + outer_radius >= h
    ):
        return None

    yy, xx = np.mgrid[
        -outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1
    ]
    dist_sq = yy.astype(np.float32) ** 2 + xx.astype(np.float32) ** 2
    inner = dist_sq <= float(inner_radius) ** 2
    ring = (dist_sq > float(inner_radius) ** 2) & (
        dist_sq <= float(outer_radius) ** 2
    )

    patch = grey[
        yi - outer_radius : yi + outer_radius + 1,
        xi - outer_radius : xi + outer_radius + 1,
    ]
    inner_mean = float(np.mean(patch[inner]))
    ring_mean = float(np.mean(patch[ring]))
    return abs(inner_mean - ring_mean)


def _get_journal_sender() -> Callable[..., Any] | None:
    """
    Return systemd journal send function, or None when python-systemd is
    not installed. In that case records fall back to stderr logging, which
    journald captures anyway for systemd services.
    """
    try:
        from systemd import journal  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return journal.send


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


class ImageDiagnosticsLogger:
    """
    Samples image quality metrics at a fixed interval and logs one
    structured record per sample to the systemd journal.

    Records carry the syslog identifier "ethoscope-image-diagnostics", a
    JSON blob in MESSAGE, and one journal field per metric (e.g. TENENGRAD)
    so that they can be filtered with `journalctl --field`.

    All failures are swallowed: diagnostics must never break tracking.
    """

    INTERVAL_MS = DEFAULT_INTERVAL_MS

    def __init__(
        self,
        machine_id: str = "",
        machine_name: str = "",
        run_id: str = "",
        interval_ms: float | None = None,
    ):
        self._interval_ms = (
            self.INTERVAL_MS if interval_ms is None else float(interval_ms)
        )
        self._machine_id = str(machine_id)
        self._machine_name = str(machine_name)
        self._run_id = str(run_id)
        self._last_sample_t_ms: float | None = None
        self._last_mean_brightness: float | None = None
        self._journal_send = _get_journal_sender()
        self._fallback_logger = logging.getLogger("ethoscope.image_diagnostics")
        self._fallback_logger.setLevel(logging.INFO)

    @property
    def journal_available(self) -> bool:
        """Whether records are sent to the native journal (python-systemd)."""
        return self._journal_send is not None

    def maybe_log(
        self,
        t_ms: float,
        frame: np.ndarray,
        tracked_rois: list[tuple["ROI", list[Any]]],
        frame_idx: int | None = None,
    ) -> bool:
        """
        Compute metrics and log one record if at least `interval_ms` of
        experiment time elapsed since the previous sample.

        :param t_ms: experiment timestamp of the frame, in milliseconds
        :param frame: the raw camera frame (grayscale or BGR)
        :param tracked_rois: list of (roi, data_rows) pairs for this frame
        :param frame_idx: optional index of the frame within the run
        :return: True if a record was logged on this call
        """
        try:
            if self._last_sample_t_ms is not None and (
                float(t_ms) - self._last_sample_t_ms
            ) < self._interval_ms:
                return False
            self._last_sample_t_ms = float(t_ms)
            record = self._build_record(t_ms, frame, tracked_rois, frame_idx)
            self._send(record)
            return True
        except Exception as e:  # noqa: BLE001 - never break tracking
            logging.warning("Image diagnostics logging failed: %s", e)
            return False

    def _build_record(
        self,
        t_ms: float,
        frame: np.ndarray,
        tracked_rois: list[tuple["ROI", list[Any]]],
        frame_idx: int | None,
    ) -> dict[str, Any]:
        grey = _to_gray(frame)

        mean_brightness = float(np.mean(grey))
        if self._last_mean_brightness is None:
            brightness_drift = None
        else:
            brightness_drift = mean_brightness - self._last_mean_brightness
        self._last_mean_brightness = mean_brightness

        roi_boxes: list[tuple[int, int, int, int]] = []
        for roi, _ in tracked_rois:
            rx, ry, rw, rh = roi.rectangle
            roi_boxes.append((int(rx), int(ry), int(rw), int(rh)))

        widths: list[int] = []
        heights: list[int] = []
        contrasts: list[float] = []
        n_detections = 0
        for roi, data_rows in tracked_rois:
            ox, oy = roi.offset
            for data_point in data_rows:
                n_detections += 1
                w = data_point.get("w")
                h = data_point.get("h")
                if w is not None:
                    widths.append(int(w))
                if h is not None:
                    heights.append(int(h))
                x = data_point.get("x")
                y = data_point.get("y")
                if x is None or y is None:
                    continue
                contrast = fly_contrast(frame, int(x) + ox, int(y) + oy)
                if contrast is not None:
                    contrasts.append(contrast)

        fly_width = _median_or_none([float(v) for v in widths])
        fly_height = _median_or_none([float(v) for v in heights])
        fly_contrast_median = _median_or_none(contrasts)

        return {
            "t_ms": round(float(t_ms)),
            "frame_idx": None if frame_idx is None else int(frame_idx),
            "machine_id": self._machine_id,
            "machine_name": self._machine_name,
            "run_id": self._run_id,
            "tenengrad": round(tenengrad(grey), 4),
            "laplacian_variance": round(laplacian_variance(grey), 4),
            "brenner": round(brenner(grey), 4),
            "fft_hf_ratio": round(fft_hf_ratio(grey), 6),
            "edge_density": round(edge_density(grey), 6),
            "noise_floor": round(noise_floor(grey, roi_boxes), 4),
            "mean_brightness": round(mean_brightness, 4),
            "brightness_drift": (
                None if brightness_drift is None else round(brightness_drift, 4)
            ),
            "n_detections": n_detections,
            "fly_width_median_px": (
                None if fly_width is None else round(fly_width, 2)
            ),
            "fly_height_median_px": (
                None if fly_height is None else round(fly_height, 2)
            ),
            "fly_contrast_median": (
                None if fly_contrast_median is None else round(fly_contrast_median, 4)
            ),
        }

    def _send(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record)
        if self._journal_send is not None:
            fields: dict[str, Any] = {
                "MESSAGE": payload,
                "PRIORITY": 6,
                "SYSLOG_IDENTIFIER": SYSLOG_IDENTIFIER,
            }
            for key, value in record.items():
                if value is None:
                    continue
                fields[key.upper()] = str(value)
            self._journal_send(**fields)
        else:
            self._fallback_logger.info(payload)
