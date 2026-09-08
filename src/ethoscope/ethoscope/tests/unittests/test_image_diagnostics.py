"""
Unit tests for image quality diagnostics.

Covers the focus/noise/contrast metric functions and the interval-gated
journal logging performed by ImageDiagnosticsLogger.
"""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ethoscope.core.data_point import DataPoint
from ethoscope.core.roi import ROI
from ethoscope.core.variables import (
    HeightVariable,
    WidthVariable,
    XPosVariable,
    YPosVariable,
)
from ethoscope.utils.image_diagnostics import (
    ImageDiagnosticsLogger,
    brenner,
    edge_density,
    fft_hf_ratio,
    fly_contrast,
    laplacian_variance,
    noise_floor,
    tenengrad,
)


def _make_sharp_and_blurred():
    """A noisy high-frequency image and a blurred version of it."""
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    return sharp, blurred


class TestFocusMetrics(unittest.TestCase):
    """Focus metrics must score sharp images above blurred ones."""

    def setUp(self):
        self.sharp, self.blurred = _make_sharp_and_blurred()

    def test_tenengrad(self):
        self.assertGreater(tenengrad(self.sharp), tenengrad(self.blurred))

    def test_laplacian_variance(self):
        self.assertGreater(
            laplacian_variance(self.sharp), laplacian_variance(self.blurred)
        )

    def test_brenner(self):
        self.assertGreater(brenner(self.sharp), brenner(self.blurred))

    def test_fft_hf_ratio(self):
        self.assertGreater(fft_hf_ratio(self.sharp), fft_hf_ratio(self.blurred))

    def test_edge_density(self):
        self.assertGreater(edge_density(self.sharp), edge_density(self.blurred))

    def test_metrics_accept_bgr_input(self):
        bgr = cv2.cvtColor(self.sharp, cv2.COLOR_GRAY2BGR)
        self.assertAlmostEqual(
            tenengrad(bgr), tenengrad(self.sharp), places=4
        )

    def test_metrics_on_tiny_images(self):
        tiny = np.full((2, 2), 128, dtype=np.uint8)
        self.assertEqual(tenengrad(tiny), 0.0)
        self.assertEqual(laplacian_variance(tiny), 0.0)
        self.assertEqual(brenner(tiny), 0.0)


class TestNoiseFloor(unittest.TestCase):
    """The noise floor must estimate background noise, not structure."""

    def test_flat_vs_noisy(self):
        clean = np.full((240, 320), 128, dtype=np.uint8)
        rng = np.random.default_rng(7)
        noisy = rng.normal(128, 10, (240, 320)).clip(0, 255).astype(np.uint8)

        clean_floor = noise_floor(clean)
        noisy_floor = noise_floor(noisy)

        self.assertLess(clean_floor, 1.0)
        self.assertGreater(noisy_floor, 3.0)
        self.assertLess(noisy_floor, 15.0)

    def test_flat_roi_is_excluded(self):
        """A flat bright ROI (e.g. a backlit tube) must not set the floor."""
        rng = np.random.default_rng(7)
        img = rng.normal(100, 10, (240, 320)).clip(0, 255).astype(np.uint8)
        img[0:120, 0:120] = 200  # flat "tube" region

        with_exclusion = noise_floor(img, roi_boxes=[(0, 0, 120, 120)])
        without_exclusion = noise_floor(img)

        self.assertGreater(with_exclusion, 3.0)
        self.assertLess(with_exclusion, 15.0)
        self.assertLess(without_exclusion, 1.0)

    def test_fallback_when_all_patches_covered(self):
        rng = np.random.default_rng(7)
        noisy = rng.normal(128, 10, (240, 320)).clip(0, 255).astype(np.uint8)
        floor = noise_floor(noisy, roi_boxes=[(0, 0, 320, 240)])
        self.assertGreater(floor, 3.0)
        self.assertLess(floor, 15.0)


class TestFlyContrast(unittest.TestCase):
    """Contrast is |fly patch mean - surrounding ring mean|."""

    def setUp(self):
        self.img = np.full((240, 320), 200, dtype=np.uint8)
        cv2.circle(self.img, (160, 120), 9, (40, 40, 40), -1)  # dark fly on bright bg

    def test_dark_object_on_bright_background(self):
        contrast = fly_contrast(self.img, 160, 120)
        self.assertIsNotNone(contrast)
        assert contrast is not None
        self.assertGreater(contrast, 100.0)

    def test_uniform_image_has_no_contrast(self):
        flat = np.full((240, 320), 128, dtype=np.uint8)
        contrast = fly_contrast(flat, 160, 120)
        self.assertIsNotNone(contrast)
        assert contrast is not None
        self.assertAlmostEqual(contrast, 0.0, places=4)

    def test_returns_none_when_window_out_of_bounds(self):
        self.assertIsNone(fly_contrast(self.img, 5, 5))
        self.assertIsNone(fly_contrast(self.img, 315, 235))


def _make_roi():
    polygon = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.int32)
    return ROI(polygon, idx=1)


def _make_data_point():
    return DataPoint(
        [
            XPosVariable(50),
            YPosVariable(25),
            WidthVariable(14),
            HeightVariable(7),
        ]
    )


class TestImageDiagnosticsLogger(unittest.TestCase):
    """Interval gating, journal record structure and metric collection."""

    def setUp(self):
        self.frame = np.full((240, 320), 200, dtype=np.uint8)
        cv2.circle(self.frame, (60, 35), 9, (40, 40, 40), -1)  # fly inside the ROI
        self.roi = _make_roi()
        self.data_point = _make_data_point()

    def _make_logger(self, interval_ms=None):
        sent = []
        logger = ImageDiagnosticsLogger(
            machine_id="ETHO_1",
            machine_name="test_ethoscope",
            run_id="abc123",
            interval_ms=interval_ms,
        )
        logger._journal_send = lambda **kwargs: sent.append(kwargs)
        return logger, sent

    def _tracked(self):
        return [(self.roi, [self.data_point])]

    def test_interval_gating(self):
        logger, sent = self._make_logger()

        self.assertTrue(logger.maybe_log(0, self.frame, self._tracked(), frame_idx=0))
        self.assertFalse(logger.maybe_log(1000, self.frame, self._tracked()))
        self.assertTrue(
            logger.maybe_log(5000, self.frame, self._tracked(), frame_idx=75)
        )

        self.assertEqual(len(sent), 2)

    def test_record_structure(self):
        logger, sent = self._make_logger()
        logger.maybe_log(1234, self.frame, self._tracked(), frame_idx=18)

        record = sent[0]
        self.assertEqual(record["SYSLOG_IDENTIFIER"], "ethoscope-image-diagnostics")
        self.assertEqual(record["PRIORITY"], 6)
        self.assertEqual(record["MACHINE_ID"], "ETHO_1")
        self.assertEqual(record["RUN_ID"], "abc123")
        self.assertEqual(record["FRAME_IDX"], "18")

        payload = json.loads(record["MESSAGE"])
        expected_keys = [
            "t_ms",
            "frame_idx",
            "machine_id",
            "machine_name",
            "run_id",
            "tenengrad",
            "laplacian_variance",
            "brenner",
            "fft_hf_ratio",
            "edge_density",
            "noise_floor",
            "mean_brightness",
            "brightness_drift",
            "n_detections",
            "fly_width_median_px",
            "fly_height_median_px",
            "fly_contrast_median",
        ]
        for key in expected_keys:
            self.assertIn(key, payload)

        self.assertEqual(payload["t_ms"], 1234)
        self.assertEqual(payload["frame_idx"], 18)
        self.assertEqual(payload["n_detections"], 1)
        self.assertIsNone(payload["brightness_drift"])

    def test_fly_metrics_extracted_from_data_points(self):
        logger, sent = self._make_logger()
        logger.maybe_log(0, self.frame, self._tracked(), frame_idx=0)

        payload = json.loads(sent[0]["MESSAGE"])
        self.assertEqual(payload["fly_width_median_px"], 14)
        self.assertEqual(payload["fly_height_median_px"], 7)
        # dark disk on bright background inside the frame
        self.assertIsNotNone(payload["fly_contrast_median"])
        self.assertGreater(payload["fly_contrast_median"], 100.0)

    def test_brightness_drift_on_second_sample(self):
        logger, sent = self._make_logger()
        logger.maybe_log(0, self.frame, self._tracked())
        logger.maybe_log(5000, self.frame, self._tracked())

        second = json.loads(sent[1]["MESSAGE"])
        self.assertAlmostEqual(second["brightness_drift"], 0.0, places=4)

    def test_no_detections_gives_null_fly_metrics(self):
        logger, sent = self._make_logger()
        logger.maybe_log(0, self.frame, [(self.roi, [])])

        payload = json.loads(sent[0]["MESSAGE"])
        self.assertEqual(payload["n_detections"], 0)
        self.assertIsNone(payload["fly_width_median_px"])
        self.assertIsNone(payload["fly_height_median_px"])
        self.assertIsNone(payload["fly_contrast_median"])

    def test_fallback_logging_when_journal_unavailable(self):
        logger, _ = self._make_logger()
        logger._journal_send = None  # simulate missing python-systemd

        with self.assertLogs("ethoscope.image_diagnostics", level="INFO") as captured:
            self.assertTrue(logger.maybe_log(0, self.frame, self._tracked()))

        # assertLogs prefixes entries with "LEVEL:logger_name:"
        message = captured.output[0].split(":", 2)[2]
        payload = json.loads(message)
        self.assertEqual(payload["machine_id"], "ETHO_1")

    def test_exceptions_are_swallowed(self):
        logger, sent = self._make_logger()
        # a failing metric computation must not propagate
        with patch(
            "ethoscope.utils.image_diagnostics.tenengrad"
        ) as mock_tenengrad:
            mock_tenengrad.side_effect = RuntimeError("boom")
            self.assertFalse(logger.maybe_log(0, self.frame, self._tracked()))
        self.assertEqual(len(sent), 0)

        # and the logger keeps working afterwards
        self.assertTrue(logger.maybe_log(5000, self.frame, self._tracked()))


if __name__ == "__main__":
    unittest.main()
