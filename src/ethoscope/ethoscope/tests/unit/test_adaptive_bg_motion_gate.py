"""
Unit tests for the time-normalised motion gate in AdaptiveBGModel.

The gate rejects candidate detections that jump further than
``max_speed`` (ROI main-axis lengths per second) times the time elapsed
since the last accepted detection, protecting the tracked identity
against debris, shadows or a second animal entering the ROI.
"""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

try:
    from ethoscope.trackers.adaptive_bg_tracker import AdaptiveBGModel
    from ethoscope.trackers.trackers import NoPositionError
except ImportError:
    # Handle import for different test runner contexts
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
    from ethoscope.trackers.adaptive_bg_tracker import AdaptiveBGModel
    from ethoscope.trackers.trackers import NoPositionError


_ROI_SIZE = 200
_NEAR_CENTER = (50, 100)
_FAR_CENTER = (150, 100)
_MIN_CONTOUR_AREA = 6  # same minimum-area filter as AdaptiveBGModel._track
_N_BLOBS = 2
_CENTER_TOLERANCE_PX = 5
_APPEARANCE_SPLIT_X = 100  # x-coordinate separating the two test blobs
_DEFAULT_MAX_SPEED = 1.0
_DEFAULT_GATE_DT_MS = 5000.0
_CUSTOM_MAX_SPEED = 2.5
_CUSTOM_GATE_DT_MS = 1000.0


def _blobs_image(centers, radius=8):
    """Binary image with one filled blob per center."""
    img = np.zeros((_ROI_SIZE, _ROI_SIZE), dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), radius, 255, -1)
    return img


def _find_hulls(img):
    """Contours the tracker would see (same filtering as _track)."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        cv2.approxPolyDP(c, 1.2, True)
        for c in contours
        if cv2.contourArea(c) >= _MIN_CONTOUR_AREA
    ]


def _make_tracker(**data):
    return AdaptiveBGModel(Mock(), data=data or None)


def _hull_center_x(hull):
    return cv2.minAreaRect(hull)[0][0]


class TestMotionGateConfig:
    def test_defaults(self):
        tracker = _make_tracker()
        assert tracker._max_speed == _DEFAULT_MAX_SPEED
        assert tracker._motion_gate_max_dt == _DEFAULT_GATE_DT_MS
        assert tracker._old_pos_t is None

    def test_custom_values(self):
        tracker = _make_tracker(
            max_speed=_CUSTOM_MAX_SPEED, motion_gate_max_dt=_CUSTOM_GATE_DT_MS
        )
        assert tracker._max_speed == _CUSTOM_MAX_SPEED
        assert tracker._motion_gate_max_dt == _CUSTOM_GATE_DT_MS


class TestMotionGateAllows:
    def setup_method(self):
        self.tracker = _make_tracker()
        self.grey = np.zeros((_ROI_SIZE, _ROI_SIZE), dtype=np.uint8)

    def _anchor(self, center, t):
        self.tracker._old_pos = (center[0] + 1j * center[1]) / _ROI_SIZE
        self.tracker._old_pos_t = t

    def test_nearby_candidate_passes(self):
        self._anchor(_NEAR_CENTER, 900)
        # 10 px in 100 ms = 0.5 ROI/s < max_speed of 1.0 ROI/s
        assert self.tracker._motion_gate_allows(60, 100, self.grey, 1000) is True

    def test_distant_candidate_fails(self):
        self._anchor(_NEAR_CENTER, 900)
        # 100 px in 100 ms = 5 ROI/s > max_speed of 1.0 ROI/s
        assert self.tracker._motion_gate_allows(150, 100, self.grey, 1000) is False

    def test_inactive_before_first_detection(self):
        # _old_pos_t is None: no history to gate against
        assert self.tracker._motion_gate_allows(150, 100, self.grey, 1000) is True

    def test_inactive_on_non_positive_dt(self):
        self._anchor(_NEAR_CENTER, 1000)
        assert self.tracker._motion_gate_allows(150, 100, self.grey, 1000) is True
        assert self.tracker._motion_gate_allows(150, 100, self.grey, 500) is True

    def test_lifted_after_long_gap(self):
        # dt (10 s) exceeds motion_gate_max_dt (5 s): the last position is
        # too stale to constrain the detection, so the gate is lifted.
        self._anchor(_NEAR_CENTER, 1000)
        assert self.tracker._motion_gate_allows(150, 100, self.grey, 11000) is True

    def test_max_speed_zero_disables(self):
        tracker = _make_tracker(max_speed=0)
        tracker._old_pos = (_NEAR_CENTER[0] + 1j * _NEAR_CENTER[1]) / _ROI_SIZE
        tracker._old_pos_t = 900
        assert tracker._motion_gate_allows(150, 100, self.grey, 1000) is True


class TestProcessContoursGate:
    def _anchor(self, tracker, center, t):
        tracker._old_pos = (center[0] + 1j * center[1]) / _ROI_SIZE
        tracker._old_pos_t = t

    def test_single_near_contour_passes(self):
        img = _blobs_image([_NEAR_CENTER])
        contours = _find_hulls(img)
        assert len(contours) == 1

        tracker = _make_tracker()
        self._anchor(tracker, _NEAR_CENTER, 900)
        hull, _, is_ambiguous = tracker._process_contours(img, img, contours, 1000)

        assert is_ambiguous is False
        assert abs(_hull_center_x(hull) - _NEAR_CENTER[0]) < _CENTER_TOLERANCE_PX

    def test_single_far_contour_rejected(self):
        img = _blobs_image([_FAR_CENTER])
        contours = _find_hulls(img)
        assert len(contours) == 1

        tracker = _make_tracker()
        self._anchor(tracker, _NEAR_CENTER, 900)
        with pytest.raises(NoPositionError):
            tracker._process_contours(img, img, contours, 1000)

    def test_single_far_contour_rejection_does_not_erode_background(self):
        # A false gate violation (fast-moving animal) must not accelerate
        # background adaptation over the animal.
        img = _blobs_image([_FAR_CENTER])
        contours = _find_hulls(img)

        tracker = _make_tracker()
        self._anchor(tracker, _NEAR_CENTER, 900)
        with pytest.raises(NoPositionError):
            tracker._process_contours(img, img, contours, 1000)
        assert (
            tracker._bg_model._current_half_life
            == tracker._bg_model._min_half_life
        )

    def _appearance_prefers_far(self, tracker):
        """Force a ready fg model whose likelihood ranks the far blob best."""
        tracker.fg_model.compute_features = lambda img, h: np.array(
            [_hull_center_x(h)], dtype=np.float32
        )
        tracker.fg_model.distance = (
            lambda features, t: 0.0 if features[0] > _APPEARANCE_SPLIT_X else 10.0
        )

    def test_ambiguous_prefers_near_candidate_despite_appearance(self):
        img = _blobs_image([_NEAR_CENTER, _FAR_CENTER])
        contours = _find_hulls(img)
        assert len(contours) == _N_BLOBS

        # fg_ready_threshold=0 makes the fg model ready immediately.
        tracker = _make_tracker(fg_ready_threshold=0)
        self._appearance_prefers_far(tracker)
        self._anchor(tracker, _NEAR_CENTER, 900)

        hull, _, is_ambiguous = tracker._process_contours(img, img, contours, 1000)

        # The far blob is gated out, so the near one wins on identity
        # even though appearance alone would rank it worse.
        assert abs(_hull_center_x(hull) - _NEAR_CENTER[0]) < _CENTER_TOLERANCE_PX
        assert is_ambiguous is False

    def test_ambiguous_none_in_gate_rejected(self):
        img = _blobs_image([_NEAR_CENTER, _FAR_CENTER])
        contours = _find_hulls(img)
        assert len(contours) == _N_BLOBS

        tracker = _make_tracker(fg_ready_threshold=0)
        self._appearance_prefers_far(tracker)
        # Anchor far from both blobs with a fresh timestamp.
        self._anchor(tracker, (10, 10), 900)

        with pytest.raises(NoPositionError):
            tracker._process_contours(img, img, contours, 1000)

    def test_ambiguous_first_detection_not_gated(self):
        # Without history there is nothing to gate against: appearance decides.
        img = _blobs_image([_NEAR_CENTER, _FAR_CENTER])
        contours = _find_hulls(img)

        tracker = _make_tracker(fg_ready_threshold=0)
        self._appearance_prefers_far(tracker)

        hull, _, is_ambiguous = tracker._process_contours(img, img, contours, 1000)

        assert abs(_hull_center_x(hull) - _FAR_CENTER[0]) < _CENTER_TOLERANCE_PX
        assert is_ambiguous is True

    def test_ambiguous_gate_lifted_after_long_gap(self):
        img = _blobs_image([_NEAR_CENTER, _FAR_CENTER])
        contours = _find_hulls(img)

        tracker = _make_tracker(fg_ready_threshold=0)
        self._appearance_prefers_far(tracker)
        # Stale anchor: dt exceeds motion_gate_max_dt, gate is lifted.
        self._anchor(tracker, (10, 10), 0)

        hull, _, is_ambiguous = tracker._process_contours(img, img, contours, 30000)

        assert abs(_hull_center_x(hull) - _FAR_CENTER[0]) < _CENTER_TOLERANCE_PX
        assert is_ambiguous is True


if __name__ == "__main__":
    pytest.main([__file__])
