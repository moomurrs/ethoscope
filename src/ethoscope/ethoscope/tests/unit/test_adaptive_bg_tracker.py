"""
Unit tests for adaptive background tracker ObjectModel.

Tests specifically for the boundary validation and size compatibility fixes
to prevent OpenCV size mismatch crashes.
"""

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

try:
    from ethoscope.core.roi import ROI
    from ethoscope.trackers.adaptive_bg_tracker import AdaptiveBGModel, ObjectModel
except ImportError:
    # Handle import for different test runner contexts
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
    from ethoscope.core.roi import ROI
    from ethoscope.trackers.adaptive_bg_tracker import AdaptiveBGModel, ObjectModel


class TestObjectModel:
    """Test class for ObjectModel compute_features method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ObjectModel(history_length=100)

    def test_compute_features_normal_case(self):
        """Test compute_features with normal bounding rectangle."""
        # Create a test image
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255  # White square

        # Create a contour that fits within bounds
        contour = np.array([[45, 45], [55, 45], [55, 55], [45, 55]], dtype=np.int32)

        # Should not raise an exception
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3  # area, height, mean_grey
        assert np.issubdtype(features.dtype, np.floating)  # Accept any float type

    def test_compute_features_boundary_overflow(self):
        """Test compute_features with bounding rectangle extending beyond image bounds."""
        # Create a small test image
        img = np.zeros((50, 50), dtype=np.uint8)
        img[20:30, 20:30] = 255

        # Create a contour that extends beyond image boundaries
        contour = np.array([[45, 45], [60, 45], [60, 60], [45, 60]], dtype=np.int32)

        # Should handle gracefully without crashing
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        assert not np.isnan(features).any()  # No NaN values

    def test_compute_features_zero_size_region(self):
        """Test compute_features when clipping results in zero-size region."""
        img = np.zeros((50, 50), dtype=np.uint8)

        # Create a contour completely outside image bounds
        contour = np.array([[60, 60], [70, 60], [70, 70], [60, 70]], dtype=np.int32)

        # Should return default features
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        assert np.allclose(features, [0.0, 0.0, 0.0])

    def test_compute_features_negative_coordinates(self):
        """Test compute_features with negative coordinates in bounding rectangle."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[5:15, 5:15] = 255

        # Create a contour that starts at negative coordinates
        contour = np.array([[-5, -5], [10, -5], [10, 10], [-5, 10]], dtype=np.int32)

        # Should handle gracefully by clipping to valid bounds
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        assert not np.isnan(features).any()

    def test_compute_features_shape_mismatch_handling(self):
        """Test that shape mismatches are handled correctly."""
        img = np.zeros((30, 40), dtype=np.uint8)  # Rectangular image
        img[10:20, 15:25] = 255

        # Create a contour near the edge where shape mismatch might occur
        contour = np.array([[35, 25], [45, 25], [45, 35], [35, 35]], dtype=np.int32)

        # Should handle shape mismatches gracefully
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        assert not np.isnan(features).any()

    @patch("cv2.mean")
    def test_compute_features_cv2_error_handling(self, mock_mean):
        """Test that OpenCV errors are handled gracefully."""
        # Setup mock to raise cv2.error
        mock_mean.side_effect = cv2.error("Test OpenCV error")

        img = np.zeros((50, 50), dtype=np.uint8)
        contour = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32)

        # Should not crash, should use fallback value
        features = self.model.compute_features(img, contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        # The mean_col feature should be 1.0 due to fallback (mean_col=0.0 + 1)
        assert features[2] == 1.0

    def test_buffer_reallocation(self):
        """Test that image buffers are reallocated correctly when needed."""
        # Start with small image
        small_img = np.zeros((20, 20), dtype=np.uint8)
        small_contour = np.array([[5, 5], [10, 5], [10, 10], [5, 10]], dtype=np.int32)

        self.model.compute_features(small_img, small_contour)

        # Now use larger image - should trigger reallocation
        large_img = np.zeros((100, 100), dtype=np.uint8)
        large_contour = np.array(
            [[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32
        )

        features = self.model.compute_features(large_img, large_contour)

        assert isinstance(features, np.ndarray)
        assert len(features) == 3
        # Buffers should have been reallocated
        assert self.model._roi_img_buff.shape[0] >= 20
        assert self.model._roi_img_buff.shape[1] >= 20

    def test_multiple_calls_consistency(self):
        """Test that multiple calls with the same input produce consistent results."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[20:30, 20:30] = 128
        contour = np.array([[18, 18], [32, 18], [32, 32], [18, 32]], dtype=np.int32)

        features1 = self.model.compute_features(img, contour)
        features2 = self.model.compute_features(img, contour)

        np.testing.assert_array_almost_equal(features1, features2)


class TestFitAndAdjustEllipse:
    """Position must come from fg pixels near the selected hull.

    Neither the hull polygon alone (snaps front/back when a still fly's mask
    pinches into fragments) nor all ROI foreground (dragged by distant blobs).
    """

    _FLY_CENTER = (60, 100)
    _FLY_AXES = (18, 9)
    _FLY_GREY = 200
    _BLOB_CENTER = (160, 100)
    _BLOB_AXES = (25, 25)
    _BLOB_GREY = 255
    _N_BLOBS = 2
    _FG_MARK = 255
    _MIN_CONTOUR_AREA = 6
    # Minimum hull-polygon jump proving the test setup actually pinches.
    _MIN_POLYGON_JUMP_PX = 4.0

    def setup_method(self):
        self.tracker = AdaptiveBGModel(Mock())

    def _make_frame_with_distant_blob(self):
        """Fly-like blob on the left, larger brighter static blob on the right."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(
            img, self._FLY_CENTER, self._FLY_AXES, 0, 0, 360, self._FLY_GREY, -1
        )
        cv2.ellipse(
            img, self._BLOB_CENTER, self._BLOB_AXES, 0, 0, 360, self._BLOB_GREY, -1
        )
        return img

    def _fly_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [
            cv2.approxPolyDP(c, 1.2, True)
            for c in contours
            if cv2.contourArea(c) >= self._MIN_CONTOUR_AREA
        ]
        assert len(contours) == self._N_BLOBS
        return min(contours, key=lambda c: cv2.moments(c)["m10"])

    def test_position_tracks_local_fg_not_whole_roi_foreground(self):
        """The centroid must follow fg pixels near the hull, ignoring other fg."""
        img = self._make_frame_with_distant_blob()
        hull = self._fly_contour(img)

        self.tracker._buff_fg = img.copy()
        (x, y), (w, h), _ = self.tracker._fit_and_adjust_ellipse(hull, img)

        # Near the fly (within its semi-axes), far from the distant blob:
        # whole-ROI moments would sit between the two (~x=130).
        assert abs(x - self._FLY_CENTER[0]) < self._FLY_AXES[0]
        assert abs(y - self._FLY_CENTER[1]) < self._FLY_AXES[1]
        assert abs(x - self._BLOB_CENTER[0]) > self._BLOB_AXES[0]

        (_, _), (mw, mh), _ = cv2.minAreaRect(hull)
        assert (w, h) == (max(mw, mh), min(mw, mh))

    def test_position_stable_when_mask_pinches_into_fragments(self):
        """Regression: a still fly's mask pinching into front/back fragments
        must not teleport the centroid to the selected fragment."""
        roi = ROI(polygon=((0, 0), (200, 0), (200, 200), (0, 200)), idx=0)
        # Expected size ~ fly body proportion of the ROI, as in real setups.
        tracker = AdaptiveBGModel(roi, data={"object_expected_size": 0.11})

        def run(cut):
            fg = np.zeros((200, 200), dtype=np.uint8)
            cv2.ellipse(fg, (80, 100), (6, 2), 0, 0, 360, (45,), -1)
            cv2.ellipse(fg, (91, 100), (5, 2), 0, 0, 360, (65,), -1)
            cv2.line(fg, (85, 100), (87, 100), (30,), 1)
            if cut:
                fg[:, 86] = 0  # 1px pinch -> mask splits into two fragments
            mask = fg.copy()
            mask[mask > 0] = 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [
                cv2.approxPolyDP(c, 1.2, True)
                for c in contours
                if cv2.contourArea(c) >= self._MIN_CONTOUR_AREA
            ]
            assert len(contours) == (2 if cut else 1)
            hull = max(contours, key=cv2.contourArea)
            tracker._buff_fg = fg.copy()
            (x, y), _, _ = tracker._fit_and_adjust_ellipse(hull, fg)
            m = cv2.moments(hull)
            return (x, y), (m["m10"] / m["m00"], m["m01"] / m["m00"])

        (x_conn, _), (px_conn, _) = run(cut=False)
        (x_split, _), (px_split, _) = run(cut=True)

        # The old hull-polygon centroid jumps half a body length...
        assert abs(px_split - px_conn) > self._MIN_POLYGON_JUMP_PX
        # ...while the reported position must stay close to the merged one.
        assert abs(x_split - x_conn) < 0.5 * abs(px_split - px_conn)

    def test_centroid_falls_back_to_hull_without_foreground(self):
        """With no usable fg pixels, the hull polygon centroid is returned."""
        img = self._make_frame_with_distant_blob()
        hull = self._fly_contour(img)

        self.tracker._buff_fg = np.zeros_like(img)
        (x, y), _, _ = self.tracker._fit_and_adjust_ellipse(hull, img)

        m = cv2.moments(hull)
        np.testing.assert_allclose(x, m["m10"] / m["m00"], atol=1e-6)
        np.testing.assert_allclose(y, m["m01"] / m["m00"], atol=1e-6)

    def test_protection_ellipse_still_drawn_on_fg_buffer(self):
        """The inflated ellipse must remain on _buff_fg for bg protection."""
        img = self._make_frame_with_distant_blob()
        hull = self._fly_contour(img)

        fg = img.copy()
        self.tracker._buff_fg = fg
        self.tracker._fit_and_adjust_ellipse(hull, img)

        cy, cx = self._FLY_CENTER[1], self._FLY_CENTER[0]
        assert fg[cy, cx] == self._FG_MARK


if __name__ == "__main__":
    pytest.main([__file__])
