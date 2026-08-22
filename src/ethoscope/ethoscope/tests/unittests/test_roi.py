"""
Unit tests for core/roi.py.

Tests ROI class including initialization, properties, image cropping,
feature extraction, and boundary handling.
"""

import unittest
import warnings
import weakref

import numpy as np

from ethoscope.core.roi import ROI
from ethoscope.utils.debug import EthoscopeException

_MARKER_GRAY = 128


class TestROIInit(unittest.TestCase):
    """Test ROI initialization with various polygon formats."""

    def test_init_with_tuple_polygon(self):
        """Test ROI creation with tuple polygon."""
        roi = ROI(polygon=((0, 0), (100, 0), (100, 50), (0, 50)), idx=1)
        assert roi.idx == 1
        assert roi._polygon.shape[1] == 1  # reshaped to (N, 1, 2)
        assert roi._polygon.shape[2] == 2

    def test_init_with_numpy_polygon(self):
        """Test ROI creation with numpy array polygon."""
        polygon = np.array([[10, 10], [200, 10], [200, 100], [10, 100]])
        roi = ROI(polygon=polygon, idx=5)
        assert roi.idx == 5

    def test_init_with_3d_polygon(self):
        """Test ROI creation with already-reshaped 3D polygon."""
        polygon = np.array([[[0, 0]], [[100, 0]], [[100, 80]], [[0, 80]]])
        roi = ROI(polygon=polygon, idx=2)
        assert roi._polygon.shape == (4, 1, 2)

    def test_value_defaults_to_idx(self):
        """Test that value defaults to idx when not provided."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=7)
        assert roi.value == 7

    def test_value_set_explicitly(self):
        """Test explicit value assignment."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=3, value=42)
        assert roi.value == 42


class TestROIProperties(unittest.TestCase):
    """Test ROI property accessors."""

    def setUp(self):
        """Create a standard ROI for property tests."""
        self.roi = ROI(
            polygon=((10, 20), (110, 20), (110, 70), (10, 70)), idx=1, value=5
        )

    def test_idx(self):
        assert self.roi.idx == 1

    def test_mask_shape_and_dtype(self):
        """Test mask is correct shape and dtype."""
        _, _, w, h = self.roi.rectangle
        assert self.roi.mask.shape == (h, w)
        assert self.roi.mask.dtype == np.uint8

    def test_mask_has_nonzero_values(self):
        """Test mask contains filled region (255 values)."""
        assert np.sum(self.roi.mask > 0) > 0

    def test_offset(self):
        """Test offset returns top-left corner."""
        x, y = self.roi.offset
        assert x == 10
        assert y == 20

    def test_polygon_shape(self):
        """Test polygon is 3D array."""
        assert len(self.roi.polygon.shape) == 3

    def test_longest_axis(self):
        """Test longest_axis returns max of w, h."""
        _, _, w, h = self.roi.rectangle
        assert self.roi.longest_axis == float(max(w, h))

    def test_rectangle(self):
        """Test rectangle returns (x, y, w, h)."""
        x, y, w, h = self.roi.rectangle
        assert x == 10
        assert y == 20
        # OpenCV boundingRect includes endpoint pixels, so w/h may be +1
        assert w >= 100
        assert h >= 50

    def test_value(self):
        assert self.roi.value == 5

    def test_regions(self):
        """Test regions property is accessible."""
        assert self.roi.regions is not None

    def test_hierarchy_is_none_without_subregions(self):
        """Test hierarchy is None when no sub-regions were provided."""
        assert self.roi.hierarchy is None

    def test_bounding_rect_delegates_to_rectangle(self):
        """Test deprecated bounding_rect returns the same rectangle."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert self.roi.bounding_rect() == self.roi.rectangle

    def test_bounding_rect_warns(self):
        """Test bounding_rect emits a FutureWarning."""
        with self.assertWarns(FutureWarning):
            self.roi.bounding_rect()


class TestROIRegionsAndHierarchy(unittest.TestCase):
    """Test sub-region storage."""

    def test_regions_and_hierarchy_stored_as_given(self):
        """Test provided contours/hierarchy are stored verbatim (no recompute)."""
        regions = np.zeros((1, 1), dtype=np.int32)  # sentinel object stand-in
        hierarchy = np.zeros((1, 4), dtype=np.int32)
        roi = ROI(
            polygon=((0, 0), (50, 0), (50, 50), (0, 50)),
            idx=1,
            regions=regions,
            hierarchy=hierarchy,
        )
        assert roi.hierarchy is hierarchy
        assert roi.regions is regions


class TestROISetValue(unittest.TestCase):
    """Test ROI value modification."""

    def test_set_value(self):
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        roi.set_value(99)
        assert roi.value == 99


class TestROIGetFeatureDict(unittest.TestCase):
    """Test ROI feature dictionary generation."""

    def test_feature_dict_keys(self):
        roi = ROI(polygon=((5, 10), (105, 10), (105, 60), (5, 60)), idx=3, value=7)
        fd = roi.get_feature_dict()
        assert set(fd.keys()) == {"x", "y", "w", "h", "value", "idx"}

    def test_feature_dict_values(self):
        roi = ROI(polygon=((5, 10), (105, 10), (105, 60), (5, 60)), idx=3, value=7)
        fd = roi.get_feature_dict()
        assert fd["x"] == 5
        assert fd["y"] == 10
        # OpenCV boundingRect may include endpoint pixels (+1)
        assert fd["w"] >= 100
        assert fd["h"] >= 50
        assert fd["idx"] == 3
        assert fd["value"] == 7


class TestROIApply(unittest.TestCase):
    """Test ROI image cropping."""

    def setUp(self):
        """Create test image and ROI."""
        self.img = np.zeros((200, 300, 3), dtype=np.uint8)
        self.img[50:100, 50:150] = _MARKER_GRAY  # Gray region
        self.roi = ROI(polygon=((50, 50), (150, 50), (150, 100), (50, 100)), idx=1)

    def test_apply_returns_cropped_image(self):
        """Test apply returns correctly sized crop."""
        out, _ = self.roi.apply(self.img)
        _, _, w, h = self.roi.rectangle
        assert out.shape[0] == h
        assert out.shape[1] == w

    def test_apply_returns_mask(self):
        """Test apply returns mask matching crop dimensions."""
        out, mask = self.roi.apply(self.img)
        assert mask.shape == out.shape[:2]

    def test_apply_single_channel_image(self):
        """Test apply works with grayscale images."""
        gray_img = np.zeros((200, 300), dtype=np.uint8)
        out, mask = self.roi.apply(gray_img)
        assert len(out.shape) == 2
        assert mask.shape == out.shape

    def test_apply_roi_at_origin(self):
        """Test apply with ROI at image origin."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        out, _ = roi.apply(img)
        # OpenCV boundingRect may add +1 for endpoint pixels
        _, _, w, h = roi.rectangle
        assert out.shape[:2] == (h, w)

    def test_apply_roi_at_edge(self):
        """Test apply with ROI at image edge."""
        roi = ROI(polygon=((250, 150), (300, 150), (300, 200), (250, 200)), idx=1)
        img = np.ones((200, 300, 3), dtype=np.uint8)
        out, _ = roi.apply(img)
        assert out.shape[0] > 0
        assert out.shape[1] > 0

    def test_apply_preserves_pixel_values(self):
        """Test cropped image contains correct pixel values."""
        out, _ = self.roi.apply(self.img)
        # The center of the crop should have the marker value
        assert np.any(out == _MARKER_GRAY)


class TestROIApplyNegativeOrigin(unittest.TestCase):
    """Regression tests for ROIs reaching past the top-left frame edge.

    The mask must be cropped by the exact amount clipped off-frame so that
    image and mask pixels stay aligned (regression test for the mask-offset
    sign bug).
    """

    def setUp(self):
        """Create a triangular ROI whose origin lies off-frame."""
        # Right triangle with legs along the axes; hypotenuse is the line
        # x + y = 20, interior is x + y < 20 (contains the off-frame vertex).
        self.img = np.zeros((20, 20), dtype=np.uint8)
        self.roi = ROI(polygon=((-10, -10), (30, -10), (-10, 30)), idx=1)

    def test_apply_returns_full_frame_window(self):
        """Test clamped output covers the whole 20x20 frame."""
        out, mask = self.roi.apply(self.img)
        assert out.shape == (20, 20)
        assert mask.shape == out.shape

    def test_mask_stays_aligned_with_image_pixels(self):
        """Test interior/exterior mask values map to the correct image pixels."""
        _, mask = self.roi.apply(self.img)
        # Image pixel (x=9, y=9) satisfies x+y < 20 -> inside the ROI.
        assert int(mask[9, 9]) == 255
        # Image pixel (x=11, y=11) lies just past the hypotenuse. With the
        # historical offset-sign bug this mask cell was sourced from frame
        # (1, 1) instead of (11, 11) and wrongly read as inside.
        assert int(mask[11, 11]) == 0


class TestROIApplyEdgeCases(unittest.TestCase):
    """Edge cases for :meth:`ROI.apply`."""

    def test_fully_off_frame_raises(self):
        """Test ROI entirely outside the image raises."""
        roi = ROI(
            polygon=((1000, 1000), (1010, 1000), (1010, 1010), (1000, 1010)),
            idx=1,
        )
        img = np.zeros((20, 20), dtype=np.uint8)
        with self.assertRaises(EthoscopeException):
            roi.apply(img)

    def test_three_channel_image(self):
        """Test apply works with 3-channel images."""
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        roi = ROI(polygon=((-10, -10), (30, -10), (-10, 30)), idx=1)
        out, mask = roi.apply(img)
        assert out.shape == (20, 20, 3)
        assert mask.shape == (20, 20)

    def test_right_bottom_clip(self):
        """Test ROI clipped on the right/bottom edge returns correct window."""
        # ROI extends beyond the 200x300 frame on the right/bottom.
        roi = ROI(polygon=((250, 150), (310, 150), (310, 210), (250, 210)), idx=1)
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        out, mask = roi.apply(img)
        # Clamped to image bounds: x1=250, x2=300 (50 px), y1=150, y2=200 (50 px)
        assert out.shape[:2] == (50, 50)
        assert mask.shape == (50, 50)


class TestROIInitValidation(unittest.TestCase):
    """Validation of polygon inputs."""

    def test_empty_polygon_raises(self):
        """Test empty polygon raises ValueError."""
        with self.assertRaises(ValueError):
            ROI(polygon=[], idx=1)

    def test_degenerate_one_point_raises(self):
        """Test single-point polygon raises."""
        with self.assertRaises(ValueError):
            ROI(polygon=[(0, 0)], idx=1)

    def test_degenerate_two_points_raises(self):
        """Test two-point polygon raises."""
        with self.assertRaises(ValueError):
            ROI(polygon=[(0, 0), (10, 0)], idx=1)

    def test_invalid_shape_raises(self):
        """Test wrong coordinate count raises."""
        with self.assertRaises(ValueError):
            ROI(polygon=[(0, 0, 0), (10, 0, 0), (10, 10, 0)], idx=1)

    def test_float_polygon_is_coerced(self):
        """Test float polygon is rounded and accepted."""
        roi = ROI(polygon=[(0.2, 0.7), (10.3, 0.1), (10, 10)], idx=1)
        assert roi.polygon.dtype == np.int32
        # (0.2, 0.7) -> (0, 1), (10.3, 0.1) -> (10, 0) via rint
        assert roi.polygon.shape == (3, 1, 2)
        assert int(roi.polygon[0, 0, 0]) == 0
        assert int(roi.polygon[0, 0, 1]) == 1

    def test_ndim_one_raises(self):
        """Test 1-D array raises."""
        with self.assertRaises(ValueError):
            ROI(polygon=np.array([0, 0]), idx=1)

    def test_regions_hierarchy_mismatch_raises(self):
        """Test providing only one of regions/hierarchy raises."""
        with self.assertRaises(ValueError):
            ROI(
                polygon=((0, 0), (50, 0), (50, 50), (0, 50)),
                idx=1,
                regions=np.zeros((1, 1), dtype=np.int32),
                hierarchy=None,
            )
        with self.assertRaises(ValueError):
            ROI(
                polygon=((0, 0), (50, 0), (50, 50), (0, 50)),
                idx=1,
                regions=None,
                hierarchy=np.zeros((1, 4), dtype=np.int32),
            )


class TestROIMutability(unittest.TestCase):
    """Masks/polygons are read-only views — mutation must raise."""

    def test_mask_is_readonly(self):
        """Test mask view cannot be mutated."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        with self.assertRaises(ValueError):
            roi.mask[0, 0] = 99
        with self.assertRaises(ValueError):
            roi.mask.fill(0)

    def test_polygon_is_readonly(self):
        """Test polygon view cannot be mutated."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        with self.assertRaises(ValueError):
            roi.polygon[0, 0, 0] = 9999

    def test_cropped_mask_is_readonly(self):
        """Test mask returned by apply is also read-only."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        img = np.zeros((100, 100), dtype=np.uint8)
        _, mask = roi.apply(img)
        with self.assertRaises(ValueError):
            mask[0, 0] = 1

    def test_weakref_supported(self):
        """Test ROI can be weak-referenced (slots includes __weakref__)."""
        roi = ROI(polygon=((0, 0), (50, 0), (50, 50), (0, 50)), idx=1)
        ref = weakref.ref(roi)
        assert ref() is roi


if __name__ == "__main__":
    unittest.main()
