"""
Unit tests for drawers/drawers.py.

Tests BaseDrawer, NullDrawer, and DefaultDrawer frame annotation.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from ethoscope.core.roi import ROI
from ethoscope.drawers.drawers import BaseDrawer, DefaultDrawer, NullDrawer


@pytest.fixture
def indicator_drawer() -> DefaultDrawer:
    return DefaultDrawer()


@pytest.fixture
def indicator_img() -> np.ndarray:
    return np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.fixture
def indicator_roi() -> ROI:
    return ROI(polygon=((10, 10), (150, 10), (150, 80), (10, 80)), idx=1, value=1)


class TestNullDrawer:
    def test_init(self):
        drawer = NullDrawer()
        assert not drawer._draw_frames
        assert drawer._video_out is None
        assert drawer._video_writer is None

    def test_annotate_frame_is_noop(self):
        drawer = NullDrawer()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        drawer._annotate_frame(img, {}, [])

    def test_draw_grayscale_frame(self):
        """Regression test: draw() must accept the full argument set."""
        drawer = NullDrawer()
        img = np.zeros((50, 60), dtype=np.uint8)
        drawer.draw(img, {}, [])
        frame = drawer.last_drawn_frame
        assert frame is not None
        assert frame.shape == (50, 60, 3)


class TestDefaultDrawerStimulatorIndicator:
    def test_inactive_state(self, indicator_drawer, indicator_img, indicator_roi):
        """Test inactive state draws empty circle."""
        indicator_drawer._draw_stimulator_indicator(
            indicator_img, indicator_roi, "inactive"
        )

    def test_scheduled_state(self, indicator_drawer, indicator_img, indicator_roi):
        """Test scheduled state draws white filled circle."""
        indicator_drawer._draw_stimulator_indicator(
            indicator_img, indicator_roi, "scheduled"
        )
        assert indicator_img.sum() > 0

    def test_stimulating_state(self, indicator_drawer, indicator_img, indicator_roi):
        """Test stimulating state draws blue filled circle."""
        indicator_drawer._draw_stimulator_indicator(
            indicator_img, indicator_roi, "stimulating"
        )
        assert indicator_img.sum() > 0

    def test_unknown_state(self, indicator_drawer, indicator_img, indicator_roi):
        """Test unknown state draws red warning circle."""
        indicator_drawer._draw_stimulator_indicator(
            indicator_img, indicator_roi, "weird_state"
        )
        assert indicator_img.sum() > 0

    def test_error_state(self, indicator_drawer, indicator_img, indicator_roi):
        """Test error state draws red circle."""
        indicator_drawer._draw_stimulator_indicator(
            indicator_img, indicator_roi, "error"
        )
        assert indicator_img.sum() > 0

    def test_roi_extent_clamps_to_minimum(self, indicator_drawer):
        """Tiny ROIs are clamped to the minimum indicator extent."""
        tiny = ROI(polygon=((0, 0), (10, 0), (10, 5), (0, 5)), idx=0)
        assert indicator_drawer._roi_extent(tiny) == (50, 50)

    def test_roi_extent_uses_polygon_span(self, indicator_drawer, indicator_roi):
        """Extent derives from the polygon bounding box."""
        assert indicator_drawer._roi_extent(indicator_roi) == (140, 70)


def make_tracking_unit(roi: ROI, stimulator: object = None) -> Mock:
    mock_tu = Mock()
    mock_tu.roi = roi
    mock_tu.stimulator = stimulator
    return mock_tu


@pytest.fixture
def drawer() -> DefaultDrawer:
    return DefaultDrawer()


@pytest.fixture
def img() -> np.ndarray:
    return np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.fixture
def roi() -> ROI:
    return ROI(polygon=((10, 10), (150, 10), (150, 80), (10, 80)), idx=1)


class TestDefaultDrawerAnnotateFrame:
    def test_annotate_frame_none_img(self, drawer):
        """Test _annotate_frame returns early for None img."""
        drawer._annotate_frame(None, {}, [])

    def test_annotate_frame_empty_tracking_units(self, drawer, img):
        """Test _annotate_frame with empty tracking units."""
        drawer._annotate_frame(img, {}, [])

    def test_annotate_frame_with_tracking_unit(self, drawer, img, roi):
        """Test _annotate_frame draws ROI info."""
        positions = {}  # No positions for this ROI
        drawer._annotate_frame(img, positions, [make_tracking_unit(roi)])

    def test_annotate_frame_with_position_data(self, drawer, img, roi):
        """Test _annotate_frame draws ellipses for positions."""
        positions = {
            1: [
                {
                    "x": 50,
                    "y": 30,
                    "w": 20,
                    "h": 10,
                    "phi": 0,
                    "has_interacted": False,
                }
            ]
        }
        drawer._annotate_frame(img, positions, [make_tracking_unit(roi)])
        assert img.sum() > 0

    def test_annotate_frame_with_interacted_position(self, drawer, img, roi):
        """Test interacted position drawn in different color."""
        positions = {
            1: [
                {
                    "x": 50,
                    "y": 30,
                    "w": 20,
                    "h": 10,
                    "phi": 0,
                    "has_interacted": True,
                }
            ]
        }
        drawer._annotate_frame(img, positions, [make_tracking_unit(roi)])
        assert img.sum() > 0

    def test_annotate_frame_draws_positions_without_stimulator(self, drawer, img, roi):
        """Positions are drawn even when the ROI has no stimulator."""
        positions = {
            1: [
                {"x": 50, "y": 30, "w": 20, "h": 10, "phi": 0},
            ]
        }
        drawer._annotate_frame(img, positions, [make_tracking_unit(roi)])
        assert img.sum() > 0

    def test_annotate_frame_with_stimulator_state(self, drawer, img, roi):
        """Test annotation with stimulator that has get_stimulator_state."""
        mock_stim = Mock()
        mock_stim.get_stimulator_state.return_value = "scheduled"
        drawer._annotate_frame(img, {}, [make_tracking_unit(roi, mock_stim)])
        assert img.sum() > 0

    def test_annotate_frame_with_failing_stimulator_state(self, drawer, img, roi):
        """A raising get_stimulator_state draws an error indicator."""
        mock_stim = Mock()
        mock_stim.get_stimulator_state.side_effect = RuntimeError("boom")
        drawer._annotate_frame(img, {}, [make_tracking_unit(roi, mock_stim)])
        assert img.sum() > 0

    def test_annotate_frame_with_reference_points(self, drawer, img):
        """Test annotation draws reference point markers."""
        ref_points = [(50, 50), (100, 100)]
        drawer._annotate_frame(img, {}, [], reference_points=ref_points)
        assert img.sum() > 0

    def test_annotate_frame_with_numpy_reference_points(self, drawer, img):
        """Regression test: numpy array reference points must not raise.

        Real ROI builders pass a (N, 2) numpy array of target coordinates,
        not a Python list. A ``not arr`` guard raised ``ValueError`` on it.
        """
        ref_points = np.array([[50, 50], [100, 100], [150, 75]], dtype=np.float32)
        drawer._annotate_frame(img, {}, [], reference_points=ref_points)
        assert img.sum() > 0


class TestBaseDrawerLifecycle:
    def test_last_drawn_frame_initially_none(self):
        drawer = BaseDrawer(draw_frames=False)
        assert drawer.last_drawn_frame is None

    def test_close_is_idempotent(self):
        drawer = BaseDrawer(draw_frames=False)
        drawer.close()
        drawer.close()

    def test_context_manager_closes_resources(self):
        with DefaultDrawer() as drawer:
            assert isinstance(drawer, DefaultDrawer)
        assert drawer._video_writer is None

    def test_draw_grayscale_frame(self):
        drawer = DefaultDrawer()
        img = np.zeros((40, 50), dtype=np.uint8)
        drawer.draw(img, {}, [])
        frame = drawer.last_drawn_frame
        assert frame is not None
        assert frame.shape == (40, 50, 3)

    def test_draw_color_frame(self):
        """Regression test: BGR frames are copied, not crash cvtColor."""
        drawer = DefaultDrawer()
        img = np.full((40, 50, 3), 7, dtype=np.uint8)
        drawer.draw(img, {}, [])
        frame = drawer.last_drawn_frame
        assert frame is not None
        assert frame.shape == (40, 50, 3)
        assert frame.mean() > 0

    def test_draw_reallocates_on_dtype_change(self):
        drawer = DefaultDrawer()
        drawer.draw(np.zeros((40, 50), dtype=np.uint8), {}, [])
        frame = drawer.last_drawn_frame
        assert frame is not None
        assert frame.dtype == np.uint8
        drawer.draw(np.zeros((40, 50), dtype=np.float32), {}, [])
        frame = drawer.last_drawn_frame
        assert frame is not None
        assert frame.dtype == np.float32
