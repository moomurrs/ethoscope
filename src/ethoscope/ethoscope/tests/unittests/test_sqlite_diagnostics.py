"""
Unit tests for the optional 'diagnostic' table in SQLiteResultWriter (io/sqlite.py).

Tests cover:
- Table creation only when enable_diagnostics is True (fresh and append flows)
- Table absence when the toggle is disabled
- Schema (columns/types) and timestamp index
- Image quality calculations identical to TargetDetectionDiagnostics (mask=None path)
- Arena-region metrics: union ROI mask, masked statistics, caching, no-ROI fallback
- Throttled per-frame recording (DIAGNOSTICS_PERIOD_SECONDS)
- Unix (wall-clock) timestamps
- Retention limit in minutes (0 or less = keep forever)
"""

import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ethoscope.core.roi import ROI
from ethoscope.io import sqlite as sqlite_io
from ethoscope.io.sqlite import DIAGNOSTICS_PERIOD_SECONDS, SQLiteResultWriter
from ethoscope.roi_builders.target_detection_diagnostics import (
    TargetDetectionDiagnostics,
)

_DIAGNOSTIC_COLUMNS = [
    "id",
    "t",
    "mean_brightness",
    "median_brightness",
    "std_brightness",
    "min_brightness",
    "max_brightness",
    "contrast_rms",
    "contrast_range",
    "histogram_entropy",
    "edge_density",
]


class TestDiagnosticTable(unittest.TestCase):
    """Test suite for the optional 'diagnostic' table."""

    def setUp(self):
        """Create temporary database and mock objects for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)

        self.db_credentials = {"name": self.db_path}
        self.rois = [
            ROI(polygon=((0, 0), (100, 0), (100, 100), (0, 100)), idx=1, value=1),
        ]
        self.metadata = {
            "machine_name": "test_device",
            "machine_id": "TEST_001",
            "date_time": "2025_01_15_120000",
        }
        self.frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        self.writers = []

    def tearDown(self):
        """Shut down writers and clean up temporary files."""
        for writer in list(self.writers):
            self._close_writer(writer)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _create_writer(self, **kwargs):
        """Helper to create and track SQLiteResultWriter instances."""
        args = {
            "db_credentials": self.db_credentials,
            "rois": self.rois,
            "metadata": self.metadata,
            "erase_old_db": False,
        }
        args.update(kwargs)
        writer = SQLiteResultWriter(**args)
        self.writers.append(writer)
        return writer

    def _close_writer(self, writer):
        """Finalize a writer (processes the whole async queue) and untrack it."""
        try:
            writer.__exit__(None, None, None)
        finally:
            if writer in self.writers:
                self.writers.remove(writer)

    def _fetch_all(self, sql):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def _diagnostic_rows(self):
        return self._fetch_all("SELECT t FROM diagnostic ORDER BY id")

    def _table_exists(self, name):
        rows = self._fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='%s'" % name
        )
        return len(rows) > 0

    # ------------------------------------------------------------------ #
    # Toggle behaviour: table creation
    # ------------------------------------------------------------------ #

    def test_constant_default_period_is_one_second(self):
        """The throttling constant defaults to 1 second and is easy to change."""
        self.assertEqual(DIAGNOSTICS_PERIOD_SECONDS, 1.0)

    def test_diagnostic_table_created_when_enabled(self):
        """Fresh DB with toggle on contains the diagnostic table + index."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)
        self._close_writer(writer)

        self.assertTrue(self._table_exists("diagnostic"))
        indexes = self._fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_diagnostic_t'"
        )
        self.assertEqual(len(indexes), 1)

    def test_diagnostic_table_absent_when_disabled(self):
        """Toggle off (default): no diagnostic table, even after flushes."""
        writer = self._create_writer(erase_old_db=True)
        writer.flush(0, self.frame)
        writer.flush(2000, self.frame)
        self._close_writer(writer)

        self.assertFalse(self._table_exists("diagnostic"))

    def test_diagnostic_columns_and_types(self):
        """Schema matches the requested columns with Unix 't' column."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)
        self._close_writer(writer)

        rows = self._fetch_all("PRAGMA table_info(diagnostic)")
        names = [r[1] for r in rows]
        types = {r[1]: r[2] for r in rows}

        self.assertEqual(names, _DIAGNOSTIC_COLUMNS)
        self.assertEqual(types["id"], "INTEGER")
        self.assertEqual(types["t"], "INTEGER")
        for metric in _DIAGNOSTIC_COLUMNS[2:]:
            self.assertEqual(types[metric], "REAL")

    def test_diagnostic_table_ensured_on_append(self):
        """Append flow (erase_old_db=False + database_to_append) creates it too."""
        # Pre-create a minimal legacy database (without the diagnostic table)
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
        conn.execute(
            "CREATE TABLE START_EVENTS ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER, event TEXT)"
        )
        conn.commit()
        conn.close()

        writer = self._create_writer(erase_old_db=False, enable_diagnostics=True)
        writer.database_to_append = self.db_path  # force the append branch
        writer._create_all_tables()
        self._close_writer(writer)

        self.assertTrue(self._table_exists("diagnostic"))

    # ------------------------------------------------------------------ #
    # Calculation fidelity
    # ------------------------------------------------------------------ #

    def test_metrics_identical_to_target_detection_diagnostics(self):
        """_analyze_frame_quality (mask=None) matches TargetDetectionDiagnostics."""
        tmp_logs = tempfile.mkdtemp(prefix="diag_logs_")
        try:
            reference = TargetDetectionDiagnostics(
                device_id="test", base_path=tmp_logs
            )
            writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)

            rng = np.random.RandomState(42)
            images = [
                rng.randint(0, 256, size=(48, 64, 3), dtype=np.uint8),
                np.full((48, 64, 3), 10, dtype=np.uint8),  # very dark
                np.full((48, 64, 3), 240, dtype=np.uint8),  # very bright
                np.tile(
                    np.linspace(0, 255, 64, dtype=np.uint8), (48, 1)
                ),  # grayscale gradient
            ]

            for image in images:
                expected = reference.analyze_image_quality(image)
                got = writer._analyze_frame_quality(image)
                for key in _DIAGNOSTIC_COLUMNS[2:]:
                    self.assertAlmostEqual(
                        got[key], expected[key], places=9, msg=key
                    )
                self.assertNotIn("image_shape", got)
            self._close_writer(writer)
        finally:
            shutil.rmtree(tmp_logs, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Arena-region masking
    # ------------------------------------------------------------------ #

    def test_masked_metrics_match_masked_pixels(self):
        """With a mask, statistics cover only the masked (arena) pixels."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)

        # 8x8 arena: 60 pixels at 100, 4 pixels at 200 -> mean 106.25
        gray = np.zeros((12, 12), dtype=np.uint8)
        gray[0:8, 0:8] = 100
        gray[0, 0] = 200
        gray[0, 1] = 200
        gray[1, 0] = 200
        gray[1, 1] = 200
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[0:8, 0:8] = 255

        pixels = gray[mask > 0].astype(np.float64)
        got = writer._analyze_frame_quality(gray, mask=mask)

        self.assertAlmostEqual(got["mean_brightness"], float(pixels.mean()), places=9)
        self.assertAlmostEqual(
            got["median_brightness"], float(np.median(pixels)), places=9
        )
        self.assertAlmostEqual(got["std_brightness"], float(pixels.std()), places=9)
        self.assertAlmostEqual(got["min_brightness"], 100.0, places=9)
        self.assertAlmostEqual(got["max_brightness"], 200.0, places=9)
        self.assertAlmostEqual(got["contrast_rms"], got["std_brightness"], places=9)
        self.assertAlmostEqual(got["contrast_range"], 100.0, places=9)

        # Two-value histogram: 60/64 at 100, 4/64 at 200
        # (places=6: calcHist accumulates in float32)
        p = np.array([60.0, 4.0]) / 64.0
        expected_entropy = float(-np.sum(p * np.log2(p)))
        self.assertAlmostEqual(
            got["histogram_entropy"], expected_entropy, places=6
        )

        # Whole-frame metrics must differ (background dominates)
        whole = writer._analyze_frame_quality(gray)
        self.assertNotAlmostEqual(whole["mean_brightness"], got["mean_brightness"])
        self._close_writer(writer)

    def test_edge_density_counted_only_inside_mask(self):
        """Edge pixels outside the arena mask are not counted."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)

        # Strong vertical edge between columns 19 and 20
        frame = np.zeros((20, 40), dtype=np.uint8)
        frame[:, :20] = 250
        # Mask covering only columns 0..15, far from the edge
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[:, :16] = 255

        masked = writer._analyze_frame_quality(frame, mask=mask)
        whole = writer._analyze_frame_quality(frame)

        self.assertEqual(masked["edge_density"], 0.0)
        self.assertGreater(whole["edge_density"], 0.0)
        self._close_writer(writer)

    def test_get_arena_mask_builds_union_of_roi_polygons(self):
        """_get_arena_mask returns a full-frame mask of all ROI polygons."""
        rois = [
            ROI(polygon=((5, 5), (15, 5), (15, 15), (5, 15)), idx=1, value=1),
            ROI(polygon=((30, 5), (40, 5), (40, 15), (30, 15)), idx=2, value=1),
        ]
        writer = self._create_writer(erase_old_db=True, rois=rois)

        frame = np.zeros((20, 50, 3), dtype=np.uint8)
        mask = writer._get_arena_mask(frame.shape)

        self.assertEqual(mask.shape, (20, 50))
        # Two 11x11 filled squares (fillPoly includes the boundary)
        self.assertEqual(int(np.count_nonzero(mask)), 2 * 11 * 11)
        self.assertEqual(int(mask[10, 10]), 255)
        self.assertEqual(int(mask[10, 35]), 255)
        self.assertEqual(int(mask[0, 0]), 0)
        self._close_writer(writer)

    def test_arena_mask_is_cached_until_shape_changes(self):
        """The mask is rebuilt only when the frame shape changes."""
        rois = [
            ROI(polygon=((5, 5), (15, 5), (15, 15), (5, 15)), idx=1, value=1),
        ]
        writer = self._create_writer(erase_old_db=True, rois=rois)

        frame = np.zeros((20, 50, 3), dtype=np.uint8)
        first = writer._get_arena_mask(frame.shape)
        second = writer._get_arena_mask(frame.shape)
        self.assertIs(first, second)

        other = writer._get_arena_mask((30, 50, 3))
        self.assertIsNot(first, other)
        self.assertEqual(other.shape, (30, 50))
        self._close_writer(writer)

    def test_no_rois_falls_back_to_whole_frame(self):
        """Without usable ROIs, the mask is None and metrics cover the frame."""
        writer = self._create_writer(erase_old_db=True, rois=[])

        frame = np.full((20, 50, 3), 100, dtype=np.uint8)
        mask = writer._get_arena_mask(frame.shape)
        self.assertIsNone(mask)

        got = writer._analyze_frame_quality(frame, mask=mask)
        self.assertAlmostEqual(got["mean_brightness"], 100.0, places=9)
        self._close_writer(writer)

    def test_flush_records_arena_only_metrics(self):
        """End-to-end: flush() stores metrics from the union of the ROIs."""
        rois = [
            ROI(polygon=((5, 5), (15, 5), (15, 15), (5, 15)), idx=1, value=1),
            ROI(polygon=((30, 5), (40, 5), (40, 15), (30, 15)), idx=2, value=1),
        ]
        writer = self._create_writer(
            erase_old_db=True, enable_diagnostics=True, rois=rois
        )

        frame = np.full((20, 50, 3), 200, dtype=np.uint8)
        frame[5:16, 5:16] = 40  # arena 1
        frame[5:16, 30:41] = 40  # arena 2
        writer.flush(0, frame)
        self._close_writer(writer)

        rows = self._fetch_all(
            "SELECT mean_brightness, min_brightness, max_brightness, "
            "histogram_entropy FROM diagnostic"
        )
        self.assertEqual(len(rows), 1)
        mean_b, min_b, max_b, entropy = rows[0]
        # Arena pixels only: whole-frame mean would be ~182
        self.assertAlmostEqual(mean_b, 40.0, places=6)
        self.assertAlmostEqual(min_b, 40.0, places=6)
        self.assertAlmostEqual(max_b, 40.0, places=6)
        self.assertAlmostEqual(entropy, 0.0, places=6)

    # ------------------------------------------------------------------ #
    # Recording: throttling and timestamps
    # ------------------------------------------------------------------ #

    def test_throttle_one_row_per_period(self):
        """At most one row per DIAGNOSTICS_PERIOD_SECONDS of experiment time."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)
        # ticks for period=1.0s: 0,0,0,0 -> 1,1,1 -> 2,2 => 3 unique ticks
        for t in (0, 100, 200, 300, 1000, 1100, 1400, 2000, 2100):
            writer.flush(t, self.frame)
        self._close_writer(writer)

        rows = self._diagnostic_rows()
        self.assertEqual(len(rows), 3)

    def test_rows_use_unix_timestamps(self):
        """Stored 't' is a wall-clock Unix timestamp in seconds."""
        start = int(time.time())
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)
        writer.flush(0, self.frame)
        writer.flush(1000, self.frame)
        self._close_writer(writer)
        end = int(time.time())

        rows = self._diagnostic_rows()
        self.assertEqual(len(rows), 2)
        previous = 0
        for (ts,) in rows:
            self.assertIsInstance(ts, int)
            self.assertGreaterEqual(ts, start - 5)
            self.assertLessEqual(ts, end + 5)
            self.assertGreaterEqual(ts, previous)
            previous = ts

    def test_no_rows_when_frame_is_none(self):
        """flush(t, None) must not record diagnostics."""
        writer = self._create_writer(erase_old_db=True, enable_diagnostics=True)
        writer.flush(0, None)
        writer.flush(2000, None)
        self._close_writer(writer)

        self.assertEqual(len(self._diagnostic_rows()), 0)

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #

    def test_retention_deletes_expired_rows(self):
        """Rows older than retention_minutes are deleted by the sweep."""
        start = int(time.time())
        writer = self._create_writer(
            erase_old_db=True,
            enable_diagnostics=True,
            diagnostics_retention_minutes=30,
        )

        # Row A: current wall-clock time
        writer.flush(0, self.frame)

        # Row B: fake the writer's clock 2h in the past
        fake_time = Mock()
        fake_time.time.return_value = start - 7200
        with patch.object(sqlite_io, "time", fake_time):
            writer.flush(1000, self.frame)

        # Row C: back to real time; force the retention sweep to fire now
        writer._diagnostics_last_retention_sweep = 0.0
        writer.flush(2000, self.frame)
        self._close_writer(writer)

        rows = self._diagnostic_rows()
        # Row B (2h old, older than the 30min limit) must be gone
        self.assertEqual(len(rows), 2)
        for (ts,) in rows:
            self.assertGreaterEqual(ts, start - 1800)

    def test_retention_zero_keeps_all_rows(self):
        """Retention <= 0 means infinite: nothing is ever deleted."""
        start = int(time.time())
        writer = self._create_writer(
            erase_old_db=True,
            enable_diagnostics=True,
            diagnostics_retention_minutes=0,
        )

        writer.flush(0, self.frame)

        fake_time = Mock()
        fake_time.time.return_value = start - 7200
        with patch.object(sqlite_io, "time", fake_time):
            writer.flush(1000, self.frame)

        writer.flush(2000, self.frame)
        self._close_writer(writer)

        rows = self._diagnostic_rows()
        self.assertEqual(len(rows), 3)
        # The 2h-old row is still there
        self.assertLessEqual(rows[1][0], start - 3600)


if __name__ == "__main__":
    unittest.main()
