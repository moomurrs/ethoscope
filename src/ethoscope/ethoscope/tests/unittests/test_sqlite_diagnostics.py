"""
Unit tests for the optional 'diagnostic' table in SQLiteResultWriter (io/sqlite.py).

Tests cover:
- Table creation only when enable_diagnostics is True (fresh and append flows)
- Table absence when the toggle is disabled
- Schema (columns/types) and timestamp index
- Image quality calculations identical to TargetDetectionDiagnostics
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
        """_analyze_frame_quality matches TargetDetectionDiagnostics exactly."""
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
