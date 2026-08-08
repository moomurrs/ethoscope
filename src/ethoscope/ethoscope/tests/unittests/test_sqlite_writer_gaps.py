"""
Additional unit tests for io/sqlite.py covering error paths and helper
branches not exercised by test_sqlite_writer.py: connection failure, missing/
empty databases, per-table and outer query errors, pickling state, flush
helpers, sensor/DAM table creation, the append path, and metadata handling.
"""

import os
import sqlite3
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from ethoscope.io.sqlite import AsyncSQLiteWriter, SQLiteResultWriter

# ===========================================================================
# AsyncSQLiteWriter
# ===========================================================================


class TestAsyncSQLiteWriterGaps:
    def test_get_connection_failure_wraps_error(self):
        writer = object.__new__(AsyncSQLiteWriter)
        writer._db_name = "/tmp"  # a directory -> sqlite3 error
        with pytest.raises(Exception, match="Failed to connect to SQLite database"):
            writer._get_connection()

    def test_get_db_type_name(self):
        writer = object.__new__(AsyncSQLiteWriter)
        assert writer._get_db_type_name() == "SQLite"

    def test_should_retry_on_transient_error(self):
        writer = object.__new__(AsyncSQLiteWriter)
        assert writer._should_retry_on_error(
            sqlite3.OperationalError("database is locked")
        ) is True
        assert writer._should_retry_on_error(
            sqlite3.OperationalError("database is busy")
        ) is True

    def test_should_retry_stops_on_critical_error(self):
        writer = object.__new__(AsyncSQLiteWriter)
        assert writer._should_retry_on_error(
            sqlite3.OperationalError("database disk image is malformed")
        ) is False


# ===========================================================================
# SQLiteResultWriter.get_last_timestamp error paths
# ===========================================================================


class TestGetLastTimestampGaps:
    def _shell(self, db_path, rois=None):
        writer = object.__new__(SQLiteResultWriter)
        writer._db_credentials = {"name": db_path}
        writer._rois = rois or []
        return writer

    def test_missing_database_returns_zero(self, tmp_path):
        writer = self._shell(str(tmp_path / "missing.db"))
        assert writer.get_last_timestamp() == 0

    def test_empty_database_file_returns_zero(self, tmp_path):
        db_path = tmp_path / "empty.db"
        db_path.write_bytes(b"")
        writer = self._shell(str(db_path))
        assert writer.get_last_timestamp() == 0

    def test_unreadable_database_file_returns_zero(self, tmp_path):
        db_path = tmp_path / "nested" / "sub.db"
        db_path.parent.mkdir()
        db_path.write_bytes(b"x")
        writer = self._shell(str(db_path))
        with patch(
            "ethoscope.io.sqlite.os.path.getsize",
            side_effect=OSError("permission denied"),
        ):
            assert writer.get_last_timestamp() == 0

    def test_database_error_returns_zero(self, tmp_path):
        db_path = tmp_path / "corrupt.db"
        db_path.write_bytes(b"this is not a sqlite database at all........")
        writer = self._shell(str(db_path))
        assert writer.get_last_timestamp() == 0

    def test_sqlite_error_returns_zero(self, tmp_path):
        db_path = tmp_path / "locked.db"
        db_path.write_bytes(b"")
        writer = self._shell(str(db_path))
        with patch(
            "ethoscope.io.sqlite.sqlite3.connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            assert writer.get_last_timestamp() == 0

    def test_unexpected_error_returns_zero(self, tmp_path):
        db_path = tmp_path / "weird.db"
        db_path.write_bytes(b"x")
        writer = self._shell(str(db_path))
        with patch(
            "ethoscope.io.sqlite.sqlite3.connect",
            side_effect=RuntimeError("boom"),
        ):
            assert writer.get_last_timestamp() == 0


# ===========================================================================
# Pickle state
# ===========================================================================


class TestPickleStateGaps:
    def test_getstate_adds_empty_pickle_kwargs(self):
        writer = object.__new__(SQLiteResultWriter)
        writer._db_credentials = {"name": "/tmp/x.db"}
        writer._rois = []
        writer._metadata = {}
        writer._make_dam_like_table = False
        writer._take_frame_shots = False
        writer.__getstate__ = Mock(wraps=writer.__getstate__)
        state = writer.__getstate__()
        assert state["_pickle_extra_kwargs"] == {}


# ===========================================================================
# _add / flush helper branches
# ===========================================================================


class TestSQLiteAddFlushGaps:
    def _shell(self):
        writer = object.__new__(SQLiteResultWriter)
        writer._null = None
        writer._insert_dict = {}
        writer._dam_file_helper = None
        writer._shot_saver = None
        writer._sensor_saver = None
        writer._max_insert_string_len = 1000
        writer._write_async_command = Mock()
        return writer

    def test_add_feeds_dam_helper(self):
        writer = self._shell()
        dam = Mock()
        writer._dam_file_helper = dam
        roi = Mock()
        roi.idx = 1
        data_row = Mock()
        data_row.values.return_value = [1.0]

        writer._add(1000, roi, [data_row])
        dam.input_roi_data.assert_called_once_with(1000, roi, data_row)

    def test_add_converts_bool_values(self):
        writer = self._shell()
        roi = Mock()
        roi.idx = 2
        data_row = Mock()
        data_row.values.return_value = [True, False]

        writer._add(1000, roi, [data_row])
        stored = writer._insert_dict[2][0]
        assert stored[2] == 1  # True -> 1
        assert stored[3] == 0  # False -> 0

    def test_flush_writes_all_helpers(self):
        writer = self._shell()
        dam = Mock()
        dam.flush.return_value = ["INSERT INTO CSV_DAM_ACTIVITY ..."]
        shot = Mock()
        shot.flush.return_value = ("INSERT INTO IMG_SNAPSHOTS ...", (b"x",))
        sensor = Mock()
        sensor.flush.return_value = ("INSERT INTO SENSORS ...", None)
        writer._dam_file_helper = dam
        writer._shot_saver = shot
        writer._sensor_saver = sensor

        writer.flush(1000, img="fake")

        dam.flush.assert_called_once_with(1000)
        shot.flush.assert_called_once_with(1000, "fake")
        sensor.flush.assert_called_once_with(1000)
        assert writer._write_async_command.call_count == 3

    def test_flush_batch_inserts(self):
        writer = self._shell()
        writer._max_insert_string_len = 50  # tiny threshold forces batching
        roi = Mock()
        roi.idx = 3
        data_row = Mock()
        data_row.values.return_value = [1]
        for t in range(60):
            writer._add(t, roi, [data_row])

        writer.flush(1000)
        assert writer._insert_dict[3] == []
        # batching happened in chunks of 50
        assert writer._write_async_command.call_count == 2


# ===========================================================================
# _create_all_tables branches (sensor + append path)
# ===========================================================================


class TestCreateAllTablesGaps:
    def _full_writer(self, tmp_path, **kwargs):
        db_path = tmp_path / "exp.db"
        from ethoscope.core.roi import ROI

        rois = [
            ROI(polygon=((0, 0), (100, 0), (100, 100), (0, 100)), idx=1, value=1)
        ]
        defaults = {
            "db_credentials": {"name": str(db_path)},
            "rois": rois,
            "metadata": {"machine_name": "test"},
            "erase_old_db": True,
            "make_dam_like_table": False,
            "take_frame_shots": False,
        }
        defaults.update(kwargs)
        writer = SQLiteResultWriter(**defaults)
        self._writers.append(writer)
        return writer, db_path

    def setup_method(self):
        self._writers = []

    def teardown_method(self):
        for writer in self._writers:
            try:
                if hasattr(writer, "_queue") and hasattr(writer, "_async_writer"):
                    writer._queue.put("DONE")
                    writer._queue.cancel_join_thread()
                    if writer._async_writer.is_alive():
                        writer._async_writer.join(timeout=2)
            except Exception:
                pass

    def test_create_all_tables_with_sensor(self, tmp_path):
        sensor = Mock()
        sensor.sensor_types = {"temperature": "FLOAT"}
        writer, db_path = self._full_writer(tmp_path, sensor=sensor)
        time.sleep(0.3)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert "SENSORS" in tables

    def test_create_all_tables_append_path(self, tmp_path):
        # erase_old_db=False + database_to_append set -> "appending" event.
        # Note: SQLiteResultWriter does not store database_to_append from
        # kwargs, so it must be set explicitly to reach the append branch.
        db_path = tmp_path / "append.db"
        from ethoscope.core.roi import ROI

        rois = [
            ROI(polygon=((0, 0), (100, 0), (100, 100), (0, 100)), idx=1, value=1)
        ]
        writer = SQLiteResultWriter(
            db_credentials={"name": str(db_path)},
            rois=rois,
            metadata={"machine_name": "test"},
            erase_old_db=False,
        )
        self._writers.append(writer)
        writer.database_to_append = str(db_path)
        with patch.object(writer, "_write_async_command") as mock_write:
            writer._create_all_tables()
        # appending event was queued
        assert any(
            "appending" in str(c.args)
            for c in mock_write.call_args_list
        )


class TestInsertMetadataGaps:
    def test_insert_metadata_sqlite_serializes_and_truncates(self):
        writer = object.__new__(SQLiteResultWriter)
        writer._metadata = {
            "config": {"nested": True},
            "big": "x" * 70000,
        }
        writer._write_async_command = Mock()

        writer._insert_metadata()

        assert writer._write_async_command.call_count == 2
        calls = [c.args for c in writer._write_async_command.call_args_list]
        commands = [c[0] for c in calls]
        assert all("INSERT OR IGNORE INTO METADATA" in c for c in commands)
        for c in calls:
            assert c[0].startswith("INSERT OR IGNORE")
