"""
Additional unit tests for io/base.py covering paths not exercised by
test_base_writer.py: the full BaseAsyncSQLWriter.run() loop, BaseResultWriter
construction with frame-shot/sensor helpers, writer restart, queue handling
in __exit__/flush, and dbAppender database type detection + writer creation.
"""

import os
import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from ethoscope.io.base import (
    ASYNC_WRITER_TIMEOUT,
    MAX_BUFFERED_COMMANDS,
    BaseAsyncSQLWriter,
    BaseResultWriter,
    ImgSnapshotHelper,
    SensorDataHelper,
    dbAppender,
)
from ethoscope.io.helpers import Null

# ===========================================================================
# BaseAsyncSQLWriter.run() loop
# ===========================================================================


class _ConcreteAsyncWriter(BaseAsyncSQLWriter):
    """Concrete writer that records commands it executes."""

    def __init__(self, queue, erase_old_db=True):
        super().__init__(queue, erase_old_db)
        self._initialized = False
        self.executed = []

    def _initialize_database(self):
        self._initialized = True

    def _get_connection(self):
        return Mock()

    def _get_db_type_name(self):
        return "Test"

    def _should_retry_on_error(self, error):
        return False

    def _handle_command_error(self, error, command, args):
        self.executed.append(("error", command))


class TestBaseAsyncSQLWriterRun:
    def _make_queue(self, items):
        q = Mock()
        q.get.side_effect = list(items)
        q.empty.return_value = True
        return q

    def test_run_processes_commands_and_done(self):
        db = Mock()
        writer = _ConcreteAsyncWriter(self._make_queue([("SELECT 1", None), "DONE"]))
        writer._get_connection = Mock(return_value=db)

        writer.run()

        assert writer._initialized is True
        assert writer._ready_event.is_set()
        db.cursor().execute.assert_called_once_with("SELECT 1")
        db.commit.assert_called_once()
        db.close.assert_called_once()
        writer._queue.close.assert_called_once()

    def test_run_executes_commands_with_args(self):
        db = Mock()
        writer = _ConcreteAsyncWriter(
            self._make_queue([("INSERT INTO t VALUES (?)", (1,)), "DONE"])
        )
        writer._get_connection = Mock(return_value=db)

        writer.run()

        db.cursor().execute.assert_called_once_with("INSERT INTO t VALUES (?)", (1,))

    def test_run_stops_on_critical_error(self):
        db = Mock()
        db.cursor().execute.side_effect = RuntimeError("disk full")
        writer = _ConcreteAsyncWriter(self._make_queue([("CMD", None)]))
        writer._get_connection = Mock(return_value=db)

        # _should_retry_on_error returns False -> writer stops after the error
        writer.run()

        assert writer.executed == [("error", "CMD")]

    def test_run_retries_transient_error(self):
        db = Mock()
        db.cursor().execute.side_effect = [RuntimeError("busy"), None]
        writer = _ConcreteAsyncWriter(
            self._make_queue([("CMD", None), ("CMD2", None), "DONE"])
        )
        writer._get_connection = Mock(return_value=db)

        # Override to retry on the first error
        writer._should_retry_on_error = lambda error: error.args == ("busy",)
        writer.run()

        assert db.cursor().execute.call_count >= 2

    def test_run_keyboard_interrupt_sets_ready(self):
        writer = _ConcreteAsyncWriter(Mock())
        writer._get_connection = Mock(side_effect=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            writer.run()
        assert writer._ready_event.is_set()

    def test_run_init_exception_sets_ready(self):
        writer = _ConcreteAsyncWriter(Mock())
        writer._get_connection = Mock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            writer.run()
        assert writer._ready_event.is_set()


# ===========================================================================
# BaseResultWriter construction helpers
# ===========================================================================


class _ShellResultWriter(BaseResultWriter):
    """Minimal BaseResultWriter subclass that never spawns a real process."""

    _database_type = "SQLite3"
    _null = Null()

    def __init__(self, db_credentials, rois, ready=True, alive=True, **kwargs):
        self._ready = ready
        self._alive = alive
        super().__init__(db_credentials, rois, **kwargs)

    def _create_async_writer(self, db_credentials, erase_old_db, **kwargs):
        mock = Mock()
        event = threading.Event()
        if self._ready:
            event.set()
        mock._ready_event = event
        mock.is_alive.return_value = self._alive
        return mock

    def _create_all_tables(self):
        pass


class TestBaseResultWriterConstruction:
    def test_init_with_frame_shots_and_sensor(self):
        rois = []
        sensor = Mock()
        sensor.sensor_types = {"temperature": "FLOAT"}
        writer = _ShellResultWriter(
            {"name": "/tmp/x.db"},
            rois,
            take_frame_shots=True,
            make_dam_like_table=False,
            sensor=sensor,
        )
        assert isinstance(writer._shot_saver, ImgSnapshotHelper)
        assert isinstance(writer._sensor_saver, SensorDataHelper)
        assert writer._dam_file_helper is None
        writer._async_writer._ready_event.is_set()
        writer._queue.put("DONE")

    def test_init_writer_still_alive_times_out(self):
        with pytest.raises(Exception, match="failed to initialize within 30 seconds"):
            _ShellResultWriter({"name": "/tmp/x.db"}, [], ready=False, alive=True)

    def test_init_writer_died(self):
        with pytest.raises(Exception, match="process died during initialization"):
            _ShellResultWriter({"name": "/tmp/x.db"}, [], ready=False, alive=False)

    def test_create_async_writer_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseResultWriter._create_async_writer(
                object.__new__(BaseResultWriter), {}, True
            )


# ===========================================================================
# BaseResultWriter __exit__ / __setstate__ / flush
# ===========================================================================


def _writer_shell():
    writer = object.__new__(BaseResultWriter)
    writer._queue = Mock()
    writer._queue.empty.return_value = True
    writer._async_writer = Mock()
    writer._async_writer.is_alive.return_value = False
    writer._insert_dict = {}
    writer._dam_file_helper = None
    writer._shot_saver = None
    writer._sensor_saver = None
    writer._database_type = "SQLite3"
    writer._write_async_command = Mock()
    return writer


class TestBaseResultWriterExit:
    def test_exit_flushes_string_inserts(self):
        writer = _writer_shell()
        writer._insert_dict = {1: "INSERT INTO ROI_1 VALUES (1)"}
        writer.__exit__(None, None, None)
        writer._write_async_command.assert_called()
        writer._queue.put.assert_called()  # "DONE" sent
        writer._queue.cancel_join_thread.assert_called_once()

    def test_exit_ignores_list_inserts(self):
        writer = _writer_shell()
        writer._insert_dict = {1: [(1, 2)]}
        writer.__exit__(None, None, None)
        # list inserts are left to the subclass; still closes cleanly
        assert writer._queue.put.called

    def test_exit_survives_write_error(self):
        writer = _writer_shell()
        writer._insert_dict = {}
        writer._write_async_command.side_effect = RuntimeError("boom")
        writer.__exit__(None, None, None)  # should not raise
        writer._queue.put.assert_called()  # "DONE" still sent

    def test_setstate_recreates_queue_and_writer(self):
        writer = object.__new__(BaseResultWriter)
        writer._db_credentials = {"name": "/tmp/x.db"}
        writer._rois = []
        writer._metadata = {}
        writer._make_dam_like_table = False
        writer._take_frame_shots = False
        writer._pickle_extra_kwargs = {}
        writer._create_async_writer = Mock(return_value=Mock())

        state = {"_queue": None, "_async_writer": None, "_pickle_init_args": {}}
        writer.__setstate__(state)

        assert writer._queue is not None
        assert writer._async_writer is not None
        writer._create_async_writer.assert_called_once()


class TestBaseResultWriterFlush:
    def test_flush_writes_shot_saver(self):
        writer = _writer_shell()
        shot = Mock()
        shot.flush.return_value = ("INSERT INTO IMG_SNAPSHOTS ...", (b"x",))
        writer._shot_saver = shot
        writer.flush(1000, img="fake")
        shot.flush.assert_called_once_with(1000, "fake")

    def test_flush_writes_string_insert_dict(self):
        writer = _writer_shell()
        writer._insert_dict = {1: "INSERT INTO ROI_1 VALUES (1)"}
        writer._max_insert_string_len = 1
        writer.flush(1000)
        writer._write_async_command.assert_called_once()

    def test_flush_leaves_list_insert_dict_alone(self):
        writer = _writer_shell()
        writer._insert_dict = {1: [(1, 2)]}
        writer._max_insert_string_len = 1
        writer.flush(1000)
        writer._write_async_command.assert_not_called()

    def test_add_builds_insert_command(self):
        writer = _writer_shell()
        writer._null = Null()
        writer._dam_file_helper = None
        roi = Mock()
        roi.idx = 7
        data_row = Mock()
        data_row.values.return_value = [42]
        writer._add(1000, roi, [data_row])
        assert "INSERT INTO ROI_7" in writer._insert_dict[7]

    def test_initialise_var_map(self):
        writer = _writer_shell()
        var = Mock()
        var.header_name = "x"
        var.sql_data_type = "SMALLINT"
        var.functional_type = "distance"
        data_row = Mock()
        data_row.values.return_value = [var]
        writer._initialise_var_map(data_row)
        writer._write_async_command.assert_called()


class TestBaseResultWriterAppend:
    def test_append_returns_last_timestamp(self):
        writer = _writer_shell()
        writer.get_last_timestamp = Mock(return_value=1234)
        assert writer.append() == 1234

    def test_close_is_noop(self):
        writer = _writer_shell()
        writer.close()  # should not raise


# ===========================================================================
# BaseResultWriter resilience internals
# ===========================================================================


class TestResilienceInternals:
    def _shell(self):
        writer = object.__new__(BaseResultWriter)
        writer._queue = Mock()
        writer._async_writer = Mock()
        writer._async_writer.is_alive.return_value = True
        writer._failed_commands_buffer = deque(maxlen=MAX_BUFFERED_COMMANDS)
        writer._writer_restart_count = 0
        writer._last_restart_time = 0
        writer._db_credentials = {"name": "/tmp/x.db"}
        return writer

    def test_restart_async_writer_success(self):
        writer = self._shell()
        writer._last_restart_time = 0
        new_writer = Mock()
        new_writer._ready_event = threading.Event()
        new_writer._ready_event.set()
        writer._create_async_writer = Mock(return_value=new_writer)
        with patch("ethoscope.io.base.time") as mock_time:
            mock_time.time.return_value = 1000
            result = writer._restart_async_writer()
        assert result is True
        assert writer._writer_restart_count == 1
        assert writer._last_restart_time == 1000

    def test_write_async_command_resilient_retries_on_exception(self):
        writer = self._shell()
        writer._queue.put.side_effect = [RuntimeError("full"), None]
        with patch("ethoscope.io.base.time.sleep"):
            result = writer._write_async_command_resilient("CMD")
        assert result is True
        assert writer._queue.put.call_count == 2

    def test_write_async_command_resilient_buffers_on_exhaustion(self):
        writer = self._shell()
        writer._queue.put.side_effect = RuntimeError("full")
        with patch("ethoscope.io.base.time.sleep"):
            result = writer._write_async_command_resilient("CMD")
        assert result is False
        assert len(writer._failed_commands_buffer) == 1

    def test_buffer_command_exception(self):
        writer = self._shell()
        writer._failed_commands_buffer = Mock()
        writer._failed_commands_buffer.append.side_effect = RuntimeError("boom")
        result = writer._buffer_command("CMD", None)
        assert result is False

    def test_retry_buffered_commands_puts_back_on_dead_writer(self):
        writer = self._shell()
        writer._failed_commands_buffer.append(("CMD", None, time.time()))
        writer._async_writer.is_alive.return_value = False
        writer._retry_buffered_commands()
        # command should be put back on the buffer for later retry
        assert len(writer._failed_commands_buffer) == 1

    def test_log_io_diagnostics_with_disk_info(self, tmp_path):
        writer = self._shell()
        db_path = str(tmp_path / "x.db")
        writer._db_credentials = {"name": db_path}
        writer.log_io_diagnostics("test")  # should not raise


# ===========================================================================
# dbAppender
# ===========================================================================


class TestDbAppenderGaps:
    def test_create_sqlite_writer(self):
        appender = object.__new__(dbAppender)
        appender.database_to_append = "/tmp/x.db"
        appender.db_credentials = {"name": "/tmp/x.db"}
        appender.rois = []
        appender.metadata = None
        appender.make_dam_like_table = False
        appender.take_frame_shots = False
        appender.sensor = None
        appender.args = ()
        appender.kwargs = {}
        appender._find_sqlite_database_path = Mock(return_value="/tmp/x.db")

        writer = Mock()
        with patch("ethoscope.io.sqlite.SQLiteResultWriter", return_value=writer):
            appender._create_sqlite_writer()

        assert appender._writer is writer
        assert appender.kwargs["erase_old_db"] is False

    def test_find_sqlite_database_path_walks_tree(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        db_file = sub / "exp.db"
        db_file.write_bytes(b"x")

        appender = object.__new__(dbAppender)
        # New implementation uses pathlib.Path, so patch Path methods
        # Keep backward compat for os.path.exists mock as well
        def fake_exists(self: object) -> bool:  # type: ignore[no-untyped-def]
            # self is a Path instance
            s = str(self)  # type: ignore[arg-type]
            if s == "exp.db":
                return False
            if s in {str(db_file), "/ethoscope_data/results", str(sub)}:
                return True
            # For other paths, use real exists
            import pathlib

            return pathlib.Path(s).exists() if s != "exp.db" else False

        with patch("ethoscope.io.base.Path.exists", fake_exists):
            with patch(
                "ethoscope.io.base.Path.rglob",
                return_value=[db_file],
            ):
                result = appender._find_sqlite_database_path("exp.db")
        assert result == str(db_file)

    def test_get_available_databases(self):
        appender = object.__new__(dbAppender)
        with patch(
            "ethoscope.io.cache.get_all_databases_info",
            create=True,
            return_value={
                "SQLite": {
                    "exp.db": {"file_exists": True, "filesize": 50000, "path": "/x"}
                },
            },
        ):
            dbs = appender.get_available_databases({"name": "test"})
        names = {d["name"] for d in dbs}
        assert "exp.db" in names
