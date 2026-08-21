"""Database Writers for Ethoscope Experiment Data Storage.

This module provides various classes for storing experimental tracking data
from the Ethoscope behavioral monitoring system. Uses SQLite as the sole
database backend.

Architecture:
    - :class:`BaseAsyncSQLWriter` - template for async DB writers
    - :class:`BaseResultWriter` - coordinator for tracking data persistence
    - :class:`dbAppender` - append to existing database
    - Helpers are injected via composition (see :mod:`ethoscope.io.helpers`).

Design patterns:
    - Producer-Consumer via :class:`multiprocessing.JoinableQueue`
    - Template Method for DB-specific writers
    - Composition over inheritance for helpers
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import queue as queue_mod
import re
import time
import weakref
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self, cast

from ._constants import (
    ASYNC_WRITER_TIMEOUT,
    BUFFERED_COMMAND_MAX_AGE,
    MAX_BUFFERED_COMMANDS,
    MAX_BUFFERED_RETRY_FAILURES,
    MAX_DB_RETRIES,
    MAX_RETRY_DELAY,
    METADATA_MAX_VALUE_LENGTH,
    MIN_DB_SIZE_BYTES,
    QUEUE_CHECK_INTERVAL,
    RESTART_THROTTLE_SECONDS,
    RETRY_BASE_DELAY,
)
from .helpers import DAMFileHelper, ImgSnapshotHelper, SensorDataHelper

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from multiprocessing import JoinableQueue
    from multiprocessing.synchronize import Event as MpEvent

    from ethoscope.core.roi import ROI

    from ._types import DataPointProtocol, DbCredentials, MetadataDict, SqlArgs

_LOGGER: Final = logging.getLogger(__name__)


def _finalize_queue(qref: weakref.ref[JoinableQueue[Any]]) -> None:
    """Close a queue via weakref so finalize does not pin it alive."""
    q = qref()
    if q is not None:
        with contextlib.suppress(Exception):
            q.cancel_join_thread()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WriterNotReadyError(RuntimeError):
    """Raised when the async writer fails to become ready in time."""

    def __init__(self) -> None:
        super().__init__(
            f"Async database writer failed to initialize within "
            f"{int(ASYNC_WRITER_TIMEOUT)} seconds - check database connection"
        )


class WriterInitError(RuntimeError):
    """Raised when the async writer process dies during init."""

    def __init__(self) -> None:
        super().__init__(
            "Async database writer process died during initialization - "
            "check database configuration and logs"
        )


class MissingDatabaseNameError(ValueError):
    """Raised when database_to_append is not provided."""

    def __init__(self) -> None:
        super().__init__("database_to_append parameter is required")


class DatabaseFileNotFoundError(FileNotFoundError):
    """Raised when SQLite database file cannot be found."""

    def __init__(self, path: str) -> None:
        super().__init__(f"SQLite database not found: {path}")
        self.path = path


# ---------------------------------------------------------------------------
# Base async writer - template method with SRP decomposition
# ---------------------------------------------------------------------------


class BaseAsyncSQLWriter(multiprocessing.Process):
    """Abstract async SQL writer running in a separate process.

    Template Method: subclasses implement DB-specific hooks.

    Attributes:
        _queue: Queue for SQL commands.
        _erase_old_db: Whether to erase existing DB on startup.
        _ready_event: Signals readiness to accept commands.
    """

    def __init__(
        self,
        queue: JoinableQueue[Any],
        erase_old_db: bool = True,
    ) -> None:
        self._queue: JoinableQueue[Any] = queue
        self._erase_old_db: bool = erase_old_db
        self._ready_event: MpEvent = multiprocessing.Event()
        super().__init__()

    # -- template ---------------------------------------------------------

    def run(self) -> None:
        """Main process loop - init DB, signal ready, process queue."""
        db: Any = None

        try:
            _LOGGER.info("%s async writer starting up...", self._get_db_type_name())
            self._initialize_database()
            db = self._get_connection()
            _LOGGER.info(
                "%s database connection established successfully",
                self._get_db_type_name(),
            )
            _LOGGER.info(
                "%s async writer ready to accept commands", self._get_db_type_name()
            )
            self._ready_event.set()
            self._run_command_loop(db)

        except KeyboardInterrupt:
            _LOGGER.warning(
                "%s async process interrupted with KeyboardInterrupt",
                self._get_db_type_name(),
            )
            self._ready_event.set()
            raise
        except Exception:  # process boundary: must set ready event on fatal init errors
            _LOGGER.exception(
                "%s async process stopped with an exception",
                self._get_db_type_name(),
            )
            self._ready_event.set()
            raise
        finally:
            _LOGGER.info("Closing async %s writer", self._get_db_type_name().lower())
            self._drain_queue()
            if db is not None:
                with contextlib.suppress(Exception):
                    db.close()

    def _run_command_loop(self, db: Any) -> None:
        """Consume queue messages until the DONE sentinel arrives."""
        do_run = True
        while do_run:
            command: str | None = None
            args: SqlArgs = None
            try:
                msg = self._queue.get()
                if msg == "DONE":
                    do_run = False
                    continue
                command, args = cast("tuple[str, SqlArgs]", msg)
                if command is not None:
                    do_run = self._execute_command(db, command, args)
            except Exception as exc:  # noqa: BLE001  # process boundary: survive malformed commands
                do_run = self._handle_loop_error(exc, command, args)
            finally:
                if self._queue.empty():
                    time.sleep(QUEUE_CHECK_INTERVAL)

    # -- helpers to keep run() small ------------------------------------

    def _execute_command(self, db: Any, command: str, args: SqlArgs) -> bool:
        """Execute a single SQL command; return whether to continue."""
        cursor = db.cursor()
        if args is None:
            cursor.execute(command)
        else:
            cursor.execute(command, args)
        db.commit()
        return True

    def _handle_loop_error(
        self, exc: Exception, command: str | None, args: SqlArgs
    ) -> bool:
        """Log the error and decide whether the loop should continue."""
        should_continue = self._should_retry_on_error(exc)
        try:
            _LOGGER.exception(
                "Failed to run %s command: %s",
                self._get_db_type_name().lower(),
                command,
            )
            _LOGGER.error("Error details: %s", exc)
            _LOGGER.error("Arguments: %s", args)
            self._handle_command_error(exc, command, args)
        except Exception:
            _LOGGER.exception("Failed to log error details")
            _LOGGER.exception("Did not retrieve queue value or failed to log command")
            return False
        return should_continue

    def _drain_queue(self) -> None:
        """Empty and close the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue_mod.Empty:
                break
        try:
            self._queue.close()
        except Exception:
            _LOGGER.debug("Failed to close queue", exc_info=True)

    # -- abstract hooks ---------------------------------------------------

    def _initialize_database(self) -> None:
        """Initialize DB-specific setup."""
        raise NotImplementedError

    def _get_connection(self) -> Any:
        """Create and return a DB connection."""
        raise NotImplementedError

    def _get_db_type_name(self) -> str:
        """Return DB type name for logging."""
        raise NotImplementedError

    def _should_retry_on_error(self, error: Exception) -> bool:
        """Whether to continue after ``error``."""
        raise NotImplementedError

    def _handle_command_error(
        self, error: Exception, command: str | None, args: SqlArgs
    ) -> None:
        """Handle DB-specific error processing."""
        # Default no-op; subclasses may override
        _ = (error, command, args)


# ---------------------------------------------------------------------------
# Helpers for BaseResultWriter - SRP via composition
# ---------------------------------------------------------------------------


class _BufferedCommandQueue:
    """Resilience buffer for failed commands - composition over inline code."""

    def __init__(self, maxlen: int = MAX_BUFFERED_COMMANDS) -> None:
        self._buffer: deque[tuple[str, SqlArgs, float]] = deque(maxlen=maxlen)
        self._maxlen: int = maxlen

    def append(self, command: str, args: SqlArgs) -> bool:
        """Buffer a command; always returns False (buffered, not sent)."""
        try:
            self._buffer.append((command, args, time.time()))
        except Exception:  # buffering must never break the writer
            _LOGGER.exception("Failed to buffer command")
            return False
        if len(self._buffer) >= self._maxlen:
            _LOGGER.warning(
                "Command buffer full (%s commands), oldest will be dropped",
                self._maxlen,
            )
        return False

    def popleft(self) -> tuple[str, SqlArgs, float] | None:
        """Pop oldest or None if empty."""
        try:
            return self._buffer.popleft()
        except IndexError:
            return None

    def appendleft(self, item: tuple[str, SqlArgs, float]) -> None:
        """Push back to front."""
        self._buffer.appendleft(item)

    def as_deque(self) -> deque[tuple[str, SqlArgs, float]]:
        """Direct deque view (test/compat surface)."""
        return self._buffer

    def replace_deque(self, value: deque[tuple[str, SqlArgs, float]]) -> None:
        """Swap the underlying deque (test/compat surface)."""
        self._buffer = value

    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return bool(self._buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self._buffer.clear()


class _DiagnosticsReporter:
    """I/O diagnostics - separate concern from writer logic."""

    @staticmethod
    def log(
        db_credentials: DbCredentials,
        queue: JoinableQueue[Any],
        status: dict[str, Any],
        context: str = "",
    ) -> None:
        """Emit diagnostics to logger."""
        try:
            db_path = db_credentials.get("name", "unknown")
            _LOGGER.error("Database I/O Issue - %s", context)
            _LOGGER.error("  Database path: %s", db_path)
            _LOGGER.error("  Writer alive: %s", status.get("writer_alive"))
            _LOGGER.error("  Buffered commands: %s", status.get("buffered_commands"))
            _LOGGER.error("  Writer restarts: %s", status.get("restart_count"))
            since = status.get("time_since_last_restart")
            if since is not None:
                _LOGGER.error("  Time since last restart: %.1fs", since)
            else:
                _LOGGER.error("  Never restarted")

            # Disk space
            if db_path != "unknown":
                try:
                    parent = Path(db_path).parent
                    if parent.exists() and hasattr(parent, "stat"):
                        parent.stat()
                        if hasattr(os, "statvfs"):
                            vfs = os.statvfs(parent)
                            free = vfs.f_frsize * vfs.f_bavail
                            total = vfs.f_frsize * vfs.f_blocks
                            _LOGGER.error(
                                "  Disk space: %.2fGB free (%.1f%% of %.2fGB)",
                                free / (1024**3),
                                (free / total * 100) if total else 0,
                                total / (1024**3),
                            )
                except Exception:
                    _LOGGER.exception("  Could not check disk space")

            try:
                _LOGGER.error("  Queue size: %s", queue.qsize())
            except Exception:
                _LOGGER.exception("  Could not check queue size")
        except Exception:
            _LOGGER.exception("Failed to log I/O diagnostics")


# ---------------------------------------------------------------------------
# Base result writer - coordinator (SRP: delegates to helpers)
# ---------------------------------------------------------------------------


class BaseResultWriter:
    """Abstract base for result writers - coordinator with injected helpers.

    Each public method does one thing and delegates resilience/queue logic to
    composed objects.
    """

    _null: ClassVar[Any] = None
    _max_insert_string_len: ClassVar[int] = 1000

    def __init__(  # noqa: PLR0913,PLR0917
        self,
        db_credentials: DbCredentials,
        rois: Sequence[ROI],
        metadata: MetadataDict | None = None,
        make_dam_like_table: bool = True,
        take_frame_shots: bool = False,
        erase_old_db: bool = True,
        sensor: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._queue: JoinableQueue[Any] = multiprocessing.JoinableQueue()
        self._async_writer: BaseAsyncSQLWriter = self._create_async_writer(
            db_credentials, erase_old_db, **kwargs
        )
        self._async_writer.start()

        self._erase_old_db: bool = erase_old_db
        self._last_t: int = 0
        self._last_flush_t: int = 0
        self._last_dam_t: int = 0
        self._metadata: MetadataDict = metadata or {}
        self._rois: Sequence[ROI] = rois
        self._db_credentials: DbCredentials = db_credentials
        self._make_dam_like_table: bool = make_dam_like_table
        self._take_frame_shots: bool = take_frame_shots

        self._buffer = _BufferedCommandQueue()
        self._writer_restart_count: int = 0
        self._last_restart_time: float = 0.0

        self._init_helpers(sensor)
        self._var_map_initialised: bool = False
        self._initialized_rois: set[int] = set()
        self._insert_dict: dict[int, Any] = {}
        # Weakref-based so the queue is not pinned for process lifetime
        self._finalizer = weakref.finalize(
            self, _finalize_queue, weakref.ref(self._queue)
        )

        self._wait_for_writer_ready()
        _LOGGER.warning("Creating database tables...")
        self._create_all_tables()
        _LOGGER.info("Result writer initialised")

    # -- helper init (SRP) ----------------------------------------------

    def _init_helpers(self, sensor: Any | None) -> None:
        """Initialize DAM / snapshot / sensor helpers via composition."""
        if self._make_dam_like_table:
            self._dam_file_helper: DAMFileHelper | None = DAMFileHelper(
                n_rois=len(self._rois)
            )
        else:
            self._dam_file_helper = None

        if self._take_frame_shots:
            # database_type kept for compat - always SQLite3 now
            db_type = getattr(self, "_database_type", "SQLite3")
            self._shot_saver: ImgSnapshotHelper | None = ImgSnapshotHelper(
                database_type=db_type
            )
        else:
            self._shot_saver = None

        if sensor is not None:
            db_type = getattr(self, "_database_type", "SQLite3")
            self._sensor_saver: SensorDataHelper | None = SensorDataHelper(
                sensor, database_type=db_type
            )
            _LOGGER.info("Creating connection to a sensor to store its data in the db")
        else:
            self._sensor_saver = None

    def _wait_for_writer_ready(self) -> None:
        """Block until async writer signals readiness."""
        _LOGGER.warning("Waiting for async writer to initialize database...")
        if not self._async_writer._ready_event.wait(timeout=ASYNC_WRITER_TIMEOUT):
            if self._async_writer.is_alive():
                raise WriterNotReadyError()
            raise WriterInitError()

    # -- abstract factory ------------------------------------------------

    def _create_async_writer(
        self, db_credentials: DbCredentials, erase_old_db: bool, **kwargs: Any
    ) -> BaseAsyncSQLWriter:
        """Create DB-specific async writer - subclasses override."""
        raise NotImplementedError

    def _create_all_tables(self) -> None:
        """Create all required tables - subclasses should implement."""
        raise NotImplementedError

    # -- public API -------------------------------------------------------

    def get_backup_filename(self) -> str | None:
        """Return backup filename from metadata if present."""
        if hasattr(self, "_metadata") and self._metadata:
            return self._metadata.get("backup_filename")  # type: ignore[no-any-return]
        return None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Ensure flush and clean shutdown."""
        _LOGGER.info("Closing result writer...")
        self._flush_insert_dict_for_close()
        try:
            self._write_async_command(
                "INSERT INTO METADATA VALUES (?, ?)",
                ("stop_date_time", str(int(time.time()))),
            )
            self._wait_for_queue_empty()
        except Exception:
            _LOGGER.exception("Error writing metadata stop time")
        finally:
            self._shutdown_async_writer()

    def _flush_insert_dict_for_close(self) -> None:
        """Flush string-based insert buffers (legacy base path)."""
        for key, value in list(self._insert_dict.items()):
            if isinstance(value, str) and value:
                self._write_async_command(value)
                self._insert_dict[key] = ""

    def _shutdown_async_writer(self) -> None:
        """Send DONE and join the async process."""
        _LOGGER.info("Closing async queue")
        try:
            self._queue.put("DONE")
        except Exception:
            _LOGGER.debug("Failed to put DONE", exc_info=True)
        _LOGGER.info("Freeing queue")
        try:
            self._queue.cancel_join_thread()
        except Exception:
            _LOGGER.debug("Failed to cancel join thread", exc_info=True)
        _LOGGER.info("Joining thread")
        if self._async_writer.is_alive():
            self._async_writer.join()
            _LOGGER.info("Joined OK")
        else:
            _LOGGER.info("Process was not started, skipping join")
        if hasattr(self, "_finalizer"):
            self._finalizer.detach()

    def append(self) -> int:
        """Get last timestamp to allow appending."""
        return self.get_last_timestamp()  # type: ignore[no-untyped-call]

    def close(self) -> None:
        """Placeholder - prefer context manager."""
        # Subclasses flush remaining batch data; base does nothing
        return

    # -- pickle support ---------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable multiprocessing objects."""
        state = self.__dict__.copy()
        state["_pickle_init_args"] = {
            "db_credentials": self._db_credentials,
            "rois": self._rois,
            "metadata": self._metadata,
            "make_dam_like_table": self._make_dam_like_table,
            "take_frame_shots": self._take_frame_shots,
        }
        state.pop("_queue", None)
        state.pop("_async_writer", None)
        state.pop("_finalizer", None)
        # deque is picklable but we keep buffer as private composition - remove
        state.pop("_buffer", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Recreate queue and writer after unpickling."""
        self.__dict__.update(state)
        init_args: dict[str, Any] = state.get("_pickle_init_args", {})
        self._queue = multiprocessing.JoinableQueue()
        self._buffer = _BufferedCommandQueue()
        self._async_writer = self._create_async_writer(
            init_args.get("db_credentials", self._db_credentials),
            False,
            **getattr(self, "_pickle_extra_kwargs", {}),
        )
        self._finalizer = weakref.finalize(
            self, _finalize_queue, weakref.ref(self._queue)
        )

    @property
    def metadata(self) -> MetadataDict:
        """Experimental metadata."""
        return self._metadata

    # -- tracking data paths ---------------------------------------------

    def write(
        self, t: int, roi: ROI, data_rows: Sequence[Mapping[str, DataPointProtocol]]
    ) -> None:
        """Write tracking data for a ROI - delegates to helpers."""
        self._last_t = t
        if not self._var_map_initialised:
            self._var_map_initialised = True
            self._initialise_var_map(data_rows[0])

        roi_id = roi.idx
        if roi_id not in self._initialized_rois:
            self._initialise_roi_table(roi, data_rows[0])
            self._initialized_rois.add(roi_id)

        self._add(t, roi, data_rows)

    def flush(self, t: int, img: Any | None = None) -> bool:
        """Flush helpers and batched inserts."""
        self._flush_helpers(t, img)
        self._flush_batched_inserts()
        return False

    def _flush_helpers(self, t: int, img: Any | None) -> None:
        """Flush DAM / snapshot / sensor helpers."""
        if self._dam_file_helper is not None:
            for cmd in self._dam_file_helper.flush(t):
                self._write_async_command(cmd)
        if self._shot_saver is not None and img is not None:
            c_args = self._shot_saver.flush(t, img)
            if c_args is not None:
                self._write_async_command(*c_args)
        if self._sensor_saver is not None:
            c_args = self._sensor_saver.flush(t)
            if c_args is not None:
                self._write_async_command(*c_args)

    def _flush_batched_inserts(self) -> None:
        """Flush string-based batches that exceed threshold."""
        for key, value in list(self._insert_dict.items()):
            if isinstance(value, str) and len(value) > self._max_insert_string_len:
                self._write_async_command(value)
                self._insert_dict[key] = ""
            # list-based (SQLite) is handled by subclass

    def _add(
        self, t: int, roi: ROI, data_rows: Sequence[Mapping[str, DataPointProtocol]]
    ) -> None:
        """Buffer tracking rows as string INSERTs (base strategy)."""
        roi_id = roi.idx
        for dr in data_rows:
            # dr.values() are BaseIntVariable ints
            tp = (self._null, t, *tuple(dr.values()))
            if roi_id not in self._insert_dict or self._insert_dict[roi_id] == "":
                self._insert_dict[roi_id] = f"INSERT INTO ROI_{roi_id} VALUES {tp!s}"
            else:
                self._insert_dict[roi_id] += "," + str(tp)

        if self._dam_file_helper is not None:
            for dr in data_rows:
                self._dam_file_helper.input_roi_data(t, roi, dr)

    # -- DDL helpers ------------------------------------------------------

    def _initialise_var_map(
        self, data_row: Mapping[str, DataPointProtocol]
    ) -> None:
        """Create VAR_MAP entries."""
        self._write_async_command("DELETE FROM VAR_MAP")
        for dt in data_row.values():
            self._write_async_command(
                "INSERT INTO VAR_MAP VALUES (?, ?, ?)",
                (dt.header_name, dt.sql_data_type, dt.functional_type),
            )

    def _initialise_roi_table(
        self, roi: ROI, data_row: Mapping[str, DataPointProtocol]
    ) -> None:
        """Create ROI table - SQLite-compatible."""
        fields = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "t INTEGER"]
        fields.extend(
            f"{dt.header_name} {dt.sql_data_type}" for dt in data_row.values()
        )
        self._create_table(f"ROI_{roi.idx}", ", ".join(fields))

    # -- queue / resilience -----------------------------------------------

    def _write_async_command(self, command: str, args: SqlArgs = None) -> bool:
        """Send command with resilience."""
        return self._write_async_command_resilient(command, args)

    def _write_async_command_resilient(
        self, command: str, args: SqlArgs = None
    ) -> bool:
        """Retry loop with writer restart and buffering."""
        for attempt in range(MAX_DB_RETRIES + 1):
            try:
                if not self._async_writer.is_alive():
                    if attempt < MAX_DB_RETRIES:
                        self.log_io_diagnostics(
                            f"Writer died during attempt {attempt + 1}/{MAX_DB_RETRIES}"
                        )
                        _LOGGER.warning(
                            "Async writer died, attempting restart (attempt %s/%s)",
                            attempt + 1,
                            MAX_DB_RETRIES,
                        )
                        if self._restart_async_writer():
                            self._retry_buffered_commands()
                        continue
                    self.log_io_diagnostics(
                        "Writer permanently failed, entering degraded mode"
                    )
                    _LOGGER.error("Async writer permanently failed, buffering command")
                    return self._buffer_command(command, args)

                self._queue.put((command, args))
            except Exception as exc:
                if attempt < MAX_DB_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2**attempt), MAX_RETRY_DELAY)
                    _LOGGER.warning(
                        "Database write failed (attempt %s/%s): %s. Retrying in %.1fs",
                        attempt + 1,
                        MAX_DB_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    _LOGGER.exception("All database write attempts failed. Buffering.")
                    return self._buffer_command(command, args)
            else:
                return True
        return False

    def _restart_async_writer(self) -> bool:
        """Restart the async writer with throttling."""
        try:
            now = time.time()
            if now - self._last_restart_time < RESTART_THROTTLE_SECONDS:
                _LOGGER.warning("Async writer restart attempted too recently, skipping")
                return False

            if hasattr(self, "_async_writer") and self._async_writer is not None:
                try:
                    if self._async_writer.is_alive():
                        self._async_writer.terminate()
                        self._async_writer.join(timeout=5)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning("Error cleaning up old async writer: %s", exc)

            if hasattr(self, "_queue") and self._queue is not None:
                try:
                    self._queue.close()
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning("Error closing old queue: %s", exc)

            self._queue = multiprocessing.JoinableQueue()
            self._async_writer = self._create_async_writer(self._db_credentials, False)
            self._async_writer.start()

            if not self._async_writer._ready_event.wait(timeout=ASYNC_WRITER_TIMEOUT):
                _LOGGER.error("Restarted async writer failed to initialize")
                return False

            self._writer_restart_count += 1
            self._last_restart_time = now
            _LOGGER.info(
                "Successfully restarted async writer (restart #%s)",
                self._writer_restart_count,
            )
        except Exception:
            _LOGGER.exception("Failed to restart async writer")
            return False
        else:
            return True

    def _buffer_command(self, command: str, args: SqlArgs = None) -> bool:
        """Buffer a failed command."""
        return self._buffer.append(command, args)

    def _retry_buffered_commands(self) -> None:
        """Retry buffered commands FIFO, skipping old ones."""
        if not self._buffer:
            return

        _LOGGER.info("Retrying %s buffered database commands", len(self._buffer))
        failed_retries = 0
        while self._buffer:
            item = self._buffer.popleft()
            if item is None:
                break
            command, args, timestamp = item
            age = time.time() - timestamp
            if age > BUFFERED_COMMAND_MAX_AGE:
                _LOGGER.warning("Skipping old buffered command (age: %.1fs)", age)
                continue

            if self._async_writer.is_alive():
                try:
                    self._queue.put((command, args))
                except Exception as exc:
                    failed_retries += 1
                    _LOGGER.warning("Failed to retry buffered command: %s", exc)
                    if failed_retries > MAX_BUFFERED_RETRY_FAILURES:
                        _LOGGER.exception(
                            "Too many failures retrying buffered commands, stopping"
                        )
                        break
            else:
                self._buffer.appendleft((command, args, timestamp))
                _LOGGER.error(
                    "Async writer died again while retrying buffered commands"
                )
                break

        remaining = len(self._buffer)
        if remaining:
            _LOGGER.warning("%s commands remain buffered after retry", remaining)
        else:
            _LOGGER.info("All buffered commands successfully retried")

    def get_resilience_status(self) -> dict[str, Any]:
        """Current resilience status."""
        return {
            "writer_alive": self._async_writer.is_alive()
            if hasattr(self, "_async_writer")
            else False,
            "buffered_commands": len(self._buffer) if hasattr(self, "_buffer") else 0,
            "restart_count": self._writer_restart_count,
            "last_restart_time": self._last_restart_time,
            "time_since_last_restart": (
                time.time() - self._last_restart_time
                if self._last_restart_time > 0
                else None
            ),
        }

    def log_io_diagnostics(self, error_context: str = "") -> None:
        """Delegate to diagnostics reporter."""
        status = self.get_resilience_status()
        _DiagnosticsReporter.log(
            self._db_credentials, self._queue, status, error_context
        )

    @property
    def _failed_commands_buffer(self) -> deque[tuple[str, SqlArgs, float]]:
        """Expose buffer for backward compat with tests (deque)."""
        self._ensure_buffer()
        return self._buffer.as_deque()

    @_failed_commands_buffer.setter
    def _failed_commands_buffer(self, value: deque[tuple[str, SqlArgs, float]]) -> None:
        self._ensure_buffer()
        self._buffer.replace_deque(value)

    def _ensure_buffer(self) -> None:
        """Lazily create the buffer for test shells bypassing __init__."""
        if not hasattr(self, "_buffer") or self._buffer is None:
            self._buffer = _BufferedCommandQueue()

    # -- DDL execution ----------------------------------------------------

    def _create_table(self, name: str, fields: str, engine: Any | None = None) -> None:
        """Create table if not exists."""
        _ = engine  # kept for compat
        command = f"CREATE TABLE IF NOT EXISTS {name} ({fields})"
        _LOGGER.info("Creating database table with: %s", command)
        self._write_async_command(command)

    def _insert_metadata(self) -> None:
        """Insert metadata (SQLite)."""
        for key, value in list(self.metadata.items()):
            serialized: Any = (
                json.dumps(str(value))
                if not isinstance(value, (str, int, float, bool, type(None)))
                else value
            )
            if (
                isinstance(serialized, str)
                and len(serialized) > METADATA_MAX_VALUE_LENGTH
            ):
                serialized = serialized[:METADATA_MAX_VALUE_LENGTH] + "... [TRUNCATED]"
                _LOGGER.warning("Metadata value for key '%s' was truncated", key)
            self._write_async_command(
                "INSERT INTO METADATA VALUES (?, ?)", (key, serialized)
            )

    def _wait_for_queue_empty(self) -> None:
        """Block until queue drains."""
        while not self._queue.empty():
            _LOGGER.info("waiting for queue to be processed")
            time.sleep(QUEUE_CHECK_INTERVAL)

    # Compatibility shims for old tests that access _insert_dict directly
    # (no additional code needed - attribute already exists)


# ---------------------------------------------------------------------------
# dbAppender - composition wrapper
# ---------------------------------------------------------------------------


class dbAppender:
    """Append to an existing SQLite database.

    Wraps :class:`ethoscope.io.sqlite.SQLiteResultWriter` with ``erase_old_db=False``.
    """

    _description: Final[dict[str, Any]] = {
        "overview": "Database appender - appends to existing SQLite databases.",
        "arguments": [
            {
                "name": "database_to_append",
                "description": "Database to append to",
                "type": "str",
                "default": "",
                "asknode": "database_list",
            },
            {
                "name": "take_frame_shots",
                "description": "Save periodic frame snapshots",
                "type": "boolean",
                "default": True,
            },
            {
                "name": "make_dam_like_table",
                "description": "Create DAM-compatible activity summary table",
                "type": "boolean",
                "default": False,
            },
        ],
    }

    def __init__(  # noqa: PLR0913,PLR0917
        self,
        db_credentials: DbCredentials,
        rois: Sequence[ROI],
        metadata: MetadataDict | None = None,
        database_to_append: str = "",
        make_dam_like_table: bool = False,
        take_frame_shots: bool = False,
        sensor: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.database_to_append: str = database_to_append
        self.erase_old_db: bool = False
        self.db_credentials: DbCredentials = db_credentials
        self.rois: Sequence[ROI] = rois
        self.metadata: MetadataDict | None = metadata
        self.make_dam_like_table: bool = make_dam_like_table
        self.take_frame_shots: bool = take_frame_shots
        self.sensor: Any | None = sensor
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs

        if not self.database_to_append:
            raise MissingDatabaseNameError()

        _LOGGER.info("Detected SQLite database: %s", self.database_to_append)
        self._create_sqlite_writer()
        _LOGGER.info("We will be appending database: %s", database_to_append)

    def _create_sqlite_writer(self) -> None:
        """Create SQLite writer for append."""
        from .sqlite import SQLiteResultWriter  # noqa: PLC0415

        sqlite_creds = self.db_credentials.copy()
        sqlite_path = self._find_sqlite_database_path(self.database_to_append)
        if not sqlite_path:
            raise DatabaseFileNotFoundError(self.database_to_append)
        sqlite_creds["name"] = sqlite_path
        self.kwargs.update({"erase_old_db": False})
        self._writer = SQLiteResultWriter(
            sqlite_creds,
            self.rois,
            *self.args,
            metadata=self.metadata,
            make_dam_like_table=self.make_dam_like_table,
            take_frame_shots=self.take_frame_shots,
            sensor=self.sensor,
            **self.kwargs,
        )

    def _find_sqlite_database_path(self, database_name: str) -> str | None:
        """Locate SQLite file by full path or basename search."""
        db_path = Path(database_name)
        if db_path.exists():
            _LOGGER.info("Found SQLite database at: %s", database_name)
            return database_name

        db_basename = db_path.name
        if not db_basename.endswith(".db"):
            db_basename += ".db"

        search_roots = [Path("/ethoscope_data/results"), Path("/data")]
        for search_root in search_roots:
            if not search_root.exists():
                continue
            try:
                for candidate in search_root.rglob(db_basename):
                    if candidate.is_file():
                        _LOGGER.info("Found SQLite database at: %s", candidate)
                        return str(candidate)
            except OSError as exc:
                _LOGGER.warning("Error walking directory %s: %s", search_root, exc)
                continue

        _LOGGER.warning("Could not find SQLite database: %s", database_name)
        return None

    @classmethod
    def get_available_databases(
        cls, db_credentials: DbCredentials, device_name: str = ""
    ) -> list[dict[str, Any]]:
        """List available databases for UI dropdown."""
        databases_list: list[dict[str, Any]] = []
        try:
            from .cache import get_all_databases_info  # noqa: PLC0415

            if not device_name and "name" in db_credentials:
                device_name = db_credentials["name"]
                if isinstance(device_name, str) and not device_name.startswith(
                    "ETHOSCOPE_"
                ):
                    m = re.search(r"([a-f0-9]{32})", device_name)
                    if m:
                        device_name = f"ETHOSCOPE_{m.group(1)[:8].upper()}"

            info = get_all_databases_info(device_name)
            for db_name, db_info in info.get("SQLite", {}).items():
                if (
                    db_info.get("file_exists", False)
                    and db_info.get("filesize", 0) > MIN_DB_SIZE_BYTES
                ):
                    databases_list.append(
                        {
                            "name": db_name,
                            "type": "SQLite",
                            "active": True,
                            "size": db_info.get("filesize", 0),
                            "status": db_info.get("db_status", "unknown"),
                            "path": db_info.get("path", ""),
                        }
                    )
        except Exception:
            _LOGGER.exception("Error getting available databases")
        return databases_list

    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped writer (dunder-safe to avoid recursion)."""
        if name.startswith("__") or name == "_writer":
            raise AttributeError(name)
        return getattr(self._writer, name)

    def __enter__(self) -> Any:
        """Delegate context entry."""
        return self._writer.__enter__()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Any:
        """Delegate context exit."""
        return self._writer.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[arg-type]
