"""SQLite-specific writers - async process and batched result writer."""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ._constants import METADATA_MAX_VALUE_LENGTH, SQLITE_BATCH_SIZE
from ._sql import map_sql_data_type_to_sqlite
from .base import BaseAsyncSQLWriter, BaseResultWriter
from .helpers import Null

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from multiprocessing import JoinableQueue

    from ethoscope.core.roi import ROI

    from ._types import DataPointProtocol, DbCredentials, SqlArgs

_LOGGER: Final = logging.getLogger(__name__)


class SQLiteConnectionError(RuntimeError):
    """Raised when SQLite connection fails."""

    def __init__(self, db_name: str, exc: Exception) -> None:
        super().__init__(f"Failed to connect to SQLite database {db_name}: {exc}")
        self.db_name = db_name
        self.cause = exc


# ---------------------------------------------------------------------------
# Async SQLite writer
# ---------------------------------------------------------------------------


class AsyncSQLiteWriter(BaseAsyncSQLWriter):
    """Async SQLite writer with WAL pragmas."""

    _database_type: Final[str] = "SQLite3"
    _pragmas: Final[dict[str, str]] = {
        "temp_store": "MEMORY",
        "journal_mode": "WAL",
        "locking_mode": "NORMAL",
        "busy_timeout": "30000",
        "synchronous": "NORMAL",
    }

    def __init__(
        self, db_name: str | Path, queue: JoinableQueue[Any], erase_old_db: bool = True
    ) -> None:
        super().__init__(queue, erase_old_db)
        self._db_name: str = str(db_name)

    def _get_connection(self) -> sqlite3.Connection:
        """Create SQLite connection with timeout."""
        try:
            return sqlite3.connect(self._db_name, timeout=30.0)
        except sqlite3.Error as exc:
            raise SQLiteConnectionError(self._db_name, exc) from exc

    def _initialize_database(self) -> None:
        """Delete old file and apply pragmas if ``erase_old_db``."""
        if not self._erase_old_db:
            return

        try:
            Path(self._db_name).unlink(missing_ok=True)
        except OSError:
            _LOGGER.debug(
                "Could not remove old DB file %s", self._db_name, exc_info=True
            )

        db_dir = Path(self._db_name).parent
        if str(db_dir) not in ("", "."):
            db_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info("Created SQLite directory: %s", db_dir)

        # Apply pragmas in a single connection
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            _LOGGER.info("Setting DB parameters")
            for key, value in self._pragmas.items():
                cursor.execute(f"PRAGMA {key} = {value}")
            conn.commit()
        finally:
            conn.close()

    def _get_db_type_name(self) -> str:
        return "SQLite"

    def _should_retry_on_error(self, error: Exception) -> bool:
        """Retry only on transient lock/busy errors."""
        if isinstance(error, sqlite3.OperationalError):
            msg = str(error).lower()
            if any(kw in msg for kw in ("locked", "busy", "cannot commit")):
                _LOGGER.warning("SQLite transient error, will retry: %s", error)
                return True
        _LOGGER.error("SQLite critical error, stopping writer: %s", error)
        return False


# ---------------------------------------------------------------------------
# SQLite result writer - composition over inheritance for insert strategy
# ---------------------------------------------------------------------------


class SQLiteResultWriter(BaseResultWriter):
    """SQLite result writer with parameterized batch inserts."""

    _description: Final[dict[str, Any]] = {
        "overview": (
            "SQLite result writer - stores tracking data to local SQLite database "
            "file using consistent directory structure. Each experiment creates a "
            "unique file, preserving historical data. Compatible with rsync-based "
            "backups. Supports sensor data collection when sensors are available."
        ),
        "arguments": [
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
                "default": True,
            },
        ],
    }

    _database_type: ClassVar[str] = "SQLite3"
    _async_writing_class: ClassVar[type[AsyncSQLiteWriter]] = AsyncSQLiteWriter
    _null: ClassVar[Null] = Null()

    def __init__(  # noqa: PLR0913,PLR0917
        self,
        db_credentials: DbCredentials,
        rois: Sequence[ROI],
        metadata: dict[str, Any] | None = None,
        make_dam_like_table: bool = False,
        take_frame_shots: bool = False,
        erase_old_db: bool = True,
        sensor: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("erase_old_db", None)
        super().__init__(
            db_credentials,
            rois,
            metadata,
            make_dam_like_table,
            take_frame_shots,
            erase_old_db,
            sensor,
            *args,
            **kwargs,
        )

    # -- timestamp -------------------------------------------------------

    def _is_db_file_ready(self, db_path: Path) -> bool:
        """Check that DB file exists and is non-empty."""
        try:
            if not db_path.exists():
                _LOGGER.error("SQLite database file does not exist: %s", db_path)
                return False
            if db_path.stat().st_size == 0:
                _LOGGER.error("SQLite database file is empty: %s", db_path)
                return False
        except OSError:
            _LOGGER.exception("Cannot access SQLite database file %s", db_path)
            return False
        return True

    def _get_existing_roi_tables(self, cursor: sqlite3.Cursor) -> set[str]:
        """Return set of ROI table names."""
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ROI_%'"
        )
        return {row[0] for row in cursor.fetchall()}

    def _query_single_table_max(
        self, cursor: sqlite3.Cursor, table: str
    ) -> int | None:
        """Return MAX(t) for a single table or None if not available."""
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cursor.fetchall()]
            cursor.execute(f"SELECT MAX(t) FROM {table} WHERE t IS NOT NULL")
            row = cursor.fetchone()
        except sqlite3.Error:
            _LOGGER.exception("Error querying table %s", table)
            return None

        if "t" not in cols:
            _LOGGER.error("Table %s missing required 't' column", table)
            return None
        if row and row[0] is not None:
            ts = int(row[0])
            _LOGGER.debug("Table %s max timestamp: %s", table, ts)
            return ts
        _LOGGER.info("Table %s has no data or null timestamps", table)
        return None

    def _query_max_timestamp(self, db_path: Path) -> int:
        """Query DB for max timestamp across ROIs."""
        with closing(sqlite3.connect(str(db_path), timeout=30.0)) as db:
            cursor = db.cursor()
            existing = self._get_existing_roi_tables(cursor)
            if not existing:
                _LOGGER.warning("No ROI tables found in SQLite database: %s", db_path)
                return 0

            last_ts = 0
            ok = 0
            for roi in self._rois:
                table = f"ROI_{roi.idx}"
                if table not in existing:
                    _LOGGER.warning(
                        "ROI table %s not found in database, skipping",
                        table,
                    )
                    continue
                ts = self._query_single_table_max(cursor, table)
                if ts is not None:
                    last_ts = max(last_ts, ts)
                    ok += 1

            if ok == 0:
                _LOGGER.warning("No ROI tables could be successfully queried")
                return 0
            _LOGGER.info(
                "Successfully retrieved last timestamp %s from %s ROI table(s)",
                last_ts,
                ok,
            )
            return last_ts

    def get_last_timestamp(self) -> int:
        """Return max ``t`` across ROI tables, or 0 on failure."""
        db_path = Path(self._db_credentials["name"])
        if not self._is_db_file_ready(db_path):
            return 0
        try:
            return self._query_max_timestamp(db_path)
        except sqlite3.DatabaseError:
            _LOGGER.exception("SQLite database error accessing %s", db_path)
            return 0
        except sqlite3.Error:
            _LOGGER.exception("SQLite error getting last timestamp from %s", db_path)
            return 0
        except Exception:
            _LOGGER.exception(
                "Unexpected error getting last timestamp from SQLite %s",
                db_path,
            )
            return 0

    # -- factory ---------------------------------------------------------

    def _create_async_writer(
        self, db_credentials: DbCredentials, erase_old_db: bool, **kwargs: Any
    ) -> BaseAsyncSQLWriter:
        _ = kwargs
        return self._async_writing_class(
            db_credentials["name"], self._queue, erase_old_db
        )

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_pickle_extra_kwargs"] = {}
        return state

    # -- SQL dialect -----------------------------------------------------

    def _write_async_command(self, command: str, args: SqlArgs = None) -> bool:
        """Convert MySQL placeholders and Null, then delegate."""
        sqlite_cmd = command.replace("%s", "?") if "%s" in command else command
        if "%s" in command:
            _LOGGER.debug(
                "Converting MySQL command to SQLite: %s -> %s", command, sqlite_cmd
            )

        if args is not None:
            args = tuple(None if isinstance(a, Null) else a for a in args)

        return self._write_async_command_resilient(sqlite_cmd, args)

    def _create_table(self, name: str, fields: str, engine: Any | None = None) -> None:
        _ = engine
        command = f"CREATE TABLE IF NOT EXISTS {name} ({fields})"
        _LOGGER.info("Creating database table with: %s", command)
        self._write_async_command(command)

    def _initialise_roi_table(
        self, roi: ROI, data_row: Mapping[str, DataPointProtocol]
    ) -> None:
        """Create ROI table with SQLite-mapped types via match."""
        fields = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "t INTEGER"]
        for dt in data_row.values():
            sqlite_type = map_sql_data_type_to_sqlite(str(dt.sql_data_type))
            fields.append(f"{dt.header_name} {sqlite_type}")
        self._create_table(f"ROI_{roi.idx}", ", ".join(fields))

    # -- insert strategy -------------------------------------------------

    def _add(
        self, t: int, roi: ROI, data_rows: Sequence[Mapping[str, DataPointProtocol]]
    ) -> None:
        """Buffer rows as tuples for parameterized batch INSERTs."""
        roi_id = roi.idx
        if roi_id not in self._insert_dict:
            self._insert_dict[roi_id] = []

        for dr in data_rows:
            values: list[Any] = [None, t, *list(dr.values())]
            sqlite_vals: list[Any] = []
            for val in values:
                if val is None or isinstance(val, Null):
                    sqlite_vals.append(None)
                elif isinstance(val, bool):
                    sqlite_vals.append(1 if val else 0)
                else:
                    sqlite_vals.append(val)
            self._insert_dict[roi_id].append(tuple(sqlite_vals))

        if self._dam_file_helper is not None:
            for dr in data_rows:
                self._dam_file_helper.input_roi_data(t, roi, dr)

    def _batch_insert_roi(
        self, roi_id: int, value_list: Sequence[tuple[Any, ...]]
    ) -> None:
        """Insert ``value_list`` in batches of ``SQLITE_BATCH_SIZE``."""
        if not value_list:
            return
        n_cols = len(value_list[0])
        single = "(" + ", ".join(["?"] * n_cols) + ")"
        batch_size = SQLITE_BATCH_SIZE
        full_placeholders = ", ".join([single] * batch_size)
        full_cmd = f"INSERT INTO ROI_{roi_id} VALUES {full_placeholders}"

        for i in range(0, len(value_list), batch_size):
            batch = value_list[i : i + batch_size]
            n = len(batch)
            cmd = (
                full_cmd
                if n == batch_size
                else f"INSERT INTO ROI_{roi_id} VALUES {', '.join([single] * n)}"
            )
            flat: tuple[Any, ...] = tuple(itertools.chain.from_iterable(batch))
            self._write_async_command(cmd, flat)

    def flush(self, t: int, img: Any | None = None) -> bool:
        """Flush helpers and ROI batches."""
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

        for roi_id, value_list in list(self._insert_dict.items()):
            if len(value_list) >= self._max_insert_string_len and value_list:
                self._batch_insert_roi(roi_id, value_list)
                self._insert_dict[roi_id] = []
        return False

    def close(self) -> None:
        """Flush remaining batches."""
        for roi_id, value_list in list(self._insert_dict.items()):
            if value_list:
                self._batch_insert_roi(roi_id, value_list)
                self._insert_dict[roi_id] = []
        super().close()

    # -- DDL -------------------------------------------------------------

    def _create_all_tables(self) -> None:
        """Create master tables - SQLite only."""
        if self._erase_old_db:
            _LOGGER.info("Creating master table 'ROI_MAP'")
            self._create_table(
                "ROI_MAP",
                "roi_idx INTEGER, roi_value INTEGER, "
                "x INTEGER, y INTEGER, w INTEGER, h INTEGER",
            )
            for roi in self._rois:
                fd = roi.get_feature_dict()
                self._write_async_command(
                    "INSERT INTO ROI_MAP VALUES (?, ?, ?, ?, ?, ?)",
                    (fd["idx"], fd["value"], fd["x"], fd["y"], fd["w"], fd["h"]),
                )

            _LOGGER.info("Creating variable map table 'VAR_MAP'")
            self._create_table(
                "VAR_MAP", "var_name TEXT, sql_type TEXT, functional_type TEXT"
            )

            if self._shot_saver is not None:
                _LOGGER.info("Creating table for IMG_SNAPSHOTS")
                self._create_table(
                    "IMG_SNAPSHOTS",
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER, img BLOB",
                )

            if self._sensor_saver is not None:
                _LOGGER.info("Creating table for SENSORS data")
                self._create_table(
                    self._sensor_saver.table_name, self._sensor_saver.create_command
                )

            if self._dam_file_helper is not None:
                _LOGGER.info("Creating 'CSV_DAM_ACTIVITY' table")
                fields = self._dam_file_helper.make_dam_file_sql_fields()
                fields = fields.replace(
                    "INT  NOT NULL AUTO_INCREMENT PRIMARY KEY",
                    "INTEGER PRIMARY KEY AUTOINCREMENT",
                )
                fields = fields.replace("CHAR(100)", "TEXT").replace(
                    "SMALLINT", "INTEGER"
                )
                self._create_table("CSV_DAM_ACTIVITY", fields)

            _LOGGER.info("Creating 'METADATA' table")
            self._create_table("METADATA", "field TEXT, value TEXT")

            _LOGGER.info("Creating 'START_EVENTS' table")
            self._create_table(
                "START_EVENTS",
                "id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER, event TEXT",
            )
            self._write_async_command(
                "INSERT INTO START_EVENTS VALUES (?, ?, ?)",
                (None, int(time.time()), "graceful_start"),
            )

            self._insert_metadata()
            self._wait_for_queue_empty()

        elif not self._erase_old_db and getattr(self, "database_to_append", None):
            self._write_async_command(
                "INSERT INTO START_EVENTS VALUES (?, ?, ?)",
                (None, int(time.time()), "appending"),
            )
            self._wait_for_queue_empty()

    def _insert_metadata(self) -> None:
        """Insert metadata with ``INSERT OR IGNORE``."""
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
                "INSERT OR IGNORE INTO METADATA VALUES (?, ?)", (key, serialized)
            )
