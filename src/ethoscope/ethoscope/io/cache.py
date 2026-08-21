"""Database metadata caching for ethoscope experiments.

Provides :class:`BaseDatabaseMetadataCache` (abstract), the SQLite
implementation, and :class:`DatabasesInfo` for fleet-level database listing.
Cache files are JSON documents named ``db_metadata_<ts>_<device>_db.json``.
"""

from __future__ import annotations

import ast
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ._constants import CACHE_TTL_SECONDS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ._types import DbCredentials, ExperimentInfo

_LOGGER: Final = logging.getLogger(__name__)

DEFAULT_CACHE_DIR: Final[str] = "/ethoscope_data/cache"
SECONDS_PER_DAY: Final[int] = 24 * 60 * 60
MIN_DATE_PARTS: Final[int] = 3
MAX_CACHE_FILE_BYTES: Final[int] = 10 * 1024 * 1024
SQLITE_WRITER_TYPES: Final[frozenset[str]] = frozenset(
    {"SQLiteResultWriter", "SQLite3"}
)
EMPTY_CACHE_INFO: Final[dict[str, Any]] = {
    "db_size_bytes": 0,
    "table_counts": {},
    "last_db_update": 0,
}


class DeviceNameLookupError(ValueError):
    """Raised when the device name cannot be determined."""

    def __init__(self) -> None:
        super().__init__(
            "Could not determine device_name from database or arguments"
        )


class TimestampNotFoundError(ValueError):
    """Raised when the experiment timestamp cannot be determined."""

    def __init__(self) -> None:
        super().__init__("Could not determine timestamp from database")


class BaseDatabaseMetadataCache:
    """Abstract cache of per-experiment database metadata.

    Subclasses implement :meth:`_query_database` and
    :meth:`_get_value_from_database` for their backend.

    Args:
        db_credentials: Database connection credentials.
        device_name: Device name used for cache file naming; auto-detected
            from the database when empty.
        cache_dir: Directory for cache files (created if missing).
    """

    def __init__(
        self,
        db_credentials: DbCredentials,
        device_name: str = "",
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> None:
        self.db_credentials: DbCredentials = db_credentials
        self.cache_dir: str = cache_dir
        # Track current active cache file
        self.current_cache_file_path: str | None = None

        self.allowed_metadata_fields: list[str] = [
            "backup_filename",
            "experimental_info",
            "date_time",
            "machine_name",
            "machine_id",
            "stop_date_time",
        ]

        self.device_name: str | None = device_name or self.get_device_name()
        if not self.device_name:
            raise DeviceNameLookupError

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metadata(self, tracking_start_time: float | None = None) -> dict[str, Any]:
        """Return fresh database metadata, writing it to the cache.

        Falls back to cached data when the database cannot be queried.
        """
        cache_file_path = self._get_cache_file_path(tracking_start_time)
        if not cache_file_path and self.current_cache_file_path:
            cache_file_path = self.current_cache_file_path

        try:
            db_info = self._query_database()
        except Exception:  # hook is subclass-controlled; fall back to cache
            _LOGGER.warning("Failed to query database", exc_info=True)
            return self._read_cache(cache_file_path)
        else:
            if cache_file_path:
                self._write_cache(cache_file_path, db_info, tracking_start_time)
            return db_info

    def finalize_cache(
        self,
        tracking_start_time: float | None,
        graceful: bool = True,
        stop_reason: str = "user_stop",
    ) -> None:
        """Mark the cache file as finalised when an experiment ends."""
        cache_file_path = self._get_cache_file_path(tracking_start_time)
        if cache_file_path:
            self._write_cache(
                cache_file_path,
                finalise=True,
                graceful=graceful,
                stop_reason=stop_reason,
            )
        self.current_cache_file_path = None

    def get_cached_metadata(self, cache_index: int = 0) -> dict[str, Any]:
        """Read metadata from cached JSON files without querying the database.

        Args:
            cache_index: 0 = most recent cache file, 1 = previous, etc.
        """
        return self._read_cache(None, cache_index=cache_index)

    def list_cache_files(self) -> list[dict[str, Any]]:
        """List cache files for this device, newest first."""
        file_info: list[dict[str, Any]] = []

        for i, cache_path in enumerate(self._get_all_cache_files()):
            try:
                mtime = Path(cache_path).stat().st_mtime
                age_days = (time.time() - mtime) / SECONDS_PER_DAY
                parts = Path(cache_path).name.split("_")[2:5]
                experiment_date = (
                    "_".join(parts)
                    if len(parts) >= MIN_DATE_PARTS
                    else "unknown"
                )
            except OSError as exc:
                _LOGGER.warning(
                    "Failed to get info for cache file %s: %s", cache_path, exc
                )
                continue
            file_info.append(
                {
                    "index": i,
                    "path": cache_path,
                    "filename": Path(cache_path).name,
                    "experiment_date": experiment_date,
                    "modified_time": mtime,
                    "age_days": round(age_days, 1),
                }
            )

        return file_info

    def get_cache_summary(self) -> dict[str, Any]:
        """Summarise all cache files for this device."""
        files = self.list_cache_files()

        if not files:
            return {
                "total_files": 0,
                "newest_date": None,
                "oldest_date": None,
                "files": [],
            }

        newest_time = files[0]["modified_time"]
        oldest_time = files[-1]["modified_time"]

        return {
            "total_files": len(files),
            "newest_date": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(newest_time)
            ),
            "oldest_date": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(oldest_time)
            ),
            "files": files,
        }

    def create_experiment_info_from_metadata(
        self,
        timestamp: float,
        backup_filename: str,
        result_writer_type: str,
        sqlite_source_path: str | None = None,
    ) -> ExperimentInfo:
        """Build an experiment-info dict from database metadata."""
        experimental_metadata = self.get_experimental_metadata()

        experiment_info: ExperimentInfo = {
            "date_time": timestamp,
            "backup_filename": backup_filename,
            "user": experimental_metadata.get("user", "unknown"),
            "location": experimental_metadata.get("location", "unknown"),
            "result_writer_type": result_writer_type,
            "run_id": experimental_metadata.get("run_id", "unknown"),
        }
        if sqlite_source_path:
            experiment_info["sqlite_source_path"] = sqlite_source_path
        return experiment_info

    def get_experimental_metadata(self) -> dict[str, Any]:
        """Extract user/location/run info from the METADATA table."""
        raw = self._get_value_from_database("experimental_info")
        try:
            parsed = ast.literal_eval(raw) if raw else None
        except (ValueError, SyntaxError, TypeError) as exc:
            _LOGGER.warning("Could not parse experimental_info: %s", exc)
            return {}
        if not isinstance(parsed, dict):
            _LOGGER.warning("experimental_info is not a mapping: %r", parsed)
            return {}
        return {
            "user": parsed.get("name", "unknown"),
            "location": parsed.get("location", "unknown"),
            "code": parsed.get("code", ""),
            "run_id": parsed.get("run_id", ""),
        }

    def get_database_timestamp(self) -> float | None:
        """Return the experiment timestamp from METADATA, or None."""
        timestamp_str = self._get_value_from_database("date_time")
        if timestamp_str is None:
            return None
        try:
            return float(timestamp_str)
        except (ValueError, TypeError):
            _LOGGER.warning(
                "Could not convert timestamp '%s' to float.", timestamp_str
            )
            return None

    def get_device_name(self) -> str | None:
        """Return the device name stored in METADATA."""
        return self._get_value_from_database("machine_name")

    def refresh_cache_from_database(
        self,
        device_name: str | None = None,  # unused; kept for API compat
        timestamp: float | None = None,
        backup_filename: str | None = None,
        result_writer_type: str | None = None,
        sqlite_source_path: str | None = None,
    ) -> str | None:
        """Rebuild the cache entry for this experiment from the database."""
        if timestamp is None:
            timestamp = self.get_database_timestamp()
            if timestamp is None:
                raise TimestampNotFoundError

        if backup_filename is None:
            ts_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
            if sqlite_source_path:
                backup_filename = Path(sqlite_source_path).name
            else:
                backup_filename = f"{ts_str}_{self.device_name}.db"

        if result_writer_type is None:
            result_writer_type = "SQLiteResultWriter"

        experiment_info = self.create_experiment_info_from_metadata(
            timestamp, backup_filename, result_writer_type, sqlite_source_path
        )
        self.store_experiment_info(timestamp, experiment_info)

        try:
            cache_file_path = self._get_cache_file_path(timestamp)
            if cache_file_path:
                db_info = self._query_database()
                self._write_cache(cache_file_path, db_info, timestamp)
                _LOGGER.info(
                    "Updated cache with database metadata: %s bytes, %s tables",
                    db_info.get("db_size_bytes", 0),
                    len(db_info.get("table_counts", {})),
                )
        except Exception:  # hook is subclass-controlled
            _LOGGER.warning(
                "Failed to update cache with database metadata", exc_info=True
            )

        return self._get_cache_file_path(timestamp)

    def store_experiment_info(
        self, tracking_start_time: float, experiment_info: ExperimentInfo
    ) -> None:
        """Persist experiment information into the cache file."""
        cache_file_path = self._get_cache_file_path(tracking_start_time)
        if not cache_file_path:
            return
        self.current_cache_file_path = cache_file_path
        formatted_info: ExperimentInfo = {
            "date_time": experiment_info.get("date_time"),
            "backup_filename": experiment_info.get("backup_filename"),
            "user": experiment_info.get("user"),
            "location": experiment_info.get("location"),
            "result_writer_type": experiment_info.get("result_writer_type"),
            "sqlite_source_path": experiment_info.get("sqlite_source_path"),
            "run_id": experiment_info.get("run_id"),
            "stored_timestamp": time.time(),
        }
        self._write_cache(cache_file_path, experiment_info=formatted_info)
        _LOGGER.info("Stored experiment info for %s in cache", self.device_name)

    def get_last_experiment_info(self) -> dict[str, Any]:
        """Info about the most recent experiment; empty dict if unavailable."""
        try:
            recent = self.get_cached_metadata(cache_index=0)
            payload = self._payload_if_present(recent)
            if payload is not None:
                return payload
            for cache_index in range(1, 5):  # up to 5 previous experiments
                try:
                    data = self.get_cached_metadata(cache_index=cache_index)
                except (OSError, json.JSONDecodeError):
                    continue
                payload = self._payload_if_present(data)
                if payload is not None:
                    return payload
        except Exception:  # best-effort cache introspection
            _LOGGER.warning(
                "Failed to get last experiment info from cache", exc_info=True
            )
        return {}

    def has_last_experiment_info(self) -> bool:
        """Whether information about the last experiment is available."""
        last_info = self.get_last_experiment_info()
        previous = last_info.get("experimental_info", {}).get("previous", {})
        return bool(previous.get("backup_filename"))

    def get_experiment_history(self, max_experiments: int = 10) -> list[dict[str, Any]]:
        """History of previous experiments, newest first."""
        experiments: list[dict[str, Any]] = []

        for cache_index in range(max_experiments):
            try:
                data = self.get_cached_metadata(cache_index=cache_index)
                cache_file_path = data.get("cache_file")
                if not cache_file_path or not Path(str(cache_file_path)).exists():
                    break  # No more cache files
                with Path(str(cache_file_path)).open(encoding="utf-8") as f:
                    cache_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                break  # Error reading cache file

            experiment_info = cache_data.get("experiment_info", {})
            if experiment_info:
                experiments.append(
                    {
                        "index": cache_index,
                        "date_time": experiment_info.get("date_time"),
                        "backup_filename": experiment_info.get("backup_filename"),
                        "user": experiment_info.get("user"),
                        "location": experiment_info.get("location"),
                        "result_writer_type": experiment_info.get(
                            "result_writer_type"
                        ),
                        "db_size_bytes": data.get("db_size_bytes", 0),
                        "table_counts": data.get("table_counts", {}),
                        "db_status": data.get("db_status", "unknown"),
                        "cache_file": cache_file_path,
                    }
                )

        return experiments

    def get_database_info(self) -> dict[str, Any]:
        """Structured database info; falls back to cache then error state."""
        try:
            db_info = self._query_database()
        except Exception:  # hook is subclass-controlled
            _LOGGER.warning("Failed to query database directly", exc_info=True)
            return self._cached_db_info_or_error()
        else:
            db_info.setdefault(
                "db_name", self.db_credentials.get("name", "unknown")
            )
            db_info.setdefault("db_status", "active")
            return db_info

    def get_backup_filename(self) -> str | None:
        """Backup filename recorded in the METADATA table."""
        return self._get_value_from_database("backup_filename")

    # ------------------------------------------------------------------
    # Cache-file primitives
    # ------------------------------------------------------------------

    def _payload_if_present(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Load experiment payload from a cache record, if it has one."""
        cache_file_path = data.get("cache_file")
        if not cache_file_path or not Path(str(cache_file_path)).exists():
            return None
        with Path(str(cache_file_path)).open(encoding="utf-8") as f:
            cache_data = json.load(f)
        experiment_info = cache_data.get("experiment_info", {})
        if not experiment_info:
            return None
        return {
            "experimental_info": {
                "current": {},
                "previous": {
                    "date_time": experiment_info.get("date_time"),
                    "backup_filename": experiment_info.get("backup_filename"),
                    "user": experiment_info.get("user"),
                    "location": experiment_info.get("location"),
                },
            },
            "result_writer_type": experiment_info.get("result_writer_type"),
            "sqlite_source_path": experiment_info.get("sqlite_source_path"),
            "cache_file": cache_file_path,
        }

    def _cached_db_info_or_error(self) -> dict[str, Any]:
        """Most recent cached DB info, or an error-state dict."""
        try:
            cached = self.get_cached_metadata(cache_index=0)
            if cached.get("db_size_bytes", 0) > 0 or cached.get("table_counts"):
                cached.setdefault(
                    "db_name", self.db_credentials.get("name", "unknown")
                )
                cached.setdefault("db_status", "cached")
                _LOGGER.info(
                    "Using cached database info for %s", self.device_name
                )
                return cached
            _LOGGER.warning("Cached data is empty for %s", self.device_name)
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Failed to read cached database info: %s", exc)

        return {
            "db_name": self.db_credentials.get("name", "unknown"),
            "db_size_bytes": 0,
            "table_counts": {},
            "last_db_update": 0,
            "db_status": "error",
            "db_version": "Unknown",
        }

    def _existing_cache_data(self, cache_file_path: str) -> dict[str, Any] | None:
        """Read an existing cache file, or None if absent."""
        path = Path(cache_file_path)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    @staticmethod
    def _dump_cache(cache_file_path: str, cache_data: dict[str, Any]) -> None:
        """Atomically-enough write of the cache JSON document."""
        with Path(cache_file_path).open("w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    def _get_cache_file_path(self, tracking_start_time: float | None) -> str | None:
        """Cache file path for an experiment start time (None if unknown)."""
        if not self.device_name or not tracking_start_time:
            # Avoid stale timestamps from previous experiments
            return None
        rounded = int(tracking_start_time)
        ts_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(rounded))
        filename = f"db_metadata_{ts_str}_{self.device_name}_db.json"
        return str(Path(self.cache_dir) / filename)

    def _read_cache(
        self, cache_file_path: str | None, cache_index: int | None = None
    ) -> dict[str, Any]:
        """Read a specific cache file, or auto-find by index."""
        if cache_file_path and Path(cache_file_path).exists():
            try:
                return self._read_cache_file(cache_file_path)
            except (OSError, json.JSONDecodeError) as exc:
                _LOGGER.warning(
                    "Failed to read cache file %s: %s", cache_file_path, exc
                )

        try:
            cache_files = self._get_all_cache_files()
            if cache_files:
                selected = self._select_cache_file(cache_files, cache_index)
                _LOGGER.info(
                    "Reading cache file %s: %s",
                    cache_index or 0,
                    Path(selected).name,
                )
                return self._read_cache_file(selected)
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning(
                "Failed to find cache files for %s: %s", self.device_name, exc
            )

        return dict(EMPTY_CACHE_INFO)

    @staticmethod
    def _select_cache_file(
        cache_files: list[str], cache_index: int | None
    ) -> str:
        """Pick the cache file for an index, falling back to most recent."""
        if cache_index is None or cache_index == 0:
            return cache_files[0]
        if cache_index < len(cache_files):
            return cache_files[cache_index]
        _LOGGER.warning(
            "Cache index %s out of range (max: %s), using most recent",
            cache_index,
            len(cache_files) - 1,
        )
        return cache_files[0]

    def _get_all_cache_files(self) -> list[str]:
        """All cache files for this device, sorted newest-first by mtime."""
        try:
            pattern = f"db_metadata_*_{self.device_name}_db.json"
            files = [str(p) for p in Path(self.cache_dir).glob(pattern)]
            files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        except OSError as exc:
            _LOGGER.warning("Failed to list cache files: %s", exc)
            return []
        return files

    def _read_cache_file(self, cache_file_path: str) -> dict[str, Any]:
        """Parse one cache file into its normalised representation."""
        with Path(cache_file_path).open(encoding="utf-8") as f:
            cache_data = json.load(f)

        return {
            "db_size_bytes": cache_data.get("db_size_bytes", 0),
            "table_counts": cache_data.get("table_counts", {}),
            "last_db_update": cache_data.get("last_db_update", 0),
            "cache_file": cache_file_path,
            "db_status": cache_data.get("db_status", "unknown"),
            "db_version": cache_data.get("db_version", "Unknown"),
            "creation_timestamp": cache_data.get("creation_timestamp"),
            "tracking_start_time": cache_data.get("tracking_start_time"),
            "finalized_timestamp": cache_data.get("finalized_timestamp"),
            "stopped_gracefully": cache_data.get("stopped_gracefully", False),
            "stop_reason": cache_data.get("stop_reason", "unknown"),
            "stop_timestamp": cache_data.get("stop_timestamp"),
            "experiment_info": cache_data.get("experiment_info", {}),
        }

    def _write_cache(  # noqa: PLR0913,PLR0917  # positional compat: unit tests drive this private API
        self,
        cache_file_path: str,
        db_info: Mapping[str, Any] | None = None,
        tracking_start_time: float | None = None,
        finalise: bool = False,
        experiment_info: ExperimentInfo | None = None,
        graceful: bool = True,
        stop_reason: str = "user_stop",
    ) -> None:
        """Create or update a cache file.

        Kept positional-compatible for unit tests that drive this private
        API directly.
        """
        try:
            cache_data = self._existing_cache_data(cache_file_path)
            if cache_data is None:
                if finalise and not experiment_info:
                    _LOGGER.warning(
                        "Cannot finalize non-existent cache file: %s",
                        cache_file_path,
                    )
                    return
                ts = tracking_start_time or time.time()
                cache_data = {
                    "db_name": self.db_credentials["name"],
                    "device_name": self.device_name,
                    "tracking_start_time": time.strftime(
                        "%Y-%m-%d_%H-%M-%S", time.localtime(ts)
                    ),
                    "creation_timestamp": ts,
                }

            if db_info:
                cache_data.update(
                    {
                        "last_updated": time.time(),
                        "db_size_bytes": db_info["db_size_bytes"],
                        "table_counts": db_info["table_counts"],
                        "last_db_update": db_info["last_db_update"],
                        "db_version": db_info["db_version"],
                        "db_status": "tracking",
                    }
                )

            stop_timestamp = self._get_value_from_database("stop_date_time")
            if finalise or stop_timestamp:
                cache_data.update(
                    {
                        "db_status": "finalised" if finalise else "terminated",
                        "finalized_timestamp": (
                            time.time() if finalise else "unknown"
                        ),
                        "stopped_gracefully": graceful or finalise or "unknown",
                        "stop_reason": stop_reason or "unknown",
                        "stop_timestamp": stop_timestamp or "unknown",
                    }
                )
                _LOGGER.info(
                    "Finalizing cache with graceful=%s, reason=%s",
                    graceful,
                    stop_reason,
                )

            if experiment_info:
                cache_data["experiment_info"] = experiment_info
                if "db_size_bytes" not in cache_data:
                    cache_data.update(
                        {
                            "db_size_bytes": 0,
                            "table_counts": {},
                            "last_db_update": time.time(),
                            "db_version": "Unknown",
                        }
                    )

            self._dump_cache(cache_file_path, cache_data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            action = "finalize" if finalise else "update"
            _LOGGER.warning(
                "Failed to %s cache file %s: %s", action, cache_file_path, exc
            )

    # ------------------------------------------------------------------
    # Backend hooks (abstract)
    # ------------------------------------------------------------------

    def _query_database(self) -> dict[str, Any]:
        """Return live database metadata (version/size/table counts)."""
        raise NotImplementedError

    def _get_value_from_database(self, field: str) -> str | None:
        """Return a single field value from the backend METADATA table."""
        raise NotImplementedError


class SQLiteDatabaseMetadataCache(BaseDatabaseMetadataCache):
    """SQLite-backed metadata cache implementation."""

    def _query_database(self) -> dict[str, Any]:
        """Query SQLite for size, table counts and version."""
        db_path = self.db_credentials["name"]
        path = Path(db_path)

        try:
            db_size = path.stat().st_size if path.exists() else 0
        except OSError:
            db_size = 0

        table_counts: dict[str, int] = {}
        db_version = "Unknown"
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                table_counts[table] = self._count_table_rows(cursor, table)

            try:
                cursor.execute("SELECT sqlite_version()")
                row = cursor.fetchone()
                if row and row[0]:
                    db_version = f"SQLite {row[0]}"
            except sqlite3.Error as exc:
                _LOGGER.warning("Failed to get SQLite version: %s", exc)

        return {
            "db_version": db_version,
            "db_size_bytes": int(db_size),
            "table_counts": table_counts,
            "last_db_update": time.time(),
        }

    @staticmethod
    def _count_table_rows(cursor: sqlite3.Cursor, table: str) -> int:
        """Row estimate via MAX(id)+1 when an id column exists, else COUNT."""
        try:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            has_id = any(col[1] == "id" for col in cursor.fetchall())
            if has_id:
                cursor.execute(f"SELECT MAX(id) FROM `{table}`")
                row = cursor.fetchone()
                value = row[0] if row and row[0] is not None else 0
                return int(value) + 1
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            row = cursor.fetchone()
            value = row[0] if row and row[0] is not None else 0
            return int(value)
        except sqlite3.Error as exc:
            _LOGGER.warning("Could not get count for table %s: %s", table, exc)
            return 0

    def get_database_info(self) -> dict[str, Any]:
        """Structured info including the SQLite source path."""
        db_info = super().get_database_info()
        db_info["sqlite_source_path"] = self.db_credentials["name"]
        return db_info

    def _get_value_from_database(self, field: str) -> str | None:
        """Read one field from the SQLite METADATA table (parameterized)."""
        try:
            db_path = self.db_credentials["name"]
            if not Path(db_path).exists():
                # Suppress warning for dummy 'temp' database lookups
                if db_path == "temp" and field == "machine_name":
                    return None
                _LOGGER.warning("SQLite database path does not exist: %s", db_path)
                return None
            with closing(sqlite3.connect(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM METADATA WHERE field = ?", (field,)
                )
                row = cursor.fetchone()
        except (sqlite3.Error, KeyError) as exc:
            _LOGGER.exception("Failed to get %s from SQLite metadata", field)
            _LOGGER.debug("Underlying error: %s", exc)
            return None
        else:
            return str(row[0]) if row else None


def create_metadata_cache(
    db_credentials: DbCredentials,
    device_name: str = "",
    cache_dir: str = DEFAULT_CACHE_DIR,
    database_type: str | None = None,  # back-compat, always SQLite
) -> SQLiteDatabaseMetadataCache:
    """Factory returning the SQLite metadata cache."""
    _ = database_type
    return SQLiteDatabaseMetadataCache(db_credentials, device_name, cache_dir)


# Backward compatibility alias
DatabaseMetadataCache = SQLiteDatabaseMetadataCache


class DatabasesInfo:
    """Fleet-level view of historical databases for one device."""

    def __init__(self, device_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        self.device_name: str = device_name
        self.cache_dir: str = cache_dir
        self._databases_info_cache: dict[str, Any] | None = None
        self._databases_info_cache_time: float = 0
        self.get_all_databases_info()

    def get_databases_info(self) -> dict[str, Any]:
        """Cached (TTL) view of all databases for this device."""
        now = time.time()
        if (
            self._databases_info_cache is not None
            and now - self._databases_info_cache_time < CACHE_TTL_SECONDS
        ):
            return self._databases_info_cache
        databases_info = self.get_all_databases_info()
        self._databases_info_cache = databases_info
        self._databases_info_cache_time = now
        return databases_info

    def _invalidate_databases_cache(self) -> None:
        """Force the next read to hit disk again."""
        self._databases_info_cache = None
        self._databases_info_cache_time = 0

    def get_all_databases_info_as_simple_list(self) -> dict[str, Any]:
        """Flat list form used to populate UI dropdowns."""
        databases_data = self.get_all_databases_info()
        database_list = [
            {
                "name": db_name,
                "type": "SQLite",
                "active": True,
                "size": db_info.get("filesize", 0),
                "status": db_info.get("db_status", "unknown"),
                "path": db_info.get("path", ""),
            }
            for db_name, db_info in databases_data.get("SQLite", {}).items()
        ]
        return {"database_list": database_list}

    def get_all_databases_info(self) -> dict[str, Any]:
        """Historical SQLite databases reconstructed from cache files."""
        databases: dict[str, dict[str, dict[str, Any]]] = {"SQLite": {}}

        if not self.device_name:
            _LOGGER.warning("Empty device_name provided to get_all_databases_info")
            return databases
        if not self._ensure_cache_dir():
            return databases

        bucket = databases["SQLite"]
        for experiment in self._collect_sqlite_experiments():
            self._register_experiment(bucket, experiment)

        if not bucket:
            _LOGGER.info(
                "No databases found in cache for %s, attempting direct discovery",
                self.device_name,
            )
            return self._fallback_database_discovery()
        return {"SQLite": bucket}

    # -- helpers ---------------------------------------------------------

    def _ensure_cache_dir(self) -> bool:
        """Create the cache directory if missing; False on failure."""
        if Path(self.cache_dir).exists():
            return True
        _LOGGER.warning("Cache directory does not exist: %s", self.cache_dir)
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            _LOGGER.exception(
                "Failed to create cache directory %s", self.cache_dir
            )
            return False
        _LOGGER.info("Created cache directory: %s", self.cache_dir)
        return True

    def _collect_sqlite_experiments(self) -> list[dict[str, Any]]:
        """Validated experiment records from every readable cache file."""
        probe = SQLiteDatabaseMetadataCache(
            {"name": "temp"},
            device_name=self.device_name,
            cache_dir=self.cache_dir,
        )
        experiments: list[dict[str, Any]] = []
        for cache_file in probe._get_all_cache_files():
            record = self._experiment_record(cache_file)
            if record is not None:
                experiments.append(record)
        return experiments

    def _experiment_record(self, cache_file: str) -> dict[str, Any] | None:
        """Validate one cache file and extract its experiment record."""
        cache_data = self._load_cache_data(cache_file)
        if cache_data is None:
            return None

        experiment_info = cache_data.get("experiment_info", {})
        if not isinstance(experiment_info, dict) or not experiment_info:
            _LOGGER.debug("No experiment info found in cache file: %s", cache_file)
            return None

        result_writer_type = experiment_info.get("result_writer_type")
        backup_filename = experiment_info.get("backup_filename")
        if not result_writer_type or not backup_filename:
            _LOGGER.debug("Cache file missing required fields: %s", cache_file)
            return None
        if result_writer_type not in SQLITE_WRITER_TYPES:
            _LOGGER.debug(
                "Skipping non-SQLite result writer type '%s' in %s",
                result_writer_type,
                cache_file,
            )
            return None

        return {
            "date_time": experiment_info.get("date_time", 0),
            "backup_filename": backup_filename,
            "user": experiment_info.get("user", "unknown"),
            "location": experiment_info.get("location", "unknown"),
            "result_writer_type": result_writer_type,
            "db_size_bytes": cache_data.get("db_size_bytes", 0),
            "table_counts": cache_data.get("table_counts", {}),
            "db_status": cache_data.get("db_status", "unknown"),
            "db_version": cache_data.get("db_version", "Unknown"),
            "db_name": cache_data.get("db_name", ""),
            "sqlite_source_path": experiment_info.get("sqlite_source_path", ""),
        }

    def _load_cache_data(self, cache_file: str) -> dict[str, Any] | None:
        """Read and size-check one cache file; None when unusable."""
        path = Path(cache_file)
        try:
            problem = self._cache_file_problem(path, cache_file)
            if problem is not None:
                _LOGGER.warning("%s: %s", problem, cache_file)
                return None
            with path.open(encoding="utf-8") as f:
                cache_data = json.load(f)
        except OSError as exc:
            _LOGGER.warning("Cannot access cache file %s: %s", cache_file, exc)
            return None
        except json.JSONDecodeError as exc:
            _LOGGER.warning("Invalid JSON in cache file %s: %s", cache_file, exc)
            return None

        if not isinstance(cache_data, dict):
            _LOGGER.warning("Invalid cache data format in %s", cache_file)
            return None
        return cache_data

    @staticmethod
    def _cache_file_problem(path: Path, cache_file: str) -> str | None:
        """Human-readable reason the cache file is unusable, if any."""
        if not path.exists():
            return "Cache file no longer exists"
        size = path.stat().st_size
        if size == 0:
            return "Cache file is empty"
        if size > MAX_CACHE_FILE_BYTES:
            _LOGGER.warning(
                "Cache file too large (%s bytes): %s", size, cache_file
            )
            return "Cache file too large"
        return None

    @staticmethod
    def _register_experiment(
        bucket: dict[str, dict[str, Any]], experiment: dict[str, Any]
    ) -> None:
        """Add one experiment record to the SQLite bucket."""
        backup_filename = experiment.get("backup_filename", "unknown")
        sqlite_path = experiment.get("sqlite_source_path", "")
        bucket[backup_filename] = {
            "filesize": experiment.get("db_size_bytes", 0),
            "backup_filename": backup_filename,
            "version": experiment.get("db_version", "Unknown"),
            "path": sqlite_path,
            "date": experiment.get("date_time", 0),
            "db_status": experiment.get("db_status", "unknown"),
            "table_counts": experiment.get("table_counts", {}),
            "file_exists": bool(sqlite_path) and Path(sqlite_path).exists(),
        }

    def _fallback_database_discovery(self) -> dict[str, Any]:
        """Scan default result directories when cache files are unusable."""
        databases: dict[str, dict[str, dict[str, Any]]] = {"SQLite": {}}
        bucket = databases["SQLite"]

        for search_path in (Path("/ethoscope_data/results"),):
            if not search_path.exists():
                continue
            try:
                for db_file in search_path.rglob("*.db"):
                    if self.device_name.lower() not in db_file.name.lower():
                        continue
                    record = self._probe_sqlite_file(db_file)
                    if record is not None:
                        bucket[db_file.name] = record
            except OSError as exc:
                _LOGGER.debug("Error searching path %s: %s", search_path, exc)

        _LOGGER.info(
            "Fallback discovery found %s SQLite databases for %s",
            len(bucket),
            self.device_name,
        )
        return {"SQLite": bucket}

    @staticmethod
    def _probe_sqlite_file(db_file: Path) -> dict[str, Any] | None:
        """Validate a candidate .db file and describe it."""
        try:
            with closing(sqlite3.connect(str(db_file), timeout=5.0)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
                )
                valid = cursor.fetchone() is not None
            if not valid:
                return None
            stat = db_file.stat()
        except (sqlite3.Error, OSError) as exc:
            _LOGGER.debug("Skipping file %s: %s", db_file, exc)
            return None

        _LOGGER.info("Discovered SQLite database: %s", db_file)
        return {
            "filesize": stat.st_size,
            "backup_filename": db_file.name,
            "version": "SQLite 3.x",
            "path": str(db_file),
            "date": stat.st_mtime,
            "db_status": "discovered",
            "table_counts": {},
            "file_exists": True,
        }


def get_all_databases_info(
    device_name: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> dict[str, Any]:
    """Top-level helper wrapping :class:`DatabasesInfo`."""
    return DatabasesInfo(device_name, cache_dir).get_all_databases_info()
