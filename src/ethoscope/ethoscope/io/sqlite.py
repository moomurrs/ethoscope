import importlib
import logging
import os
import socket
import sqlite3
import time
import traceback

import cv2
import numpy as np

from .base import BaseAsyncSQLWriter, BaseResultWriter
from .helpers import Null

# Throttle for per-frame diagnostics writes: at most one 'diagnostic' row is
# recorded per this many seconds of experiment time (t is in milliseconds).
# Change this constant to adjust the diagnostics sampling rate.
DIAGNOSTICS_PERIOD_SECONDS = 1.0

# Wall-clock seconds between retention sweeps (DELETE of expired diagnostics rows).
DIAGNOSTICS_RETENTION_SWEEP_SECONDS = 60.0

_DIAGNOSTICS_TABLE_NAME = "diagnostic"
_DIAGNOSTICS_TABLE_FIELDS = (
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "t INTEGER, "
    "mean_brightness REAL, "
    "median_brightness REAL, "
    "std_brightness REAL, "
    "min_brightness REAL, "
    "max_brightness REAL, "
    "contrast_rms REAL, "
    "contrast_range REAL, "
    "histogram_entropy REAL, "
    "edge_density REAL"
)
_DIAGNOSTICS_INSERT_COLUMNS = (
    "t, mean_brightness, median_brightness, std_brightness, min_brightness, "
    "max_brightness, contrast_rms, contrast_range, histogram_entropy, edge_density"
)
_DIAGNOSTICS_INDEX_COMMAND = (
    f"CREATE INDEX IF NOT EXISTS idx_{_DIAGNOSTICS_TABLE_NAME}_t "
    f"ON {_DIAGNOSTICS_TABLE_NAME}(t)"
)

# Journald identifier for the diagnostics journal mirror (filter with
# 'journalctl -t ethoscope-diagnostics -f').
_DIAGNOSTICS_JOURNAL_IDENTIFIER = "ethoscope-diagnostics"

# Native journald datagram socket and priority (6 = journald LOG_INFO) used
# by the diagnostics journal mirror.
_JOURNALD_SOCKET_PATH = "/run/systemd/journal/socket"
_JOURNALD_LOG_INFO = 6


class AsyncSQLiteWriter(BaseAsyncSQLWriter):
    """
    Asynchronous SQLite database writer running in a separate process.

    Similar to AsyncMySQLWriter but for SQLite databases. Uses specific
    PRAGMA settings for optimal performance with single-writer pattern.
    Each experiment creates a unique database file, preserving historical data.

    Attributes:
        _pragmas (dict): SQLite PRAGMA settings for performance optimization
        _db_name (str): Path to SQLite database file
    """

    _database_type = "SQLite3"
    _pragmas = {
        "temp_store": "MEMORY",
        "journal_mode": "WAL",
        "locking_mode": "NORMAL",
        "busy_timeout": "30000",
        "synchronous": "NORMAL",
    }

    def __init__(self, db_name, queue, erase_old_db=True):
        """
        Initialize the async SQLite writer.

        Args:
            db_name (str): Path to SQLite database file (typically unique per experiment)
            queue (multiprocessing.Queue): Queue for receiving SQL commands
            erase_old_db (bool): Whether to delete existing database (typically False since
                                filenames are unique per experiment)
        """
        super().__init__(queue, erase_old_db)
        self._db_name = db_name

    def _get_connection(self):
        """
        Create SQLite database connection.

        Returns:
            sqlite3.Connection: Database connection object

        Raises:
            Exception: If SQLite connection fails
        """
        try:
            db = sqlite3.connect(self._db_name, timeout=30.0)
            return db
        except sqlite3.Error as e:
            raise Exception(
                f"Failed to connect to SQLite database {self._db_name}: {e}"
            ) from e

    # Implementation of abstract methods from BaseAsyncSQLWriter
    def _initialize_database(self):
        """Initialize SQLite database setup - delete file and set PRAGMAs if needed."""
        if self._erase_old_db:
            try:
                os.remove(self._db_name)
            except Exception:
                pass

            # Ensure directory exists before creating database connection
            db_dir = os.path.dirname(self._db_name)
            if db_dir:  # Only create directory if path contains a directory component
                os.makedirs(db_dir, exist_ok=True)
                logging.info(f"Created SQLite directory: {db_dir}")

            conn = self._get_connection()
            c = conn.cursor()
            logging.info("Setting DB parameters")
            for k, v in list(self._pragmas.items()):
                command = f"PRAGMA {str(k)} = {str(v)}"
                c.execute(command)
            conn.close()

    def _get_db_type_name(self):
        """Return database type name for logging."""
        return "SQLite"

    def _should_retry_on_error(self, error):
        """
        Determine if SQLite writer should continue after an error.

        Retries on transient errors like database locks, but stops on critical errors.
        """
        import sqlite3

        # Retry on transient SQLite errors
        if isinstance(error, sqlite3.OperationalError):
            error_msg = str(error).lower()
            # Retry on database lock, busy, or temporary errors
            if any(
                keyword in error_msg for keyword in ["locked", "busy", "cannot commit"]
            ):
                logging.warning(f"SQLite transient error, will retry: {error}")
                return True

        # Stop on all other errors (corrupted database, disk full, etc.)
        logging.error(f"SQLite critical error, stopping writer: {error}")
        return False


class SQLiteResultWriter(BaseResultWriter):
    """
    SQLite-specific result writer.

    Extends BaseResultWriter with SQLite-specific modifications including:
    - Use of AsyncSQLiteWriter instead of AsyncMySQLWriter
    - NULL instead of 0 for auto-increment fields
    - Removal of MySQL-specific table options
    - Automatic placeholder conversion from MySQL (%s) to SQLite (?)
    """

    _description = {
        "overview": "SQLite result writer - stores tracking data to local SQLite database file using consistent directory structure. Each experiment creates a unique file, preserving historical data. Compatible with rsync-based backups. Supports sensor data collection when sensors are available. Optionally records per-frame image quality metrics (computed over the union of the arena ROI polygons) to a 'diagnostic' table when enable_diagnostics is turned on.",
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
            {
                "name": "enable_diagnostics",
                "description": "Record per-frame image quality diagnostics (brightness, contrast, entropy, edge density; computed over the union of the arena ROI polygons) to the 'diagnostic' table",
                "type": "boolean",
                "default": False,
            },
            {
                "name": "diagnostics_retention_minutes",
                "description": "Diagnostics retention limit in minutes: delete diagnostic rows older than this (0 = keep all)",
                "type": "number",
                "min": 0,
                "step": 1,
                "default": 0,
                "depends_on": {"enable_diagnostics": [True]},
            },
            {
                "name": "enable_diagnostics_journal",
                "description": "Also log every diagnostics sample to journalctl with identifier 'ethoscope-diagnostics' (requires enable_diagnostics)",
                "type": "boolean",
                "default": False,
                "depends_on": {"enable_diagnostics": [True]},
            },
        ],
    }

    _database_type = "SQLite3"
    _async_writing_class = AsyncSQLiteWriter
    _null = Null()

    def __init__(
        self,
        db_credentials,
        rois,
        metadata=None,
        make_dam_like_table=False,
        take_frame_shots=False,
        erase_old_db=True,
        sensor=None,
        enable_diagnostics=False,
        diagnostics_retention_minutes=0,
        enable_diagnostics_journal=False,
        *args,
        **kwargs,
    ):
        """
        Initialize SQLite result writer.

        Note: DAM-like tables are disabled by default for SQLite.
        Args:
            sensor: Optional sensor object for environmental data collection
            enable_diagnostics: When True, record per-frame image quality metrics
                to the 'diagnostic' table (throttled to DIAGNOSTICS_PERIOD_SECONDS).
                When False, the 'diagnostic' table is not created at all.
            diagnostics_retention_minutes: Delete diagnostic rows older than this
                many minutes. 0 or less means keep all rows forever. Only used
                when enable_diagnostics is True.
            enable_diagnostics_journal: When True, additionally log each
                diagnostics sample to journald with identifier
                'ethoscope-diagnostics' at the same DIAGNOSTICS_PERIOD_SECONDS
                cadence. Requires enable_diagnostics to be True.
        """
        # Diagnostics configuration.
        # Must be set before super().__init__() because _create_all_tables()
        # (called during parent initialisation) reads these flags.
        self._enable_diagnostics = bool(enable_diagnostics)
        self._enable_diagnostics_journal = bool(enable_diagnostics_journal)
        try:
            self._diagnostics_retention_minutes = int(diagnostics_retention_minutes)
        except (TypeError, ValueError):
            self._diagnostics_retention_minutes = 0
        self._diagnostics_period = DIAGNOSTICS_PERIOD_SECONDS
        self._diagnostics_last_tick = -1
        self._diagnostics_last_retention_sweep = 0.0
        # Arena mask cache (built lazily by _get_arena_mask).
        self._diagnostics_arena_mask = None
        self._diagnostics_arena_mask_shape = None
        self._diagnostics_mask_warned = False

        # SQLite-specific parameter overrides
        # Remove any conflicting arguments from kwargs to avoid duplicate argument errors
        kwargs.pop("erase_old_db", None)

        # SQLite databases are unique per experiment, don't erase them

        # Call parent initialization with all common logic
        super().__init__(
            db_credentials,
            rois,
            metadata,
            make_dam_like_table,
            take_frame_shots,
            erase_old_db,
            sensor,
            **kwargs,
        )

    def get_last_timestamp(self):
        """
        Connects to the database and retrieves the last timestamp
        from all ROI tables with enhanced error handling and validation.
        Returns:
            int: The last timestamp in milliseconds, or 0 if not found.
        """
        db_path = self._db_credentials["name"]

        # Check if database file exists
        if not os.path.exists(db_path):
            logging.error(f"SQLite database file does not exist: {db_path}")
            return 0

        # Check if database file is readable and not empty
        try:
            file_size = os.path.getsize(db_path)
            if file_size == 0:
                logging.error(f"SQLite database file is empty: {db_path}")
                return 0
        except OSError as e:
            logging.error(f"Cannot access SQLite database file {db_path}: {e}")
            return 0

        try:
            # Use a timeout to prevent hanging on locked databases
            db = sqlite3.connect(db_path, timeout=30.0)
            cursor = db.cursor()

            # Check if database has the expected structure by looking for required tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ROI_%'"
            )
            existing_roi_tables = {row[0] for row in cursor.fetchall()}

            if not existing_roi_tables:
                logging.warning(f"No ROI tables found in SQLite database: {db_path}")
                cursor.close()
                db.close()
                return 0

            last_ts = 0
            successful_queries = 0

            for roi in self._rois:
                table_name = f"ROI_{roi.idx}"

                # Check if this specific ROI table exists
                if table_name not in existing_roi_tables:
                    logging.warning(
                        f"ROI table {table_name} not found in database, skipping"
                    )
                    continue

                try:
                    # Validate table structure by checking for required columns
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [
                        col[1] for col in cursor.fetchall()
                    ]  # col[1] is the column name

                    if "t" not in columns:
                        logging.error(f"Table {table_name} missing required 't' column")
                        continue

                    # Get the maximum timestamp from this table
                    cursor.execute(
                        f"SELECT MAX(t) FROM {table_name} WHERE t IS NOT NULL"
                    )
                    result = cursor.fetchone()

                    if result and result[0] is not None:
                        table_max_ts = int(result[0])  # Ensure it's an integer
                        last_ts = max(last_ts, table_max_ts)
                        successful_queries += 1
                        logging.debug(
                            f"Table {table_name} max timestamp: {table_max_ts}"
                        )
                    else:
                        logging.info(
                            f"Table {table_name} has no data or null timestamps"
                        )

                except sqlite3.Error as table_err:
                    logging.error(f"Error querying table {table_name}: {table_err}")
                    continue

            cursor.close()
            db.close()

            if successful_queries == 0:
                logging.warning("No ROI tables could be successfully queried")
                return 0

            logging.info(
                f"Successfully retrieved last timestamp {last_ts} from {successful_queries} ROI table(s)"
            )
            return last_ts

        except sqlite3.DatabaseError as db_err:
            logging.error(f"SQLite database error accessing {db_path}: {db_err}")
            return 0
        except sqlite3.Error as err:
            logging.error(f"SQLite error getting last timestamp from {db_path}: {err}")
            return 0
        except Exception as e:
            logging.error(
                f"Unexpected error getting last timestamp from SQLite {db_path}: {e}"
            )
            logging.error(f"Traceback: {traceback.format_exc()}")
            return 0

    def _create_async_writer(self, db_credentials, erase_old_db, **kwargs):
        """Create SQLite-specific async writer."""
        # SQLite uses the db path directly from db_credentials["name"]
        return self._async_writing_class(
            db_credentials["name"], self._queue, erase_old_db
        )

    def __getstate__(self):
        """Extend base pickle state with SQLite-specific parameters."""
        state = super().__getstate__()
        # SQLite doesn't need extra kwargs, but we set empty dict for consistency
        state["_pickle_extra_kwargs"] = {}
        return state

    def _write_async_command(self, command, args=None):
        """
        Send SQL command to async writer process with SQLite placeholder conversion and resilience.

        Args:
            command (str): SQL command to execute (may contain MySQL placeholders)
            args (tuple): Optional arguments for parameterized query

        Returns:
            bool: True if command was sent successfully, False if buffered
        """
        # Convert MySQL placeholders (%s) to SQLite placeholders (?)
        if "%s" in command:
            sqlite_command = command.replace("%s", "?")
            logging.debug(
                f"Converting MySQL command to SQLite: {command} -> {sqlite_command}"
            )
        else:
            sqlite_command = command

        # Convert Null() objects to None for SQLite compatibility
        if args is not None:
            sqlite_args = []
            for arg in args:
                if isinstance(arg, Null):
                    sqlite_args.append(None)  # SQLite expects None for NULL
                else:
                    sqlite_args.append(arg)
            sqlite_args = tuple(sqlite_args)
        else:
            sqlite_args = None

        # Use the resilient write method from parent class
        return self._write_async_command_resilient(sqlite_command, sqlite_args)

    def _create_table(self, name, fields, engine=None):
        """
        Create SQLite table (ignores engine parameter).

        Args:
            name (str): Table name
            fields (str): Field definitions
            engine: Ignored for SQLite
        """
        # Don't modify fields for SQLite - they should already be SQLite-compatible
        command = f"CREATE TABLE IF NOT EXISTS {name} ({fields})"
        logging.info("Creating database table with: " + command)
        self._write_async_command(command)

    def _initialise_roi_table(self, roi, data_row):
        """Initialize ROI-specific database table with SQLite-compatible syntax."""
        # SQLite-specific field definitions
        fields = ["id INTEGER PRIMARY KEY AUTOINCREMENT", "t INTEGER"]
        for dt in list(data_row.values()):
            # Convert MySQL types to SQLite equivalents
            sql_type = dt.sql_data_type.upper()
            if "INT" in sql_type:
                sqlite_type = "INTEGER"
            elif "FLOAT" in sql_type or "DOUBLE" in sql_type:
                sqlite_type = "REAL"
            elif "TEXT" in sql_type or "CHAR" in sql_type or "VARCHAR" in sql_type:
                sqlite_type = "TEXT"
            else:
                sqlite_type = "TEXT"  # Default fallback
            fields.append(f"{dt.header_name} {sqlite_type}")
        fields = ", ".join(fields)
        table_name = f"ROI_{roi.idx}"
        self._create_table(table_name, fields, engine=None)

    def _add(self, t, roi, data_rows):
        """
        Add data with proper type preservation and parameterized queries.

        Uses parameterized queries to prevent SQL injection and preserve data types.
        Converts booleans to integers (0/1) for SQLite storage.
        """
        t = int(round(t))
        roi_id = roi.idx

        # Initialize insert data list for this ROI if not exists
        if roi_id not in self._insert_dict:
            self._insert_dict[roi_id] = []

        for dr in data_rows:
            # Build values tuple with proper type handling
            values = [None if isinstance(self._null, Null) else self._null, t] + list(
                dr.values()
            )

            # Convert values to proper SQLite types
            sqlite_values = []
            for val in values:
                if val is None or isinstance(val, Null):
                    sqlite_values.append(None)  # SQLite NULL
                elif isinstance(val, bool):
                    sqlite_values.append(1 if val else 0)  # Convert bool to int
                else:
                    sqlite_values.append(val)  # Keep original type

            # Store as tuple for parameterized query
            self._insert_dict[roi_id].append(tuple(sqlite_values))

        # now this is irrelevant when tracking multiple animals
        if self._dam_file_helper is not None:
            for dr in data_rows:
                self._dam_file_helper.input_roi_data(t, roi, dr)

    def _batch_insert_roi(self, roi_id, value_list):
        """
        Helper to insert ROI data in batches of 50 rows using compound INSERT statements.
        Handles the pre-computation of the SQL string and the trailing layout edge-cases.
        """
        if not value_list:
            return

        num_cols = len(value_list[0])
        single_row_placeholders = "(" + ", ".join(["?"] * num_cols) + ")"
        batch_size = 50

        # Pre-compute the query string layout for a perfect full batch of 50
        full_batch_placeholders = ", ".join([single_row_placeholders] * batch_size)
        full_batch_command = (
            f"INSERT INTO ROI_{roi_id} VALUES {full_batch_placeholders}"
        )

        for i in range(0, len(value_list), batch_size):
            batch = value_list[i : i + batch_size]
            actual_batch_size = len(batch)

            # If it's a standard batch of 50, use the pre-computed command string
            if actual_batch_size == batch_size:
                command = full_batch_command
            else:
                # Edge-case: Last batch contains fewer than 50 rows, adjust query dynamically
                partial_placeholders = ", ".join(
                    [single_row_placeholders] * actual_batch_size
                )
                command = f"INSERT INTO ROI_{roi_id} VALUES {partial_placeholders}"

            # Flatten the multi-row data matrix into a 1D tuple of arguments
            flattened_args = tuple(val for row in batch for val in row)
            self._write_async_command(command, flattened_args)

    def flush(self, t, img=None):
        """
        Flush accumulated data to database using parameterized queries.

        Overrides base class flush to handle list-based insert data with proper types.
        """
        # Per-frame image quality diagnostics (throttled to DIAGNOSTICS_PERIOD_SECONDS)
        # getattr guard: writers reconstructed via pickle/object.__new__ may predate
        # this feature or bypass __init__
        if (
            getattr(self, "_enable_diagnostics", False)
            and img is not None
            and t is not None
        ):
            self._record_diagnostics(t, img)

        # Handle helper flushes (dam, shots, sensors) same as base class
        if self._dam_file_helper is not None:
            out = self._dam_file_helper.flush(t)
            for c in out:
                self._write_async_command(c)
        if self._shot_saver is not None and img is not None:
            c_args = self._shot_saver.flush(t, img)
            if c_args is not None:
                self._write_async_command(*c_args)
        if self._sensor_saver is not None:
            c_args = self._sensor_saver.flush(t)
            if c_args is not None:
                self._write_async_command(*c_args)

        # Handle ROI data inserts via chunked batch queries of 50 rows
        for roi_id, value_list in list(self._insert_dict.items()):
            if len(value_list) >= self._max_insert_string_len:
                if value_list:  # Only if we have data
                    self._batch_insert_roi(roi_id, value_list)
                    # Clear the list after flushing
                    self._insert_dict[roi_id] = []
        return False

    def _create_diagnostics_table(self):
        """Create the 'diagnostic' table and its timestamp index (idempotent)."""
        logging.info(
            "Creating '%s' table (frame diagnostics enabled)" % _DIAGNOSTICS_TABLE_NAME
        )
        self._create_table(_DIAGNOSTICS_TABLE_NAME, _DIAGNOSTICS_TABLE_FIELDS)
        self._write_async_command(_DIAGNOSTICS_INDEX_COMMAND)

    def _warn_no_arena_mask(self):
        """Warn once that diagnostics fall back to whole-frame metrics."""
        if not getattr(self, "_diagnostics_mask_warned", False):
            self._diagnostics_mask_warned = True
            logging.warning(
                "Frame diagnostics: no usable ROI polygons; "
                "falling back to whole-frame metrics."
            )

    def _get_arena_mask(self, frame_shape):
        """
        Return a full-frame binary mask covering the union of all ROI polygons.

        The mask is built once with cv2.fillPoly over every ROI polygon and
        cached until the frame shape changes. Returns None (and warns once)
        when no usable ROIs are available, so callers fall back to
        whole-frame metrics.

        Args:
            frame_shape: Shape of the current camera frame (np.ndarray.shape)

        Returns:
            uint8 mask of shape (h, w) with 255 inside the arenas, or None
            when no ROI polygons are usable.
        """
        shape_key = (int(frame_shape[0]), int(frame_shape[1]))
        if getattr(self, "_diagnostics_arena_mask_shape", None) == shape_key:
            mask = getattr(self, "_diagnostics_arena_mask", None)
            if mask is not None:
                return mask

        rois = getattr(self, "_rois", None) or []
        polygons = [r.polygon for r in rois if getattr(r, "polygon", None) is not None]
        if not polygons:
            self._warn_no_arena_mask()
            return None

        mask = np.zeros(shape_key, dtype=np.uint8)
        cv2.fillPoly(mask, polygons, 255)
        if not np.count_nonzero(mask):
            # e.g. polygons entirely outside the frame
            self._warn_no_arena_mask()
            return None

        self._diagnostics_arena_mask = mask
        self._diagnostics_arena_mask_shape = shape_key
        return mask

    @staticmethod
    def _calculate_entropy(hist):
        """Calculate entropy of histogram for image complexity measure."""
        hist = hist + 1e-10  # Avoid log(0)
        hist_norm = hist / np.sum(hist)
        return float(-np.sum(hist_norm * np.log2(hist_norm)))

    def _analyze_frame_quality(self, image, mask=None):
        """
        Compute per-frame image quality metrics for the 'diagnostic' table.

        When a mask is given, all statistics are restricted to the masked
        (arena) pixels; otherwise they cover the whole frame. With mask=None
        the calculations are identical to
        ethoscope.roi_builders.target_detection_diagnostics.TargetDetectionDiagnostics.analyze_image_quality
        (only the 'image_shape' entry is omitted).

        Args:
            image: Input frame (BGR or grayscale)
            mask: Optional uint8 mask with the same (h, w) as the frame;
                non-zero pixels are included in the statistics

        Returns:
            Dictionary with the 9 image quality metrics stored in the table
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if mask is not None:
            pixels = gray[mask > 0]
            if pixels.size == 0:
                # Degenerate mask (no arena pixels): use the whole frame
                mask = None
        if mask is None:
            pixels = gray

        # Basic statistics
        mean_brightness = float(np.mean(pixels))
        median_brightness = float(np.median(pixels))
        std_brightness = float(np.std(pixels))
        min_brightness = float(np.min(pixels))
        max_brightness = float(np.max(pixels))

        # Contrast measures
        contrast_rms = float(np.sqrt(np.mean((pixels - mean_brightness) ** 2)))
        contrast_range = max_brightness - min_brightness

        # Histogram analysis (restricted to the mask when given)
        hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
        hist_entropy = self._calculate_entropy(hist.flatten())

        # Edge density (proxy for detail/noise). Canny runs on the full
        # grayscale frame to avoid fake edges at the mask boundary, but only
        # edge pixels inside the mask are counted.
        edges = cv2.Canny(gray, 50, 150)
        if mask is not None:
            edge_density = float(
                np.sum((edges > 0) & (mask > 0)) / np.count_nonzero(mask)
            )
        else:
            edge_density = float(np.sum(edges > 0) / edges.size)

        return {
            "mean_brightness": mean_brightness,
            "median_brightness": median_brightness,
            "std_brightness": std_brightness,
            "min_brightness": min_brightness,
            "max_brightness": max_brightness,
            "contrast_rms": contrast_rms,
            "contrast_range": contrast_range,
            "histogram_entropy": hist_entropy,
            "edge_density": edge_density,
        }

    def _record_diagnostics(self, t, img):
        """
        Record one throttled diagnostics row for the current frame.

        Rows are written at most once per DIAGNOSTICS_PERIOD_SECONDS of
        experiment time (t is the experiment timestamp in milliseconds).
        Metrics are computed over the union of the arena ROI polygons
        (see _get_arena_mask); they cover the whole frame only when no
        usable ROIs exist. The stored 't' column is the wall-clock Unix
        timestamp in seconds, like START_EVENTS and METADATA stop_date_time.

        Args:
            t: Experiment timestamp in milliseconds
            img: Current camera frame (np.ndarray)

        If the journal toggle is enabled, the same throttled sample is also
        logged to journald (see _log_diagnostics_to_journal).
        """
        tick = int(round((int(t) / 1000.0) / self._diagnostics_period))
        if tick == self._diagnostics_last_tick:
            return
        self._diagnostics_last_tick = tick

        try:
            mask = self._get_arena_mask(img.shape)
            metrics = self._analyze_frame_quality(img, mask=mask)
        except Exception:
            logging.error(
                "Failed to compute frame diagnostics:\n%s" % traceback.format_exc()
            )
            return

        command = (
            f"INSERT INTO {_DIAGNOSTICS_TABLE_NAME} "
            f"({_DIAGNOSTICS_INSERT_COLUMNS}) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        args = (
            int(time.time()),
            metrics["mean_brightness"],
            metrics["median_brightness"],
            metrics["std_brightness"],
            metrics["min_brightness"],
            metrics["max_brightness"],
            metrics["contrast_rms"],
            metrics["contrast_range"],
            metrics["histogram_entropy"],
            metrics["edge_density"],
        )
        self._write_async_command(command, args)
        if getattr(self, "_enable_diagnostics_journal", False):
            self._log_diagnostics_to_journal(metrics)
        self._enforce_diagnostics_retention()

    def _log_diagnostics_to_journal(self, metrics):
        """
        Log one diagnostics sample to journald under a dedicated identifier.

        Entries carry SYSLOG_IDENTIFIER=_DIAGNOSTICS_JOURNAL_IDENTIFIER so
        they can be monitored with 'journalctl -t ethoscope-diagnostics -f'.
        All metrics are included in the message text and attached as
        structured fields (visible with 'journalctl -o verbose').

        Transports are tried in order:
        1. The journald native datagram socket (_send_to_journald_socket).
        2. The systemd python bindings (systemd.journal.send).
        3. Standard logging, which still reaches the journal when the device
           service runs under systemd (without the dedicated identifier).

        Journal failures are logged at debug level and never propagate to
        the caller.
        """
        message = (
            "[BRIGHTNESS"
            " mean={mean_brightness:.1f}"
            " median={median_brightness:.1f}"
            " std={std_brightness:.2f}"
            " min={min_brightness:.0f}"
            " max={max_brightness:.0f} ]"
            " [CONTRAST"
            " rms={contrast_rms:.2f}"
            " range={contrast_range:.0f} ]"
            " entropy={histogram_entropy:.2f}"
            " edge_density={edge_density:.4f}"
            " region=arenas"
        ).format(**metrics)

        fields = {
            "REGION": "arenas",
            "MEAN_BRIGHTNESS": "{mean_brightness:.2f}".format(**metrics),
            "MEDIAN_BRIGHTNESS": "{median_brightness:.2f}".format(**metrics),
            "STD_BRIGHTNESS": "{std_brightness:.2f}".format(**metrics),
            "MIN_BRIGHTNESS": "{min_brightness:.0f}".format(**metrics),
            "MAX_BRIGHTNESS": "{max_brightness:.0f}".format(**metrics),
            "CONTRAST_RMS": "{contrast_rms:.3f}".format(**metrics),
            "CONTRAST_RANGE": "{contrast_range:.0f}".format(**metrics),
            "HISTOGRAM_ENTROPY": "{histogram_entropy:.3f}".format(**metrics),
            "EDGE_DENSITY": "{edge_density:.5f}".format(**metrics),
        }

        try:
            self._send_to_journald_socket(message, fields)
            return
        except OSError:
            pass

        try:
            journal = importlib.import_module("systemd.journal")
        except ImportError:
            logging.info(message)
            return

        try:
            journal.send(
                message,
                PRIORITY=journal.LOG_INFO,
                SYSLOG_IDENTIFIER=_DIAGNOSTICS_JOURNAL_IDENTIFIER,
                **fields,
            )
        except Exception:
            logging.debug(
                "Failed to log diagnostics to journald:\n%s" % traceback.format_exc()
            )

    def _send_to_journald_socket(self, message, fields):
        """
        Send one entry to journald via its native datagram socket.

        Implements the minimal journald wire format: newline-separated
        FIELD=value entries sent as a single datagram to
        _JOURNALD_SOCKET_PATH. This is what systemd.journal.send does under
        the hood, without requiring the python3-systemd package. All values
        sent here are single-line, so the simple form is sufficient. Raises
        OSError when the socket is unavailable (e.g. journald not running);
        callers fall back to other transports.
        """
        entries = [
            "MESSAGE=" + message,
            "PRIORITY=%d" % _JOURNALD_LOG_INFO,
            "SYSLOG_IDENTIFIER=" + _DIAGNOSTICS_JOURNAL_IDENTIFIER,
        ]
        entries.extend("%s=%s" % (name, value) for name, value in fields.items())
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto("\n".join(entries).encode("utf-8"), _JOURNALD_SOCKET_PATH)

    def _enforce_diagnostics_retention(self):
        """
        Delete 'diagnostic' rows older than the retention limit.

        Retention is expressed in minutes; 0 or less means keep all rows
        forever. Sweeps are throttled to DIAGNOSTICS_RETENTION_SWEEP_SECONDS
        of wall-clock time. The index on 't' keeps the DELETE cheap.
        """
        if self._diagnostics_retention_minutes <= 0:
            return

        now = time.time()
        if (
            now - self._diagnostics_last_retention_sweep
            < DIAGNOSTICS_RETENTION_SWEEP_SECONDS
        ):
            return
        self._diagnostics_last_retention_sweep = now

        cutoff = int(now) - int(self._diagnostics_retention_minutes * 60)
        self._write_async_command(
            f"DELETE FROM {_DIAGNOSTICS_TABLE_NAME} WHERE t < ?", (cutoff,)
        )

    def close(self):
        """
        Close the writer and flush any remaining data.

        Ensures all accumulated data is written before shutdown.
        """
        # Final flush of all remaining data in batches of 50 rows
        for roi_id, value_list in list(self._insert_dict.items()):
            if value_list:  # Only if we have data
                self._batch_insert_roi(roi_id, value_list)
                # Clear the list after flushing
                self._insert_dict[roi_id] = []

        # Call parent close method
        super().close()

    def _create_all_tables(self):
        """
        Create all necessary SQLite database tables for the experiment.

        Creates SQLite-compatible tables for:
        - ROI_MAP: ROI definitions and positions
        - VAR_MAP: Variable type mappings
        - IMG_SNAPSHOTS: Image snapshot storage (if enabled)
        - CSV_DAM_ACTIVITY: DAM-compatible activity data (if enabled)
        - diagnostic: per-frame image quality metrics over the arena regions (if enable_diagnostics)
        - METADATA: Experimental metadata
        - START_EVENTS: Experiment start/stop events

        Note: SENSORS table is not created as SQLite doesn't support sensors yet
        """
        if self._erase_old_db:
            logging.info("Creating master table 'ROI_MAP'")
            self._create_table(
                "ROI_MAP",
                "roi_idx INTEGER, roi_value INTEGER, x INTEGER, y INTEGER, w INTEGER, h INTEGER",
            )
            for r in self._rois:
                fd = r.get_feature_dict()
                command = "INSERT INTO ROI_MAP VALUES (?, ?, ?, ?, ?, ?)"
                self._write_async_command(
                    command,
                    (fd["idx"], fd["value"], fd["x"], fd["y"], fd["w"], fd["h"]),
                )

            logging.info("Creating variable map table 'VAR_MAP'")
            self._create_table(
                "VAR_MAP", "var_name TEXT, sql_type TEXT, functional_type TEXT"
            )

            if self._shot_saver is not None:
                logging.info("Creating table for IMG_SNAPSHOTS")
                # SQLite-compatible version of image snapshots table
                self._create_table(
                    "IMG_SNAPSHOTS",
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER, img BLOB",
                )

            if self._sensor_saver is not None:
                logging.info("Creating table for SENSORS data")
                # SensorDataHelper handles SQLite-compatible field generation
                self._create_table(
                    self._sensor_saver.table_name, self._sensor_saver.create_command
                )

            if self._dam_file_helper is not None:
                logging.info("Creating 'CSV_DAM_ACTIVITY' table")
                # Convert DAM table fields to SQLite-compatible format
                mysql_fields = self._dam_file_helper.make_dam_file_sql_fields()
                # Convert MySQL field definitions to SQLite equivalents
                sqlite_fields = mysql_fields.replace(
                    "INT  NOT NULL AUTO_INCREMENT PRIMARY KEY",
                    "INTEGER PRIMARY KEY AUTOINCREMENT",
                )
                sqlite_fields = sqlite_fields.replace("CHAR(100)", "TEXT")
                sqlite_fields = sqlite_fields.replace("SMALLINT", "INTEGER")
                self._create_table("CSV_DAM_ACTIVITY", sqlite_fields)

            if self._enable_diagnostics:
                self._create_diagnostics_table()

            logging.info("Creating 'METADATA' table")
            self._create_table("METADATA", "field TEXT, value TEXT")

            logging.info("Creating 'START_EVENTS' table")
            self._create_table(
                "START_EVENTS",
                "id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER, event TEXT",
            )
            event = "graceful_start"
            command = "INSERT INTO START_EVENTS VALUES (?, ?, ?)"
            self._write_async_command(command, (None, int(time.time()), event))

            # Insert experimental metadata using SQLite-specific method
            self._insert_metadata()

            self._wait_for_queue_empty()

        elif not self._erase_old_db and getattr(self, "database_to_append", None):
            event = "appending"
            command = "INSERT INTO START_EVENTS VALUES (?, ?, ?)"
            self._write_async_command(command, (None, int(time.time()), event))
            if self._enable_diagnostics:
                # CREATE TABLE IF NOT EXISTS is idempotent: this also covers
                # databases created before this feature existed (or by runs
                # with the diagnostics toggle off).
                self._create_diagnostics_table()
            self._wait_for_queue_empty()

    def _insert_metadata(self):
        """Insert experimental metadata into METADATA table with SQLite duplicate prevention."""
        import json

        from .base import METADATA_MAX_VALUE_LENGTH

        for k, v in list(self.metadata.items()):
            # Properly serialize complex metadata values to avoid SQL injection and formatting issues
            v_serialized = (
                json.dumps(str(v))
                if not isinstance(v, (str, int, float, bool, type(None)))
                else v
            )

            # Truncate extremely large values as a safety measure
            max_value_length = METADATA_MAX_VALUE_LENGTH
            if isinstance(v_serialized, str) and len(v_serialized) > max_value_length:
                v_serialized = v_serialized[:max_value_length] + "... [TRUNCATED]"
                logging.warning(
                    f"Metadata value for key '{k}' was truncated due to size limit"
                )

            # Use SQLite INSERT OR IGNORE to prevent duplicate key errors
            command = "INSERT OR IGNORE INTO METADATA VALUES (?, ?)"
            self._write_async_command(command, (k, v_serialized))
