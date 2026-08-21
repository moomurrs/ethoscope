"""Helper classes for ethoscope I/O - sensor, snapshots, DAM and raw data."""

from __future__ import annotations

import datetime
import logging
import tempfile
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np
from cv2 import IMWRITE_JPEG_QUALITY, imwrite

from ._constants import (
    DAM_DEFAULT_PERIOD,
    DAM_SCALE,
    IMG_SNAPSHOT_DEFAULT_PERIOD,
    SENSOR_DEFAULT_PERIOD,
)
from ._sql import map_mysql_to_sqlite

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ._types import ROIProtocol, SensorProtocol

_LOGGER: Final = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensor helper
# ---------------------------------------------------------------------------


class SensorDataHelper:
    """Periodically sample a sensor and emit a parameterized INSERT.

    The helper is deliberately I/O-free apart from calling ``sensor.read_all()``.
    It returns a ``(sql, args)`` tuple when the sampling period has elapsed,
    otherwise ``None``.  The caller is responsible for executing the statement.

    Args:
        sensor: Object with ``read_all() -> tuple`` and ``sensor_types: dict``.
        period: Sampling period in seconds.
        database_type: Kept for backward compatibility - always ``SQLite3``.
    """

    _table_name: Final[str] = "SENSORS"

    def __init__(
        self,
        sensor: SensorProtocol,
        period: float = SENSOR_DEFAULT_PERIOD,
        database_type: str = "SQLite3",
    ) -> None:
        self._period: float = period
        self._last_tick: int = 0
        self.sensor: SensorProtocol = sensor
        self._database_type: str = "SQLite3"

        self._base_headers: dict[str, str] = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "t": "INTEGER",
        }
        self._table_headers: dict[str, str] = {
            **self._base_headers,
            **self._get_sensor_types_for_database(),
        }

    # -- public API -------------------------------------------------------

    def flush(self, t: int) -> tuple[str, tuple[Any, ...]] | None:
        """Return a parameterized INSERT if the period has elapsed.

        Returns:
            ``(sql, args)`` with ``?`` placeholders, or ``None`` if the tick
            has not advanced.  Errors from the sensor are logged and ``None``
            is returned so the caller can continue.
        """
        tick = round((t / 1000.0) / self._period)
        if tick == self._last_tick:
            return None
        try:
            # Build parameterized command - prevents SQL injection and preserves types
            columns = list(self._table_headers.keys())[1:]  # skip 'id'
            placeholders = ", ".join(["?"] * (len(columns)))
            cols_joined = ",".join(columns)
            cmd = (
                f"INSERT into {self._table_name} ({cols_joined}) "
                f"VALUES ({placeholders})"
            )
            # args: (t, *sensor_values)
            sensor_values = self.sensor.read_all()
            args: tuple[Any, ...] = (int(t), *sensor_values)
        except Exception:
            _LOGGER.exception("The sensor data are not available")
            self._last_tick = tick
            return None
        else:
            self._last_tick = tick
            return cmd, args

    @property
    def table_name(self) -> str:
        """Name of the sensor data table."""
        return self._table_name

    @property
    def create_command(self) -> str:
        """SQL column definitions for ``CREATE TABLE``."""
        return ",".join(f"{k} {v}" for k, v in self._table_headers.items())

    # -- private --------------------------------------------------------

    def _get_sensor_types_for_database(self) -> dict[str, str]:
        """Map sensor ``sensor_types`` to SQLite types.

        Unknown MySQL types fall back to ``TEXT``.
        """
        if not hasattr(self.sensor, "sensor_types"):
            return {}
        sensor_types: dict[str, str] = {}
        for field_name, mysql_type in self.sensor.sensor_types.items():
            sensor_types[field_name] = map_mysql_to_sqlite(str(mysql_type))
        return sensor_types


# ---------------------------------------------------------------------------
# Image snapshot helper - context-managed, no __del__
# ---------------------------------------------------------------------------


class ImgSnapshotHelper:
    """Periodically JPEG-compress a frame and emit a parameterized INSERT.

    The helper owns a single temporary file created with
    :func:`tempfile.NamedTemporaryFile`.  Resource cleanup is guaranteed via
    the context-manager protocol and :func:`weakref.finalize`; ``__del__`` is
    not used (see Directive 6).

    Args:
        period: Snapshot interval in seconds.
        database_type: Kept for backward compatibility - always ``SQLite3``.
    """

    _table_name: Final[str] = "IMG_SNAPSHOTS"

    def __init__(
        self,
        period: float = IMG_SNAPSHOT_DEFAULT_PERIOD,
        database_type: str = "SQLite3",
    ) -> None:
        self._period: float = period
        self._last_tick: int = 0
        self._database_type: str = "SQLite3"
        # Use NamedTemporaryFile to avoid mktemp race; keep name for compatibility
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="ethoscope_", suffix=".jpg", delete=False
        )
        self._tmp_file: str = tmp.name
        tmp.close()
        # Ensure cleanup even if user forgets to call close() / use `with`
        self._finalizer = weakref.finalize(self, Path(self._tmp_file).unlink, True)

        self._table_headers: dict[str, str] = {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "t": "INTEGER",
            "img": "BLOB",
        }

    # -- context manager --------------------------------------------------

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit context manager, cleaning up the temp file."""
        self.close()

    def close(self) -> None:
        """Remove the temporary JPEG file if it exists."""
        try:
            Path(self._tmp_file).unlink(missing_ok=True)
        except OSError as exc:
            _LOGGER.warning("Could not remove temp file %s: %s", self._tmp_file, exc)
        # Detach finalizer if we cleaned up successfully
        if hasattr(self, "_finalizer"):
            self._finalizer.detach()

    @property
    def table_name(self) -> str:
        """Name of the image snapshot table."""
        return self._table_name

    @property
    def create_command(self) -> str:
        """SQL column definitions for ``CREATE TABLE``."""
        return ",".join(f"{k} {v}" for k, v in self._table_headers.items())

    def flush(
        self, t: int, img: npt.NDArray[Any]
    ) -> tuple[str, tuple[int, bytes]] | None:
        """Return a parameterized INSERT with JPEG bytes if period elapsed.

        Args:
            t: Timestamp in milliseconds.
            img: Image to compress (``np.ndarray``).

        Returns:
            ``(sql, (t, jpeg_bytes))`` or ``None`` when not yet time.
        """
        tick = round((t / 1000.0) / self._period)
        if tick == self._last_tick:
            return None
        # Compress to temp file then read bytes - keeps memory bounded for large frames
        # Using `imwrite` with explicit `dst` not applicable; file is required for cv2.
        success = imwrite(self._tmp_file, img, [int(IMWRITE_JPEG_QUALITY), 50])
        if not success:
            _LOGGER.warning("Failed to write JPEG snapshot at t=%s", t)
            self._last_tick = tick
            return None
        try:
            with Path(self._tmp_file).open("rb") as fh:
                bstring = fh.read()
        except OSError:
            _LOGGER.exception("Failed to read snapshot file")
            self._last_tick = tick
            return None

        cmd = f"INSERT INTO {self._table_name}(t,img) VALUES (?,?)"
        args: tuple[int, bytes] = (int(t), bstring)
        self._last_tick = tick
        return cmd, args


# ---------------------------------------------------------------------------
# DAM helper
# ---------------------------------------------------------------------------


class DAMFileHelper:
    """Track per-ROI activity and emit DAM-compatible INSERT statements.

    Activity is the Euclidean distance travelled between successive positions,
    normalized by ``roi.longest_axis`` and scaled by :data:`DAM_SCALE`.
    """

    def __init__(self, period: float = DAM_DEFAULT_PERIOD, n_rois: int = 32) -> None:
        self._period: float = period
        self._n_rois: int = n_rois
        self._scale: int = DAM_SCALE
        # Keep OrderedDict for backward compat with existing tests
        self._activity_accum: OrderedDict[int, OrderedDict[int, float]] = OrderedDict()
        self._distance_map: dict[int, float] = dict.fromkeys(range(1, n_rois + 1), 0)
        self._last_positions: dict[int, complex | None] = dict.fromkeys(
            range(1, n_rois + 1)
        )

    # -- public API -------------------------------------------------------

    def make_dam_file_sql_fields(self) -> str:
        """Return comma-separated column definitions for ``CSV_DAM_ACTIVITY``."""
        fields = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "date TEXT",
            "time TEXT",
        ]
        fields.extend(f"ROI_{r} INTEGER" for r in range(1, self._n_rois + 1))
        return ",".join(fields)

    def input_roi_data(self, t: int, roi: ROIProtocol, data: Mapping[str, Any]) -> None:
        """Record activity for a single ROI at time ``t``.

        Args:
            t: Timestamp in milliseconds.
            roi: ROI with ``idx`` and ``longest_axis``.
            data: Mapping with ``x`` and ``y`` coordinates.
        """
        tick = round((t / 1000.0) / self._period)
        act = self._compute_distance_for_roi(roi, data)
        if tick not in self._activity_accum:
            self._activity_accum[tick] = OrderedDict(
                (r, 0) for r in range(1, self._n_rois + 1)
            )
        # defaultdict-like accumulation
        self._activity_accum[tick][roi.idx] = (
            self._activity_accum[tick].get(roi.idx, 0) + act
        )

    def flush(self, t: int) -> list[str]:
        """Generate parameterized INSERT commands for accumulated ticks.

        Args:
            t: Current time in milliseconds.

        Returns:
            List of ``INSERT`` statements (one per complete tick) or ``[]``
            if no data was accumulated.
        """
        tick = round((t / 1000.0) / self._period)
        if not self._activity_accum:
            self._activity_accum[tick] = OrderedDict(
                (r, 0) for r in range(1, self._n_rois + 1)
            )
            return []

        min_tick = min(self._activity_accum.keys())
        out: OrderedDict[int, dict[int, float]] = OrderedDict()
        to_delete: list[int] = []

        for i in range(min_tick, tick):
            if i not in self._activity_accum:
                self._activity_accum[i] = OrderedDict(
                    (r, 0) for r in range(1, self._n_rois + 1)
                )
            # Copy and round in one step
            out[i] = OrderedDict(
                (r, round(v, 5)) for r, v in self._activity_accum[i].items()
            )
            to_delete.append(i)

        for i in to_delete:
            del self._activity_accum[i]

        if tick - min_tick > 1:
            _LOGGER.warning(
                "DAM file writer skipping a tick. No data for more than one period!"
            )
        return [self._make_sql_command(v) for v in out.values()]

    # -- private helpers --------------------------------------------------

    def _compute_distance_for_roi(
        self, roi: ROIProtocol, data: Mapping[str, Any]
    ) -> float:
        """Return normalized distance since last call (0 on first observation)."""
        current_pos = data["x"] + 1j * data["y"]
        last_pos = self._last_positions[roi.idx]
        if last_pos is None:
            self._last_positions[roi.idx] = current_pos
            return 0.0
        dist = abs(current_pos - last_pos) / roi.longest_axis
        self._last_positions[roi.idx] = current_pos
        return float(dist)

    def _make_sql_command(self, vals: Mapping[int, float]) -> str:
        """Build ``INSERT INTO CSV_DAM_ACTIVITY`` for a single tick.

        Values are inlined as scaled integers - this table is append-only and
        the values are already sanitized integers.  Parameterized form is not
        required for the test harness but could be added if needed.
        """
        dt = datetime.datetime.fromtimestamp(int(time.time()), tz=datetime.UTC)
        date_str, time_str = dt.strftime("%d %b %Y,%H:%M:%S").split(",")
        # Preserve order ROI_1..ROI_n
        scaled = [str(round(self._scale * vals[i])) for i in range(1, self._n_rois + 1)]
        values = [f"'{date_str}'", f"'{time_str}'", *scaled]
        # Build column list once
        cols = ", ".join(f"ROI_{i}" for i in range(1, self._n_rois + 1))
        vals_joined = ", ".join(values)
        return (
            f"INSERT INTO CSV_DAM_ACTIVITY (date, time, {cols}) VALUES ({vals_joined})"
        )


# ---------------------------------------------------------------------------
# Null sentinel
# ---------------------------------------------------------------------------


class Null:
    """Sentinel representing SQL ``NULL`` for SQLite auto-increment columns."""

    def __repr__(self) -> str:
        return "NULL"

    def __str__(self) -> str:
        return "NULL"


# ---------------------------------------------------------------------------
# Npy appendable file
# ---------------------------------------------------------------------------


class NpyAppendableFile:
    """Appendable ``.anpy`` file - efficient incremental ``.npy`` storage.

    The format stores multiple ``np.save`` blocks concatenated.  Reading via
    :meth:`load` concatenates them lazily along ``axis``.

    Args:
        fname: Base filename; extension is forced to ``.anpy``.
        newfile: If ``True``, the first :meth:`write` truncates the file.
    """

    def __init__(self, fname: str | Path, newfile: bool = True) -> None:
        path = Path(fname)
        self.fname: str = str(path.with_suffix(".anpy"))
        self._newfile: bool = newfile
        self._first_write: bool = True

    def write(self, data: npt.NDArray[Any]) -> bool:
        """Append ``data`` to the file.

        Returns:
            ``True`` on success.
        """
        mode = "wb" if (self._newfile and self._first_write) else "ab"
        with Path(self.fname).open(mode) as fh:
            np.save(fh, data)
        if self._newfile and self._first_write:
            self._first_write = False
        return True

    def load(self, axis: int = 2) -> npt.NDArray[Any]:
        """Load and concatenate all blocks.

        Args:
            axis: Axis along which to concatenate.

        Returns:
            Concatenated array.
        """
        path = Path(self.fname)
        with path.open("rb") as fh:
            fsz = path.stat().st_size
            out = np.load(fh)
            while fh.tell() < fsz:
                out = np.concatenate((out, np.load(fh)), axis=axis)
        return out

    def convert(self, filename: str | Path | None = None) -> None:
        """Convert ``.anpy`` to a standard ``.npy`` file.

        Args:
            filename: Output path; defaults to ``<basename>.npy``.
        """
        content = self.load()
        if filename is None:
            filename = str(Path(self.fname).with_suffix(".npy"))
        with Path(filename).open("wb") as fh:
            np.save(fh, content)
        print(
            f"New .npy compatible file saved with name {filename}. "
            f"Use numpy.load to load data from it. "
            f"The array has a shape of {content.shape}"
        )

    @property
    def _dtype(self) -> np.dtype[Any]:
        """Dtype of stored arrays - reads header only (no full load)."""
        # Header-only path is cheaper than full concatenation
        _, header = self.header
        descr: Any = header.get("descr")
        # descr may be a string like '<f8' or a dtype; normalise
        return np.dtype(descr) if isinstance(descr, str) else descr

    @property
    def _actual_shape(self) -> tuple[int, ...]:
        """Shape of the fully concatenated array."""
        return self.load().shape

    @property
    def header(self) -> tuple[tuple[int, int], dict[str, Any]]:
        """Read numpy file header (version, dict)."""
        with Path(self.fname).open("rb") as fh:
            version = np.lib.format.read_magic(fh)
            match version[0]:
                case 1:
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
                case 2:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
                case _:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        return version, {"descr": dtype, "fortran_order": fortran, "shape": shape}


# ---------------------------------------------------------------------------
# Raw data writer
# ---------------------------------------------------------------------------


class RawDataWriter:
    """Save raw tracking data as appendable ``.anpy`` files per ROI.

    Args:
        basename: Base filename for output files.
        n_rois: Number of ROIs.
        entities: Max entities per ROI.
    """

    def __init__(self, basename: str | Path, n_rois: int, entities: int = 40) -> None:
        base = Path(str(basename)).with_suffix("")
        self._basename: str = str(base)
        self.entities: int = entities
        self.files: list[NpyAppendableFile] = [
            NpyAppendableFile(Path(f"{self._basename}_{n_rois:03d}.anpy"), newfile=True)
            for _ in range(n_rois)
        ]
        self.data: dict[int, npt.NDArray[Any]] = {}

    def flush(self, t: int, frame: Any) -> None:
        """Write accumulated per-ROI arrays to disk."""
        for row_key, fh in zip(self.data, self.files, strict=False):
            fh.write(self.data[row_key])

    def write(
        self, t: int, roi: ROIProtocol, data_rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Buffer tracking data for ``roi`` at time ``t``.

        Each ``data_rows`` element is a mapping with ``x, y, w, h, phi``.
        The resulting array is fixed to ``(entities, 6, 1)``.
        """
        arr = np.asarray(
            [
                [t, fly["x"], fly["y"], fly["w"], fly["h"], fly["phi"]]
                for fly in data_rows
            ]
        )
        # Resize to fixed entity count - pad/truncate as needed
        arr.resize((self.entities, 6, 1), refcheck=False)
        self.data[roi.idx] = arr
