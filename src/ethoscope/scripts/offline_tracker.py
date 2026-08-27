# ruff: noqa: E501, E402, I001, RUF100, BLE001, TRY300
"""
Offline sleep-tracking harness: video file -> SQLite DB via existing pipeline.

Re-uses the production bricks (MovieVirtualCamera + FileBasedROIBuilder +
Monitor + AdaptiveBGModel + SQLiteResultWriter) so a one-line edit to the
tracker / ROI template can be re-tested on a recorded video in seconds
instead of waiting days for a live experiment.

Canonical pattern mirrors ``ethoscope/__init__.py:38-73`` and
``tests/integration_api_tests/test_whole_api.py``; this file only adds
argparse, rethomics-friendly METADATA, and the NullDrawer / DefaultDrawer
choice.

Standalone usage (no pip install required)::

    # Default: rethomics-compliant hierarchy
    #   src/ethoscope/output/ethoscope_data/results/999offline.../ETHOSCOPE_999/YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_999offline....db
    python src/ethoscope/scripts/offline_tracker.py \\
        src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4 --verbose

    # Explicit results root directory (structured hierarchy created underneath)
    python src/ethoscope/scripts/offline_tracker.py video.mp4 --db /tmp/my_results --verbose

    # Historic flat file (escape hatch, still supported):
    python src/ethoscope/scripts/offline_tracker.py video.mp4 \\
        --db src/ethoscope/output/offline.db --verbose

    or

    cd src/ethoscope/
    python3 scripts/offline_tracker.py eth10.mp4 --drop-each 3 --video-out output/annotated.avi

Annotated video::

    python src/ethoscope/scripts/offline_tracker.py \\  # noqa: E501
        src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4 \\  # noqa: E501
        --video-out src/ethoscope/output/annotated.avi  # noqa: E501

With PYTHONPATH (dev)::

    PYTHONPATH=src/ethoscope python -m scripts.offline_tracker \\  # noqa: E501
        src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4 --verbose

Programmatic use::

    from scripts.offline_tracker import run_offline_tracking  # noqa: E501
    # Default structured path (rethomics):
    db = run_offline_tracking(  # noqa: E501
        "src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4", verbose=True)  # noqa: E501
    # Explicit file:
    db = run_offline_tracking(  # noqa: E501
        "src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4",  # noqa: E501
        "src/ethoscope/output/out.db", verbose=True)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

# Allow running as ``python src/ethoscope/scripts/offline_tracker.py`` without
# ``pip install`` — ensure ``src/ethoscope`` is on sys.path.  # noqa: E501
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # src/ethoscope
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ``ethoscope.hardware.input.cameras`` imports ``picamera2`` at module load.
# On dev/CI machines the Pi camera stack (libcamera/pykms) is missing and the
# import raises ``ModuleNotFoundError`` for ``pykms`` even though the
# ``picamera2`` package is installed.  ``src/ethoscope/conftest.py`` installs a
# stub for pytest; replicate it here so the offline harness works outside
# pytest (e.g. ``offline-tracker`` CLI).  Keep this block *before* any
# ``ethoscope`` imports.
try:
    __import__("picamera2")
except ImportError:
    for _k in [
        k for k in sys.modules if k == "picamera2" or k.startswith("picamera2.")
    ]:
        sys.modules.pop(_k, None)

    class _MappedArray:  # minimal stand-in for picamera2.MappedArray
        def __init__(self, request, stream):
            self.request = request
            self.stream = stream
            self.array = None

        def __enter__(self):
            return self

        def __exit__(self, *_args: object, **_kwargs: Any) -> None:
            pass

    class _Picamera2:  # minimal stand-in for picamera2.Picamera2
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(  # noqa: TRY003
                "picamera2 is not available in this environment; "
                "camera hardware tests are disabled"
            )

        @staticmethod
        def set_logging(level: Any) -> None:
            pass

        @staticmethod
        def load_tuning_file(path: Any) -> None:
            raise RuntimeError(  # noqa: TRY003
                "picamera2 is not available in this environment"
            )

    _stub = types.ModuleType("picamera2")
    _stub.MappedArray = _MappedArray  # type: ignore[attr-defined]
    _stub.Picamera2 = _Picamera2  # type: ignore[attr-defined]
    sys.modules["picamera2"] = _stub

from ethoscope.core.monitor import Monitor  # noqa: E402
from ethoscope.drawers.drawers import BaseDrawer, DefaultDrawer  # noqa: E402
from ethoscope.hardware.input.cameras import MovieVirtualCamera  # noqa: E402
from ethoscope.io import SQLiteResultWriter  # noqa: E402
from ethoscope.roi_builders.file_based_roi_builder import FileBasedROIBuilder  # noqa: E402
from ethoscope.trackers.adaptive_bg_tracker import AdaptiveBGModel  # noqa: E402

logger = logging.getLogger(__name__)


class _HeadlessDrawer(BaseDrawer):
    """Headless drawer that does no annotation, video writing or display.

    Works around ``NullDrawer`` signature mismatch (missing
    ``reference_points``) without editing ``drawers.py``.
    """

    def __init__(self) -> None:
        super().__init__(draw_frames=False)

    def _annotate_frame(
        self, img, positions, tracking_units, reference_points=None
    ) -> None:
        pass


_DEFAULT_ROI_TEMPLATE = "sleep_monitor_20tube"
_DEFAULT_VIDEO_DIR = _PROJECT_ROOT / "ethoscope" / "tests" / "static_files" / "videos"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
_REPO_ROOT = _PROJECT_ROOT.parent.parent  # repo root

# Offline rethomics identity — mirrors live ethoscope layout
#   /ethoscope_data/results/{machine_id}/{machine_name}/{timestamp}/{timestamp}_{machine_id}.db
_OFFLINE_MACHINE_ID = "999offlinexxxxxxxxxxxxxxxxxxxxxx"
_OFFLINE_MACHINE_NAME = "ETHOSCOPE_999"
_OFFLINE_RESULTS_SUBDIR = Path("ethoscope_data/results")


def _configure_cv_threads(n: int | None) -> int | None:
    """Configure OpenCV thread pool via ``cv2.setNumThreads``.

    Priority: explicit ``n`` > ``OPENCV_NUM_THREADS`` > ``OMP_NUM_THREADS`` >
    keep default. Must be called on the main thread *before* any
    ``parallel_for_`` region (i.e. before ``MovieVirtualCamera`` creation).

    Returns effective thread count or ``None`` if ``cv2`` unavailable.

    ``n`` semantics match ``cv2``: ``0``/``1`` disables parallelism, ``<0``
    resets to default (``getNumberOfCPUs``).
    """
    if cv2 is None:
        return None
    # Env-var fallback when n not explicitly given
    if n is None:
        env = os.getenv("OPENCV_NUM_THREADS") or os.getenv("OMP_NUM_THREADS")
        if env is not None:
            try:
                n = int(env)
            except ValueError:
                logger.warning("Invalid OPENCV/OMP_NUM_THREADS=%r, ignoring", env)
                n = None
    if n is not None:
        try:
            cv2.setNumThreads(int(n))
        except Exception as e:  # pragma: no cover
            logger.warning("cv2.setNumThreads(%r) failed: %s", n, e)
    try:
        eff = int(cv2.getNumThreads())
        cpus = int(cv2.getNumberOfCPUs())
        # log once at INFO to make effective setting visible in offline runs
        logger.info(
            "cv2 threads=%d/%d cpus=%d optimized=%s",
            eff,
            cpus,
            cpus,
            cv2.useOptimized(),
        )
        return eff
    except Exception:  # pragma: no cover
        return None


def _resolve_output_path(path: str | Path) -> Path:
    """Resolve output DB/video paths without producing double ``src/ethoscope``.

    User is instructed to pass ``src/ethoscope/output/...`` relative to the
    repo root. If ``cwd`` is ``src/ethoscope`` the naïve ``Path.cwd() / src/...``
    would create ``.../src/ethoscope/src/ethoscope/output``. We detect the
    ``src/`` prefix and resolve against the repo root instead.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # ``src/...`` paths are repo-root-relative by convention in this project
    if str(p).startswith("src/"):
        return _REPO_ROOT / p
    # otherwise cwd-relative (and will be made absolute below)
    return (Path.cwd() / p).resolve() if not p.exists() else p.resolve()


def _resolve_input_video(path: Path) -> Path:
    """Resolve ``input_video`` against the assumed videos folder if needed.

    If ``path`` exists as given, return it unchanged. Otherwise try
    ``_DEFAULT_VIDEO_DIR / path.name`` and ``_DEFAULT_VIDEO_DIR / path``.
    This lets ``arena_10x2_sortTubes.mp4`` resolve without typing the full
    ``src/ethoscope/ethoscope/tests/static_files/videos/`` prefix.
    """
    if path.exists():
        return path
    # Bare filename or relative path that exists under the default videos dir
    for candidate in (path.name, str(path)):
        alt = _DEFAULT_VIDEO_DIR / candidate
        if alt.exists():
            return alt
    # Also try resolving relative to project root (for cwd elsewhere)
    alt = _DEFAULT_VIDEO_DIR / path
    if alt.exists():
        return alt
    return path  # caller will raise FileNotFoundError


def _get_offline_timestamp_str(start_time: float) -> str:
    """Format *start_time* as ``YYYY-MM-DD_HH-MM-SS`` (24h)."""
    return datetime.datetime.fromtimestamp(start_time).strftime(  # noqa: DTZ006
        "%Y-%m-%d_%H-%M-%S"
    )


def _build_structured_db_path(
    timestamp_str: str,
    base_results_dir: Path | None = None,
) -> Path:
    """Return rethomics-compliant DB path for offline tracking.

    Layout mirrors live ethoscope (see ``tracking.py:899-914`` and
    ``scopr`` docs)::

        {base_results_dir}/{machine_id}/{machine_name}/{timestamp}/{timestamp}_{machine_id}.db

    ``base_results_dir`` is the rethomics *results root*. When ``None``,
    defaults to ``src/ethoscope/output/ethoscope_data/results`` which is the
    directory ``rethomics::loadSQLite`` / ``scopr::link_ethoscope_metadata``
    expects as its ``result_dir`` argument. When a custom directory is given
    (e.g. via ``--db /tmp/my_results``), it is treated as the results root
    directly — no extra ``ethoscope_data/results`` prefix is added.
    """
    if base_results_dir is None:
        base_results_dir = _DEFAULT_OUTPUT_DIR / _OFFLINE_RESULTS_SUBDIR
    else:
        base_results_dir = Path(base_results_dir)
        # If user passed a path that already contains the leaf structure
        # (e.g. .../ETHOSCOPE_999 or .../999offline...), avoid duplicating it.
        # Detect by checking if the last parts already match offline ids.
        parts = base_results_dir.parts
        if parts and parts[-1] == _OFFLINE_MACHINE_NAME:
            # base is already .../ETHOSCOPE_999 — strip to its parent's parent
            # so final path becomes .../999offline.../ETHOSCOPE_999/<ts>/...
            # but user explicitly pointing inside ETHOSCOPE_999 should just
            # get <ts>/<file> underneath. Handle as special case.
            return base_results_dir / timestamp_str / f"{timestamp_str}_{_OFFLINE_MACHINE_ID}.db"
        if parts and parts[-1] == _OFFLINE_MACHINE_ID:
            return base_results_dir / _OFFLINE_MACHINE_NAME / timestamp_str / f"{timestamp_str}_{_OFFLINE_MACHINE_ID}.db"
    return (
        base_results_dir
        / _OFFLINE_MACHINE_ID
        / _OFFLINE_MACHINE_NAME
        / timestamp_str
        / f"{timestamp_str}_{_OFFLINE_MACHINE_ID}.db"
    )


def _resolve_output_db(
    output_db: str | Path | None,
    timestamp_str: str,
) -> Path:
    """Resolve ``--db`` / ``output_db`` argument to a concrete DB file path.

    Rules (in order):
    1. ``None`` → default structured path (``output/ethoscope_data/results/...``).
    2. Path ending with ``.db`` that is exactly the historic default
       ``.../output/offline.db`` → treat as ``None`` (migrate to structured).
    3. Path ending with ``.db`` → use as explicit file path (legacy escape hatch).
    4. Otherwise treat as a *results root directory* and build structured path
       underneath it.
    """
    if output_db is None:
        return _build_structured_db_path(timestamp_str)

    p = Path(str(output_db))

    # Historic default flat file — migrate to structured rethomics layout
    # (both repo-root-relative and cwd-relative forms)
    default_flat = _DEFAULT_OUTPUT_DIR / "offline.db"
    try:
        # Compare after resolving via _resolve_output_path logic
        resolved = _resolve_output_path(p)
        if resolved == default_flat.resolve() or (
            p.name == "offline.db" and p.parent.name == "output"
        ):
            return _build_structured_db_path(timestamp_str)
        # Also handle "src/ethoscope/output/offline.db" form
        if str(p).endswith("output/offline.db"):
            return _build_structured_db_path(timestamp_str)
    except Exception:  # noqa: S110
        pass

    # Explicit file path (escape hatch)
    if p.suffix.lower() == ".db":
        return _resolve_output_path(p)

    # Directory → treat as results root for structured generation
    base_dir = _resolve_output_path(p)
    return _build_structured_db_path(timestamp_str, base_results_dir=base_dir)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_metadata_from_cli(
    metadata_json: str | None,
    metadata_file: str | None,
) -> dict[str, Any]:
    """Merge --metadata JSON string and --metadata-file into a single dict."""
    meta: dict[str, Any] = {}
    if metadata_file:
        p = Path(metadata_file)
        if not p.exists():
            raise FileNotFoundError(f"Metadata file not found: {p}")  # noqa: TRY003
        with p.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            msg = f"Metadata file must contain a JSON object, got {type(data)}"
            raise ValueError(msg)
        meta.update(data)
    if metadata_json:
        try:
            data = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"--metadata is not valid JSON: {e}") from e  # noqa: TRY003
        if not isinstance(data, dict):
            raise ValueError("--metadata must be a JSON object (dict)")  # noqa: TRY003
        meta.update(data)
    return meta


def _build_default_metadata(  # noqa: PLR0913, PLR0917
    camera: Any,
    reference_points: Any,
    roi_template: str,
    input_video: Path,
    output_db: Path,
    extra: dict[str, Any] | None = None,
    start_time: float | None = None,
    timestamp_str: str | None = None,  # noqa: ARG001 — kept for future templating / explicit override
) -> dict[str, Any]:
    """Build rethomics-friendly METADATA expected in ``METADATA`` table.

    Mirrors the keys set in ``ethoscope/control/tracking.py:920-934`` so the
    resulting DB is directly loadable with ``rethomics::loadSQLite``.
    ``extra`` keys override defaults (user-supplied --metadata).

    ``start_time`` is the tracking start epoch (seconds). When ``None``,
    ``time.time()`` is used. ``output_db`` should already be the final
    structured path (``.../ETHOSCOPE_999/YYYY-MM.../YYYY-MM..._999...db``) so
    ``backup_filename`` / ``sqlite_source_path`` match the file on disk and
    the live-ethoscope convention.
    """
    now = start_time if start_time is not None else time.time()
    meta: dict[str, Any] = {
        "machine_id": _OFFLINE_MACHINE_ID,
        "machine_name": _OFFLINE_MACHINE_NAME,
        "date_time": now,
        "frame_width": int(getattr(camera, "width", 0) or 0),
        "frame_height": int(getattr(camera, "height", 0) or 0),
        "version": "offline-1.0",
        "experimental_info": json.dumps(
            {"name": "offline", "location": "offline", "roi_template": roi_template}
        ),
        "selected_options": json.dumps(
            {"roi_template": roi_template, "input_video": str(input_video)}
        ),
        "reference_points": str(
            [
                (int(p[0]), int(p[1]))
                for p in (reference_points if reference_points is not None else [])
            ]  # noqa: E501
        ),
        "backup_filename": output_db.name,
        "result_writer_type": "SQLite3",
        "sqlite_source_path": str(output_db),
        "input_video": str(input_video),
        "roi_template": roi_template,
    }
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# core harness
# ---------------------------------------------------------------------------


def run_offline_tracking(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915
    input_video: str | Path,
    output_db: str | Path | None = None,
    roi_template: str = _DEFAULT_ROI_TEMPLATE,
    template_file: str | Path | None = None,
    template_id: str | None = None,
    template_data: dict[str, Any] | None = None,
    output_video: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    metadata_json: str | None = None,
    metadata_file: str | Path | None = None,
    draw_frames: bool = False,
    verbose: bool = False,
    drop_each: int | None = None,
    max_duration: float | None = None,
    erase_old_db: bool = True,
    take_frame_shots: bool = False,
    make_dam_like_table: bool = False,
    cv_threads: int | None = None,
) -> Path:
    """Run sleep tracking on *input_video* and write results to *output_db*.

    This is the importable entry point; the CLI ``main()`` delegates here.
    When ``output_db`` is ``None`` (default) the DB is written to a
    rethomics-compliant hierarchy::

        src/ethoscope/output/ethoscope_data/results/
            999offlinexxxxxxxxxxxxxxxxxxxxxx/ETHOSCOPE_999/
                YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_999offline....db

    where ``YYYY-MM-DD_HH-MM-SS`` is the wall-clock time at which offline
    tracking started. This matches the live ethoscope layout
    (``tracking.py:899-914``) and is directly discoverable by
    ``rethomics::loadSQLite`` / ``scopr::link_ethoscope_metadata`` when
    ``result_dir`` is ``.../output/ethoscope_data/results``.

    Args:
        input_video: Path to video file readable by OpenCV (mp4/avi/mkv).
            Raw ``.h264`` chunk directories are not supported directly — merge
            them first with ``accessories/h264_to_mp4.py``.
        output_db: Path to SQLite file to create (parents are created; existing
            file is removed when ``erase_old_db`` is True — default for offline
            re-runs). The DB contains ``ROI_<idx>``, ``METADATA``, ``VAR_MAP``
            and is loadable with ``rethomics``. When ``None`` (default) a
            timestamped path inside ``ETHOSCOPE_999`` is auto-generated (see
            above). When a *directory* is given it is treated as the rethomics
            ``results`` root and the same hierarchy is created underneath it.
            When an explicit ``.db`` file is given it is used verbatim (legacy
            escape hatch). The historic default ``.../output/offline.db`` is
            automatically migrated to the structured layout.
        roi_template: Builtin template name (default ``sleep_monitor_20tube``).
            Ignored when ``template_file`` / ``template_id`` / ``template_data``
            is given. Available builtins live in
            ``ethoscope/roi_builders/roi_templates/builtin/``.
        template_file: Path to a custom ROI JSON template.
        template_id: MD5 id of a template managed by ``ROITemplateManager``.
        template_data: Inline template dict (programmatic use).
        output_video: When set, annotated tracking is written to this ``.avi``
            via ``DefaultDrawer``; otherwise ``NullDrawer`` is used (headless,
            faster, no GUI — recommended for rapid iteration).
        metadata: Dict of extra ``METADATA`` key/values (merged over defaults).
            Use for ``rethomics`` grouping (e.g. ``{"treatment":"drugA"}``).
        metadata_json: JSON object string merged over ``metadata``.
        metadata_file: Path to JSON file merged over ``metadata``.
        draw_frames: Also pop up a live window (requires display). Only
            meaningful with ``output_video`` or for debugging; ignored for
            ``NullDrawer``.
        verbose: Print progress timestamps (``Monitor.run(verbose=True)``).
        drop_each: Forwarded to ``MovieVirtualCamera`` (process every Nth frame).
        max_duration: Forwarded to ``MovieVirtualCamera`` (seconds).
        erase_old_db: Whether to delete an existing DB at ``output_db``.
        take_frame_shots: Store periodic frame snapshots (``IMG_SNAPSHOTS``).
        make_dam_like_table: Create DAM-compatible activity table.
        cv_threads: OpenCV thread pool size via ``cv2.setNumThreads``.
            ``0``/``1`` disables, ``<0`` resets to default. ``None`` keeps
            default unless ``OPENCV_NUM_THREADS``/``OMP_NUM_THREADS`` env is set.

    Returns:
        Path to the written DB (same as ``output_db`` or auto-generated).

    Raises:
        FileNotFoundError: If input_video or template/metadata file is missing.
        ValueError: On invalid template / metadata arguments.
        EthoscopeException: If video cannot be opened.
    """
    # Capture wall-clock start *before* any heavy work so folder / filename
    # and METADATA ``date_time`` are identical to live-ethoscope convention.
    tracking_start_time = time.time()
    timestamp_str = _get_offline_timestamp_str(tracking_start_time)

    input_video_p = _resolve_input_video(Path(input_video))
    output_db_p = _resolve_output_db(output_db, timestamp_str)

    if not input_video_p.exists():
        raise FileNotFoundError(  # noqa: TRY003
            f"Input video not found: {input_video_p} "
            f"(also checked {_DEFAULT_VIDEO_DIR})"
        )
    # Hint for common user error: raw h264 chunks
    if input_video_p.suffix.lower() == ".h264" and not input_video_p.is_dir():
        logger.warning(
            "Input is a single .h264 chunk. For a full experiment merge "
            "chunks first: python accessories/h264_to_mp4.py <results_dir> "
            "-o /tmp/merged.mp4"
        )

    # Resolve ROI builder args — exactly one source wins
    # (priority mirrors FileBasedROIBuilder)
    roi_kwargs: dict[str, Any] = {}
    if template_data is not None:
        roi_kwargs["template_data"] = template_data
    elif template_file is not None:
        p = Path(template_file)
        if not p.exists():
            raise FileNotFoundError(f"Template file not found: {p}")  # noqa: TRY003
        roi_kwargs["template_file"] = str(p)
    elif template_id is not None:
        roi_kwargs["template_id"] = template_id
    else:
        roi_kwargs["template_name"] = roi_template

    # Resolve metadata merging (explicit dict < file < json string, so CLI can override)
    cli_extra = _load_metadata_from_cli(
        metadata_json, str(metadata_file) if metadata_file else None
    )
    merged_extra: dict[str, Any] = {}
    if metadata:
        merged_extra.update(metadata)
    merged_extra.update(cli_extra)

    # Configure OpenCV threading before any parallel_for_ region
    _configure_cv_threads(cv_threads)

    # Camera and ROIs — builder consumes ~6 frames for
    # median/target detection, so restart is mandatory
    cam_kwargs: dict[str, Any] = {}
    if drop_each is not None:
        cam_kwargs["drop_each"] = drop_each
    if max_duration is not None:
        cam_kwargs["max_duration"] = max_duration

    cam = MovieVirtualCamera(str(input_video_p), **cam_kwargs)
    roi_builder = FileBasedROIBuilder(**roi_kwargs)  # type: ignore[arg-type]
    reference_points, rois = roi_builder.build(cam)
    cam.restart()

    # Ensure DB parent exists (rethomics hierarchy is created here)
    output_db_p.parent.mkdir(parents=True, exist_ok=True)

    default_meta = _build_default_metadata(
        cam,
        reference_points,
        roi_kwargs.get("template_name", roi_template),
        input_video_p,
        output_db_p,
        merged_extra,
        start_time=tracking_start_time,
        timestamp_str=timestamp_str,
    )

    # Drawer choice: headless vs annotated video
    if output_video is not None:
        out_vid = _resolve_output_path(Path(output_video))
        out_vid.parent.mkdir(parents=True, exist_ok=True)
        drawer: Any = DefaultDrawer(video_out=str(out_vid), draw_frames=draw_frames)
    else:
        if draw_frames:
            logger.warning(
                "--draw-frames without --video-out has no effect "
                "(headless drawer ignores it)."
            )
        drawer = _HeadlessDrawer()

    # Monitor — AdaptiveBGModel is the sleep-tracking pipeline
    monitor = Monitor(cam, AdaptiveBGModel, rois, reference_points=reference_points)

    # SQLiteResultWriter expects {"name": path}; erase_old_db=True is offline default
    db_credentials = {"name": str(output_db_p)}
    with SQLiteResultWriter(
        db_credentials,
        rois,
        metadata=default_meta,
        erase_old_db=erase_old_db,
        take_frame_shots=take_frame_shots,
        make_dam_like_table=make_dam_like_table,
    ) as rw:
        monitor.run(result_writer=rw, drawer=drawer, verbose=verbose)

    # Drawer cleanup: DefaultDrawer holds a VideoWriter
    with contextlib.suppress(Exception):
        if hasattr(drawer, "_video_writer") and drawer._video_writer is not None:
            drawer._video_writer.release()
    with contextlib.suppress(Exception):
        cam._close()

    logger.info(
        "Offline tracking complete: %s -> %s (%d ROIs, %d frames)",
        input_video_p,
        output_db_p,
        len(rois),
        monitor.last_frame_idx + 1,
    )
    return output_db_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline sleep tracking: video file -> SQLite DB.",
        epilog=(
            "Examples (videos in src/ethoscope/ethoscope/tests/static_files/videos/,\n"  # noqa: E501
            " output in src/ethoscope/output/ethoscope_data/results/...):\n"
            "  python src/ethoscope/scripts/offline_tracker.py arena_10x2_sortTubes.mp4 --verbose\n"  # noqa: E501
            "  python src/ethoscope/scripts/offline_tracker.py "  # noqa: E501
            "src/ethoscope/ethoscope/tests/static_files/videos/arena_10x2_sortTubes.mp4 --verbose\n"  # noqa: E501
            "  python src/ethoscope/scripts/offline_tracker.py eth10.mp4 "  # noqa: E501
            "--db /tmp/my_results --video-out /tmp/annotated.avi\n"  # noqa: E501
            "  python src/ethoscope/scripts/offline_tracker.py eth10.mp4 "  # noqa: E501
            '--roi-template sleep_monitor_30tube --metadata \'{"treatment":"drug"}\'\n'  # noqa: E501
            "  # Historic flat file (escape hatch): --db src/ethoscope/output/offline.db\n"  # noqa: E501
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input_video",
        help="Path to input video file (mp4/avi/mkv). Bare filenames are resolved "
        "against src/ethoscope/ethoscope/tests/static_files/videos/. For .h264 chunks "
        "merge with accessories/h264_to_mp4.py first.",
    )
    p.add_argument(
        "--db",
        "--output-db",
        dest="output_db",
        required=False,
        default=None,
        help="Path to output SQLite DB file or results root directory. "  # noqa: E501
        "When omitted (default) a rethomics-compliant hierarchy is created at "  # noqa: E501
        "src/ethoscope/output/ethoscope_data/results/999offline.../ETHOSCOPE_999/"  # noqa: E501
        "YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_999offline...db where "  # noqa: E501
        "YYYY-MM-DD_HH-MM-SS is the wall-clock start of this run. "  # noqa: E501
        "When a directory is given it is treated as the results root. "  # noqa: E501
        "When an explicit .db file is given it is used verbatim. "  # noqa: E501
        "Historic default src/ethoscope/output/offline.db is auto-migrated to the "  # noqa: E501
        "structured layout. Parents are created.",
    )
    p.add_argument(
        "--roi-template",
        default=_DEFAULT_ROI_TEMPLATE,
        help="Builtin ROI template name (default: %(default)s). "
        "See ethoscope/roi_builders/roi_templates/builtin/.",
    )
    p.add_argument(
        "--template-file",
        dest="template_file",
        default=None,
        help="Path to custom ROI template JSON (overrides --roi-template).",
    )
    p.add_argument(
        "--template-id",
        default=None,
        help="MD5 id of a template via ROITemplateManager (overrides --roi-template).",
    )
    p.add_argument(
        "--video-out",
        "--output-video",
        dest="output_video",
        default=None,
        help="When set, write annotated video to this .avi path (DefaultDrawer). "
        "Omit for headless NullDrawer.",
    )
    p.add_argument(
        "--draw-frames",
        action="store_true",
        help="Also pop up live tracking window (requires display; "
        "implies --video-out is useful).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress timestamps every 5 s.",
    )
    p.add_argument(
        "--drop-each",
        type=int,
        default=None,
        help="Process every Nth frame (forwarded to MovieVirtualCamera).",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max duration in seconds (forwarded to MovieVirtualCamera).",
    )
    p.add_argument(
        "--cv-threads",
        type=int,
        default=None,
        help="OpenCV thread pool size via cv2.setNumThreads. 0/1 disables, "
        "<0 resets to default. Env OPENCV_NUM_THREADS/OMP_NUM_THREADS respected.",
    )
    p.add_argument(
        "--metadata",
        dest="metadata_json",
        default=None,
        help='Extra METADATA as JSON object string, e.g. \'{"treatment":"drugA"}\' '
        "(merged over defaults, rethomics-visible).",
    )
    p.add_argument(
        "--metadata-file",
        default=None,
        help="Path to JSON file containing extra METADATA dict (merged over defaults).",
    )
    p.add_argument(
        "--keep-db",
        action="store_true",
        help="Do not erase existing DB (default erases).",
    )
    p.add_argument(
        "--take-frame-shots",
        action="store_true",
        help="Store periodic frame snapshots (IMG_SNAPSHOTS).",
    )
    p.add_argument(
        "--make-dam-like-table",
        action="store_true",
        help="Create DAM-compatible activity table.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        run_offline_tracking(
            input_video=args.input_video,
            output_db=args.output_db,
            roi_template=args.roi_template,
            template_file=args.template_file,
            template_id=args.template_id,
            output_video=args.output_video,
            metadata_json=args.metadata_json,
            metadata_file=args.metadata_file,
            draw_frames=args.draw_frames,
            verbose=args.verbose,
            drop_each=args.drop_each,
            max_duration=args.max_duration,
            erase_old_db=not args.keep_db,
            take_frame_shots=args.take_frame_shots,
            make_dam_like_table=args.make_dam_like_table,
            cv_threads=args.cv_threads,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        logger.exception("Offline tracking failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
