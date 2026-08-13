"""
Video Utilities Module

This module provides utility functions for video file management including
file listing, video file indexing operations, and migration utilities
for directory structure updates.
"""

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

type VideoInfo = dict[str, str]
type VideoIndex = dict[str, VideoInfo]

_VIDEO_SUFFIXES: tuple[str, ...] = (".h264",)


def _iter_video_files(rootdir: str | Path) -> Iterator[Path]:
    """Yield video file paths below ``rootdir`` matching the known suffixes.

    Args:
        rootdir: Root directory to scan recursively.

    Yields:
        A :class:`~pathlib.Path` for every video file found.

    Notes:
        Unreadable files encountered during inspection (e.g., races, stat
        failures) are skipped with a warning; siblings continue to be
        visited. Directory-level enumeration errors still abort the scan.
    """
    root = Path(rootdir)
    for path in root.rglob("*"):
        try:
            if path.is_file() and path.suffix in _VIDEO_SUFFIXES:
                yield path
        except OSError as exc:
            logger.warning("Skipping unreadable path %s: %s", path, exc)


def list_local_video_files(
    rootdir: str | Path,
    _createMD5: bool = False,  # kept for API compatibility
) -> VideoIndex:
    """Create an index of all video files below ``rootdir``.

    Scans ``rootdir`` and its subdirectories for video files with the known
    formats and returns a mapping of basename to file information.

    Args:
        rootdir: Root directory to scan for video files.
        createMD5: Deprecated and ignored (hashing is no longer used);
            retained for backwards compatibility.

    Returns:
        A dictionary with video file basenames as keys and their info as
        values, e.g. ``{"video1.h264": {"path": "/path/to/video1.h264"}}``.

    Raises:
        IOError: If there is an issue accessing the video files.
    """

    result: VideoIndex = {}
    for path in _iter_video_files(rootdir):
        filename = path.name
        if filename in result:
            logger.warning(
                "Duplicate video filename '%s' (already indexed from %s); overwriting.",
                filename,
                result[filename]["path"],
            )
        result[filename] = {"path": str(path)}
    return result


def ensure_video_directory_structure(
    ethoscope_root_dir: str | Path,
    videos_dir: str | Path,
) -> str:
    """Ensure the videos directory exists, migrating legacy data if needed.

    If a legacy ``<ethoscope_root_dir>/results`` directory exists and
    ``videos_dir`` does not, the legacy directory is moved to ``videos_dir``.
    Finally, ``videos_dir`` is created if it is still missing.

    Args:
        ethoscope_root_dir: Root directory that may contain a legacy
            ``results`` directory.
        videos_dir: Target videos directory to create or migrate to.

    Returns:
        The ``videos_dir`` path as a string.
    """
    legacy_results_dir = Path(ethoscope_root_dir) / "results"
    videos_path = Path(videos_dir)

    if legacy_results_dir.exists() and not videos_path.exists():
        try:
            _ = shutil.move(str(legacy_results_dir), str(videos_path))
        except OSError:
            logger.exception("Failed to move %s to %s", legacy_results_dir, videos_path)
        else:
            logger.info("Migrated %s to %s", legacy_results_dir, videos_path)

    if not videos_path.exists():
        videos_path.mkdir(parents=True, exist_ok=True)
        logger.info("Created videos directory: %s", videos_path)

    return str(videos_path)
