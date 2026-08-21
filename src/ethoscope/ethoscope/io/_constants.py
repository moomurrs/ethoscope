"""Central constants for ethoscope I/O subsystem.

This module centralises magic numbers and timeouts so that callers import
a single source of truth instead of duplicating literals across helpers,
base writers and cache layers.
"""

from __future__ import annotations

from typing import Final

ASYNC_WRITER_TIMEOUT: Final[float] = 30.0
SENSOR_DEFAULT_PERIOD: Final[float] = 120.0
IMG_SNAPSHOT_DEFAULT_PERIOD: Final[float] = 300.0
DAM_DEFAULT_PERIOD: Final[float] = 60.0

MAX_DB_RETRIES: Final[int] = 3
RETRY_BASE_DELAY: Final[float] = 1.0
MAX_RETRY_DELAY: Final[float] = 30.0
MAX_BUFFERED_COMMANDS: Final[int] = 10000
METADATA_MAX_VALUE_LENGTH: Final[int] = 60000
QUEUE_CHECK_INTERVAL: Final[float] = 0.1

# Cache / DB helpers
DAM_SCALE: Final[int] = 100
SQLITE_BATCH_SIZE: Final[int] = 50
BUFFERED_COMMAND_MAX_AGE: Final[float] = 300.0  # seconds
RESTART_THROTTLE_SECONDS: Final[float] = 30.0
CACHE_TTL_SECONDS: Final[float] = 30.0
MAX_BUFFERED_RETRY_FAILURES: Final[int] = 10
MIN_DB_SIZE_BYTES: Final[int] = 32768
