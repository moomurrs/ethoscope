"""
Under-voltage detection for Raspberry Pi.

Reads the rpi_volt hwmon alarm. Requires Linux kernel >= 5.15
(Debian Bookworm or newer / Raspberry Pi OS Bullseye or newer).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

HWMON_NAME = "rpi_volt"
HWMON_DIR = Path("/sys/class/hwmon")
HWMON_FILE_NAME = "in0_lcrit_alarm"


class UnderVoltage(ABC):
    """Abstract reader for the Raspberry Pi under-voltage status."""

    @abstractmethod
    def get(self) -> bool:
        """Return True if the kernel currently reports under voltage."""
        ...


class UnderVoltageNew(UnderVoltage):
    """Read the current under-voltage status from the modern hwmon entry."""

    def __init__(self, hwmon: Path) -> None:
        """Initialize the reader.

        Args:
            hwmon: Path to the rpi_volt hwmon device directory.
        """
        if not hwmon.is_dir():
            raise NotADirectoryError(f"{hwmon}")
        self._hwmon: Path = hwmon

    @override
    def get(self) -> bool:
        """Return True if the kernel currently reports under voltage."""
        return (self._hwmon / HWMON_FILE_NAME).read_text().strip() == "1"


def _find_rpi_volt_hwmon() -> Path | None:
    """Locate the rpi_volt hwmon directory, or None if unavailable.

    Returns:
        Path to the hwmon device directory, or None if absent.
    """
    try:
        entries = HWMON_DIR.iterdir()
    except OSError:
        return None

    for entry in entries:
        name_file = entry / "name"
        if not name_file.is_file():
            continue
        try:
            if name_file.read_text().strip() == HWMON_NAME:
                return entry
        except OSError:
            continue
    return None


def powerChecker() -> UnderVoltage | None:
    """Return the system UnderVoltage reader, or None if not available.

    Returns:
        An UnderVoltage reader, or None if no rpi_volt hwmon device exists.
    """
    hwmon = _find_rpi_volt_hwmon()
    if hwmon is None:
        return None
    return UnderVoltageNew(hwmon)
