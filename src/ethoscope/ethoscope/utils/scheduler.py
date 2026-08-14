import datetime
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import cast

_LOGGER = logging.getLogger(__name__)

type DateRange = tuple[float, float]

_HOURS_PER_DAY = 24
_MAX_INTERVAL_HOURS = 168
_SECONDS_PER_DAY = 86400
_SECONDS_PER_HOUR = 3600
_MAX_DATE_PARTS = 2

_EMPTY_DATE_PATTERN = re.compile(r"^\s*$")
_DATE_PATTERN = re.compile(
    r"^\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\s*$"
)
_DATE_SEPARATOR_PATTERN = re.compile(r"\s*>\s*")


class DateRangeError(Exception):
    """Base error raised when a scheduler date range is invalid."""


class OverlappingDateRangesError(DateRangeError):
    """Raised when two or more date ranges overlap."""

    def __init__(self) -> None:
        super().__init__("Some date ranges overlap")


class MultipleDateSeparatorError(DateRangeError):
    """Raised when a date range contains more than one ``>`` separator."""

    def __init__(self) -> None:
        super().__init__("found several '>' symbols; only one is allowed")


class EmptyDateRangeError(DateRangeError):
    """Raised when a date range has neither a start nor an end date."""

    def __init__(self) -> None:
        super().__init__("Data range cannot include two None dates")


class InvalidDateStringError(DateRangeError):
    """Raised when a date string does not match the expected format."""

    date_str: str

    def __init__(self, date_str: str) -> None:
        self.date_str = date_str
        super().__init__(f"{date_str} not match the expected pattern")


class InvalidDateValueError(DateRangeError):
    """Raised when a well-formed date string represents an invalid date."""

    datestr: str

    def __init__(self, datestr: str, exc: ValueError | OverflowError) -> None:
        self.datestr = datestr
        super().__init__(f"Invalid date format: {datestr} ({exc!s})")


class InvalidDateRangeOrderError(DateRangeError):
    """Raised when a date range's end date precedes its start date."""

    date_range_str: str

    def __init__(self, date_range_str: str) -> None:
        self.date_range_str = date_range_str
        super().__init__(
            f"Error in date {date_range_str}, the end date appears to be in the past"
        )


class UnexpectedDateStringError(DateRangeError):
    """Raised when a date range cannot be parsed into a valid structure."""

    def __init__(self) -> None:
        super().__init__("Unexpected date string")


class DailyScheduleError(Exception):
    """Base error raised when daily schedule parameters are invalid."""


class DailyDurationRangeError(DailyScheduleError):
    """Raised when ``daily_duration_hours`` is out of the valid range."""

    def __init__(self) -> None:
        super().__init__("daily_duration_hours must be between 0 and 24")


class IntervalRangeError(DailyScheduleError):
    """Raised when ``interval_hours`` is out of the valid range."""

    def __init__(self) -> None:
        super().__init__("interval_hours must be between 0 and 168")


class DurationExceedsIntervalError(DailyScheduleError):
    """Raised when the daily duration exceeds the interval."""

    def __init__(self) -> None:
        super().__init__("daily_duration_hours cannot exceed interval_hours")


class InvalidTimeFormatError(DailyScheduleError):
    """Raised when a daily start time is not in ``HH:MM:SS`` format."""

    time_str: str

    def __init__(self, time_str: str) -> None:
        self.time_str = time_str
        super().__init__(f"Invalid time format: {time_str}. Expected HH:MM:SS")


class Scheduler:
    """Express time constraints as a list of allowed date ranges.

    Parses a formatted string into a list of allowed time ranges and can then be
    used to assess whether a date and time falls within a valid range. This is
    useful to control stimulators and other utilities.

    Args:
        in_str: A formatted string. Format described `here <https://github.com/gilestrolab/ethoscope/blob/master/user_manual/schedulers.md>`_.
    """

    def __init__(self, in_str: str) -> None:
        self._date_ranges: list[DateRange] = [
            self._parse_date_range(part) for part in in_str.split(",")
        ]
        self._check_date_ranges(self._date_ranges)

    def _check_date_ranges(self, ranges: list[DateRange]) -> None:
        """Ensure date ranges are strictly increasing and non-overlapping.

        Args:
            ranges: The list of ``(start, end)`` date ranges to validate.
        """
        all_dates = [ts for start, end in ranges for ts in (start, end)]
        if any(all_dates[i + 1] - all_dates[i] <= 0 for i in range(len(all_dates) - 1)):
            raise OverlappingDateRangesError()

    def check_time_range(self, t: float | None = None) -> bool:
        """Return whether a unix timestamp is within an allowed range.

        Args:
            t: The time to test. When ``None``, the system time is used.

        Returns:
            ``True`` if the time was in range, ``False`` otherwise.
        """
        return self._in_range(time.time() if t is None else t)

    def get_schedule_state(self, t: float | None = None) -> str:
        """Return the current scheduling state for visual feedback.

        Args:
            t: The time to test. When ``None``, the system time is used.

        Returns:
            ``"scheduled"`` if within range, ``"inactive"`` otherwise.
        """
        return "scheduled" if self.check_time_range(t) else "inactive"

    def _in_range(self, t: float) -> bool:
        return any(start < t < end for start, end in self._date_ranges)

    def _parse_date_range(self, date_range_str: str) -> DateRange:
        dates = _DATE_SEPARATOR_PATTERN.split(date_range_str)

        if len(dates) > _MAX_DATE_PARTS:
            raise MultipleDateSeparatorError()

        date_strs = [self._parse_date(part) for part in dates]

        match date_strs:
            case [single]:
                start_date, end_date = single, None
            case [first, second]:
                if first is None and second is None:
                    raise EmptyDateRangeError()
                start_date, end_date = first, second
            case _:
                raise UnexpectedDateStringError()

        start = 0.0 if start_date is None else start_date
        end = math.inf if end_date is None else end_date

        if start >= end:
            raise InvalidDateRangeOrderError(date_range_str)
        return start, end

    def _parse_date(self, date_str: str) -> float | None:
        if _EMPTY_DATE_PATTERN.match(date_str):
            return None

        match = _DATE_PATTERN.match(date_str)
        if match is None:
            raise InvalidDateStringError(date_str)

        datestr = match.group("date")
        try:
            parsed = datetime.datetime.strptime(  # noqa: DTZ007
                datestr, "%Y-%m-%d %H:%M:%S"
            )
            return time.mktime(parsed.timetuple())
        except (ValueError, OverflowError) as e:
            raise InvalidDateValueError(datestr, e) from e


class DailyScheduler:
    """Enhanced scheduler for daily time-restricted operations.

    Supports operations that run for a fixed number of hours per day at
    specified intervals, designed for sleep restriction experiments that
    inherit from mAGO stimulators.

    Args:
        daily_duration_hours: Total hours active per day.
        interval_hours: Hours between the start of active periods.
        daily_start_time: Daily start time in ``HH:MM:SS`` format.
        state_file_path: Path to the state persistence file (optional).

    Example:
        >>> DailyScheduler(8, 24, "09:00:00")
        >>> DailyScheduler(4, 12, "06:00:00")
    """

    def __init__(
        self,
        daily_duration_hours: float,
        interval_hours: float = _HOURS_PER_DAY,
        daily_start_time: str = "00:00:00",
        state_file_path: str | None = None,
    ) -> None:
        if daily_duration_hours <= 0 or daily_duration_hours > _HOURS_PER_DAY:
            raise DailyDurationRangeError()

        if interval_hours <= 0 or interval_hours > _MAX_INTERVAL_HOURS:
            raise IntervalRangeError()

        if daily_duration_hours > interval_hours:
            raise DurationExceedsIntervalError()

        self._daily_duration_hours: float = daily_duration_hours
        self._interval_hours: float = interval_hours
        self._daily_start_time: str = daily_start_time
        self._state_file_path: str | None = state_file_path

        self._start_time_seconds: int = self._parse_time_string(daily_start_time)
        self._state: dict[str, object] = self._load_state() if state_file_path else {}

        _LOGGER.info(
            "DailyScheduler initialized: %sh active every %sh starting at %s",
            daily_duration_hours,
            interval_hours,
            daily_start_time,
        )

    def _parse_time_string(self, time_str: str) -> int:
        """Parse a ``HH:MM:SS`` string into seconds since midnight.

        Args:
            time_str: Time in ``HH:MM:SS`` format.

        Returns:
            Seconds since midnight.
        """
        try:
            time_obj = datetime.time.fromisoformat(time_str)
        except ValueError as e:
            raise InvalidTimeFormatError(time_str) from e
        return (
            time_obj.hour * _SECONDS_PER_HOUR + time_obj.minute * 60 + time_obj.second
        )

    def _load_state(self) -> dict[str, object]:
        """Load the scheduler state from file, if present."""
        if not self._state_file_path:
            return {}

        state_path = Path(self._state_file_path)
        if not state_path.exists():
            return {}

        try:
            data = cast("dict[str, object]", json.loads(state_path.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            _LOGGER.warning("Could not load scheduler state: %s", e)
            return {}
        return data

    def _save_state(self) -> None:
        """Save the scheduler state to file."""
        if not self._state_file_path:
            return

        state_path = Path(self._state_file_path)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            _ = state_path.write_text(json.dumps(self._state, indent=2))
        except OSError:
            _LOGGER.exception("Could not save scheduler state")

    def _daily_anchor(self, t: float) -> float:
        """Return the epoch timestamp of today's period anchor for ``t``."""
        days_since_epoch = int(t // _SECONDS_PER_DAY)
        return days_since_epoch * _SECONDS_PER_DAY + self._start_time_seconds

    def _current_period_bounds(self, t: float) -> DateRange:
        """Return the ``(start, end)`` bounds of the period containing ``t``."""
        interval_seconds = self._interval_hours * _SECONDS_PER_HOUR
        active_seconds = self._daily_duration_hours * _SECONDS_PER_HOUR
        start_timestamp = self._daily_anchor(t)

        periods_since_start = int((t - start_timestamp) // interval_seconds)
        if t < start_timestamp:
            periods_since_start = -1

        current_period_start = start_timestamp + periods_since_start * interval_seconds
        return current_period_start, current_period_start + active_seconds

    def _record_period_activity(self, start: float, end: float, t: float) -> None:
        """Record the first observed activity within a period, if not yet seen."""
        period_key = f"period_{int(start)}"
        if period_key not in self._state:
            self._state[period_key] = {
                "start_time": start,
                "end_time": end,
                "first_activity": t,
            }
            self._save_state()

    def _isoformat_local(self, t: float) -> str:
        """Return a naive local-time ISO string for a unix timestamp."""
        return datetime.datetime.fromtimestamp(t).isoformat()  # noqa: DTZ006

    def is_active_period(self, t: float | None = None) -> bool:
        """Return whether the current time is within an active period.

        Args:
            t: Unix timestamp to check. If ``None``, uses the current time.

        Returns:
            ``True`` if within an active period, ``False`` otherwise.
        """
        t = time.time() if t is None else t
        start, end = self._current_period_bounds(t)
        is_active = start <= t < end
        if is_active and self._state_file_path:
            self._record_period_activity(start, end, t)
        return is_active

    def get_next_active_period(self, t: float | None = None) -> DateRange:
        """Return the ``(start, end)`` timestamps of the next active period.

        Args:
            t: Reference timestamp. If ``None``, uses the current time.

        Returns:
            ``(start_timestamp, end_timestamp)`` of the next active period.
        """
        t = time.time() if t is None else t
        interval_seconds = self._interval_hours * _SECONDS_PER_HOUR
        active_seconds = self._daily_duration_hours * _SECONDS_PER_HOUR
        start_timestamp = self._daily_anchor(t)

        periods_passed = (
            int((t - start_timestamp) // interval_seconds) + 1
            if t >= start_timestamp
            else 0
        )
        next_start = start_timestamp + periods_passed * interval_seconds
        return next_start, next_start + active_seconds

    def get_time_until_next_period(self, t: float | None = None) -> float:
        """Return the number of seconds until the next active period starts.

        Args:
            t: Reference timestamp. If ``None``, uses the current time.

        Returns:
            Seconds until the next active period.
        """
        t = time.time() if t is None else t
        next_start, _ = self.get_next_active_period(t)
        return max(0.0, next_start - t)

    def get_remaining_active_time(self, t: float | None = None) -> float:
        """Return the remaining seconds in the current active period.

        Args:
            t: Reference timestamp. If ``None``, uses the current time.

        Returns:
            Remaining seconds in the active period, or ``0`` if inactive.
        """
        t = time.time() if t is None else t
        start, end = self._current_period_bounds(t)
        return max(0.0, end - t) if start <= t < end else 0.0

    def get_schedule_info(self) -> dict[str, object]:
        """Return human-readable schedule configuration and status.

        Returns:
            A dict describing the schedule configuration and current status.
        """
        now = time.time()
        is_active = self.is_active_period(now)

        info: dict[str, object] = {
            "daily_duration_hours": self._daily_duration_hours,
            "interval_hours": self._interval_hours,
            "daily_start_time": self._daily_start_time,
            "currently_active": is_active,
        }

        if is_active:
            info["remaining_active_seconds"] = self.get_remaining_active_time(now)
        else:
            info["seconds_until_next_period"] = self.get_time_until_next_period(now)

        next_start, next_end = self.get_next_active_period(now)
        info["next_period_start"] = self._isoformat_local(next_start)
        info["next_period_end"] = self._isoformat_local(next_end)

        return info
