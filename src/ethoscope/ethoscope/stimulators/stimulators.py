# author: quentin

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar, Final, override

if TYPE_CHECKING:
    from collections.abc import Mapping

from ethoscope.core.variables import BaseIntVariable, SQLDataType
from ethoscope.hardware.interfaces.interfaces import DefaultInterface
from ethoscope.utils.description import DescribedObject
from ethoscope.utils.scheduler import Scheduler

_LOGGER: Final = logging.getLogger(__name__)

STIMULATING_WINDOW_MS: Final[int] = 2000

type ChannelMap = dict[int, int]
type RoiTemplateConfig = dict[str, Any]
type Instruction = dict[str, Any]


class NoTrackerBoundError(ValueError):
    """Raised when a stimulator is used before a tracker is bound."""

    def __init__(self) -> None:
        super().__init__(
            "No tracker bound to this stimulator. Use `bind_tracker()` methods"
        )


class InvalidRoiMappingError(ValueError):
    """Raised when an ROI template mapping is malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class HasInteractedVariable(BaseIntVariable):
    """Whether the stimulator sent an instruction to its hardware interface.

    ``0`` means no interaction. Any positive integer describes a different
    interaction.
    """

    _abstract: ClassVar[bool] = False
    functional_type: str = "interaction"
    header_name: str = "has_interacted"
    sql_data_type: SQLDataType = SQLDataType.SMALLINT


class BaseStimulator(DescribedObject):
    """Template for real-time feedback between tracking and hardware.

    Derived classes must define ``_HardwareInterfaceClass`` and implement
    ``_decide``. Instances share a ``hardware_connection`` that is borrowed
    (not owned) — the caller is responsible for its lifecycle. The class is
    deliberately thin: scheduling, interaction tracking, and ROI-mapping are
    split into focused private helpers.
    """

    _tracker: Any | None = None
    _HardwareInterfaceClass: ClassVar[type | None] = None

    def __init__(
        self,
        hardware_connection: Any | None,
        date_range: str = "",
        roi_template_config: RoiTemplateConfig | None = None,
    ) -> None:
        """Create a stimulator.

        Args:
            hardware_connection: Borrowed hardware interface (e.g.
                :class:`ethoscope.hardware.interfaces.interfaces.HardwareConnection`).
                ``None`` is allowed for tracking-only subclasses.
            date_range: Scheduler range string as documented in
                ``user_manual/schedulers.md``. Empty string means always active.
            roi_template_config: ROI template containing
                ``stimulator_compatibility.roi_mappings`` overrides.
        """
        self._scheduler: Scheduler = Scheduler(date_range)
        self._hardware_connection: Any | None = hardware_connection
        self._roi_template_config: RoiTemplateConfig | None = roi_template_config

        # Interaction state — initialized here so ``hasattr`` is unnecessary.
        self._last_interaction_time: float | None = None
        self._last_interaction_value: int = 0

        # Instance-level channel maps may shadow class-level defaults.
        # Preserve ``hasattr`` semantics: only create an instance attribute
        # if the mapping already exists on the class or instance (e.g.,
        # subclass defaults or ComposedStimulator pre-sets it before super()).
        if hasattr(self, "_roi_to_channel"):
            self._roi_to_channel: ChannelMap | None = self._roi_to_channel
        if hasattr(self, "_roi_to_channel_motor"):
            self._roi_to_channel_motor: ChannelMap | None = self._roi_to_channel_motor
        if hasattr(self, "_roi_to_channel_valves"):
            self._roi_to_channel_valves: ChannelMap | None = self._roi_to_channel_valves

        self._apply_template_overrides()

    def apply(self) -> tuple[HasInteractedVariable, Instruction]:
        """Run one feedback cycle.

        1. Ensure a tracker is bound.
        2. Check the date-range schedule.
        3. Delegate to ``_decide`` and forward the result to the hardware.

        Returns:
            Tuple of interaction flag and hardware instruction dict.
        """
        if self._tracker is None:
            raise NoTrackerBoundError

        current_time = time.time() * 1000

        if not self._scheduler.check_time_range():
            self._track_interaction_state(False, current_time)
            return HasInteractedVariable(False), {}

        interact, result = self._decide()

        self._track_interaction_state(interact, current_time)

        if interact > 0:
            self._deliver(**result)

        return interact, result

    def bind_tracker(self, tracker: Any) -> None:
        """Link a tracker to this stimulator.

        Args:
            tracker: Tracker providing ``positions``, ``times`` and ``_roi``.
        """
        self._tracker = tracker

    def _decide(self) -> tuple[HasInteractedVariable, Instruction]:
        """Decide whether to interact.

        Returns:
            Interaction flag and instruction kwargs for ``_deliver``.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    def _deliver(self, **kwargs: Any) -> None:
        """Forward an instruction to the hardware interface.

        Args:
            **kwargs: Keyword arguments understood by the hardware interface.
        """
        if self._hardware_connection is None:
            return
        self._hardware_connection.send_instruction(kwargs)

    def get_stimulator_state(self, t: float | None = None) -> str:
        """Return the composite schedule + interaction state.

        Args:
            t: Timestamp to check. ``None`` uses the current time.

        Returns:
            ``"inactive"`` if outside the schedule, ``"stimulating"`` if an
            interaction occurred within the last ``STIMULATING_WINDOW_MS``,
            otherwise ``"scheduled"``.
        """
        schedule_state = self._scheduler.get_schedule_state(t)
        if schedule_state == "inactive":
            return "inactive"

        if self._last_interaction_time is None:
            return "scheduled"

        current_time = t if t is not None else time.time() * 1000
        time_since_interaction = current_time - self._last_interaction_time

        if (
            time_since_interaction < STIMULATING_WINDOW_MS
            and self._last_interaction_value > 0
        ):
            return "stimulating"

        return "scheduled"

    def _track_interaction_state(
        self,
        interact: HasInteractedVariable | int | bool,
        current_time: float,
    ) -> None:
        """Record interaction timing for :meth:`get_stimulator_state`.

        Args:
            interact: Interaction flag from ``_decide``.
            current_time: Current timestamp in milliseconds.
        """
        self._last_interaction_time = current_time
        self._last_interaction_value = int(interact) if interact else 0

    def _apply_template_overrides(self) -> None:
        """Apply ROI template overrides for channel mappings if available.

        Raises:
            InvalidRoiMappingError: If the template structure is malformed.
        """
        if not self._roi_template_config:
            return

        stimulator_compatibility = self._roi_template_config.get(
            "stimulator_compatibility", {}
        )
        if not isinstance(stimulator_compatibility, dict):
            raise InvalidRoiMappingError(  # noqa: TRY003
                "Invalid stimulator_compatibility: expected dict"
            )

        roi_mappings = stimulator_compatibility.get("roi_mappings", {})
        if not isinstance(roi_mappings, dict):
            raise InvalidRoiMappingError(  # noqa: TRY003
                "Invalid roi_mappings: expected dict"
            )

        stimulator_class = self.__class__.__name__

        if stimulator_class in roi_mappings:
            self._apply_stimulator_mapping(roi_mappings[stimulator_class])
            return

        if "default" in roi_mappings:
            self._apply_default_mapping(roi_mappings["default"])

    def _apply_stimulator_mapping(self, mapping_config: Any) -> None:
        """Apply a per-stimulator mapping.

        Args:
            mapping_config: Raw mapping value from the template.

        Raises:
            InvalidRoiMappingError: If the mapping is not a dict or keys
                cannot be coerced to integers.
        """
        if not isinstance(mapping_config, dict):
            raise InvalidRoiMappingError(  # noqa: TRY003
                f"Invalid mapping for {self.__class__.__name__}: "
                f"expected dict, got {type(mapping_config).__name__}"
            )

        # Use structural pattern matching for the three layout variants.
        match mapping_config:
            case {"motor_channels": _, "valve_channels": _}:
                self._roi_to_channel_motor = self._coerce_int_keys(
                    mapping_config["motor_channels"]
                )
                self._roi_to_channel_valves = self._coerce_int_keys(
                    mapping_config["valve_channels"]
                )
            case {"motor_channels": _}:
                self._roi_to_channel_motor = self._coerce_int_keys(
                    mapping_config["motor_channels"]
                )
            case {"valve_channels": _}:
                self._roi_to_channel_valves = self._coerce_int_keys(
                    mapping_config["valve_channels"]
                )
            case _:
                self._roi_to_channel = self._coerce_int_keys(mapping_config)

    def _apply_default_mapping(self, default_mapping: Any) -> None:
        """Apply the fallback ``default`` mapping.

        Args:
            default_mapping: Raw default mapping value from the template.

        Raises:
            InvalidRoiMappingError: If the mapping is not a dict or keys
                cannot be coerced to integers.
        """
        if not isinstance(default_mapping, dict):
            raise InvalidRoiMappingError(  # noqa: TRY003
                "Invalid default mapping: expected dict"
            )
        self._roi_to_channel = self._coerce_int_keys(default_mapping)

    def _coerce_int_keys(self, mapping: Any) -> ChannelMap:
        """Convert string keys to integers for a channel map.

        Args:
            mapping: Mapping with string or integer keys.

        Returns:
            Mapping with integer keys.

        Raises:
            InvalidRoiMappingError: If the mapping is not a dict or keys
                cannot be coerced to integers.
        """
        if not isinstance(mapping, dict):
            raise InvalidRoiMappingError(  # noqa: TRY003
                f"Invalid channel map: expected dict, got {mapping!r}"
            )
        try:
            return {int(k): int(v) for k, v in mapping.items()}
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidRoiMappingError(  # noqa: TRY003
                f"Invalid ROI mapping keys {mapping!r}: {exc}"
            ) from exc


class DefaultStimulator(BaseStimulator):
    """Default interactor that never interacts."""

    _description: ClassVar[Mapping[str, object] | None] = {
        "overview": (
            "The default 'interactor'. To use when no hardware interface is to be used."
        ),
        "arguments": [],
        "hidden": True,
    }
    _HardwareInterfaceClass: ClassVar[type | None] = DefaultInterface

    @override
    def apply(self) -> tuple[HasInteractedVariable, Instruction]:
        """Return no interaction without consulting the schedule.

        Returns:
            Always ``(HasInteractedVariable(False), {})``.

        Raises:
            NoTrackerBoundError: If no tracker is bound.
        """
        if self._tracker is None:
            raise NoTrackerBoundError
        return HasInteractedVariable(False), {}

    @override
    def _decide(self) -> tuple[HasInteractedVariable, Instruction]:
        """Never decide to interact.

        Returns:
            Always ``(HasInteractedVariable(False), {})``.
        """
        return HasInteractedVariable(False), {}

    @override
    def get_stimulator_state(self, t: float | None = None) -> str:
        """Return the stimulator state.

        Args:
            t: Unused timestamp for API compatibility.

        Returns:
            Always ``"inactive"``.
        """
        return "inactive"
