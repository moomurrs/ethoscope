"""Shared type definitions for the I/O subsystem."""

from __future__ import annotations

from typing import Any, Protocol

# Public aliases - 3.12 `type` syntax
type DbCredentials = dict[str, str]
type SqlArgs = tuple[Any, ...] | None
type MetadataDict = dict[str, Any]
type ExperimentInfo = dict[str, Any]


class ROIProtocol(Protocol):
    """Minimal surface required by writers and helpers."""

    @property
    def idx(self) -> int: ...

    @property
    def longest_axis(self) -> float: ...

    def get_feature_dict(self) -> dict[str, int]: ...


class SensorProtocol(Protocol):
    """Environmental sensor contract."""

    sensor_types: dict[str, str]

    def read_all(self) -> tuple[float | int, ...]: ...


class DataPointProtocol(Protocol):
    """One tracking variable as stored in a data-row mapping.

    Used as the value type of ``Mapping[str, DataPointProtocol]`` so writers
    can read schema metadata off each column value.
    """

    header_name: str
    sql_data_type: str
    functional_type: str
