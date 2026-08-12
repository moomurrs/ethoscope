# author: quentin
# refactor: moomurrs
"""Typed integer variables for tracking data points.

Each variable class subclasses :class:`int` so that values behave as plain
integers while carrying the schema and semantic metadata consumed by the
result writers (:mod:`ethoscope.io.base`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Self, override

if TYPE_CHECKING:
    from .roi import ROI


type VariableValue = int | float


class SQLDataType(StrEnum):
    """SQL column types that variable classes may declare."""

    SMALLINT = "SMALLINT"
    BOOLEAN = "BOOLEAN"
    INT = "INT"
    DOUBLE = "DOUBLE"
    VARCHAR100 = "VARCHAR(100)"


class BaseIntVariable(int):
    """Template class for defining typed integer variables.

    Subclasses must define the three following attributes:

    * ``sql_data_type``: the SQL data type used to store data points, so
      they occupy minimal space. It must be one of the
      :class:`~ethoscope.core.variables.SQLDataType` members.
    * ``header_name``: the column name of this variable in result tables;
      it must be unique.
    * ``functional_type``: a keyword defining what kind of variable this is
      (e.g. "distance", "angle" or "proba"), enabling per-functional-type
      post-processing.

    Incomplete definitions are rejected at class-definition time via
    :meth:`__init_subclass__`. Abstract template classes (including
    ``BaseIntVariable`` itself) declare ``_abstract = True``; all other
    subclasses are treated as concrete and validated automatically.
    """

    _abstract: ClassVar[bool] = True
    sql_data_type: SQLDataType = SQLDataType.SMALLINT
    header_name: str = ""
    functional_type: str = ""

    def __new__(cls, value: VariableValue) -> Self:
        """Create a new variable instance.

        Args:
            value: The numeric value held by this variable.

        Raises:
            NotImplementedError: If the class is abstract and cannot be
                instantiated.

        Returns:
            A new variable instance.
        """
        if cls._is_abstract():
            raise NotImplementedError(
                f"'{cls.__name__}' is abstract; it cannot be instantiated"
            )
        return super().__new__(cls, value)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls._is_abstract():
            return
        cls._validate_definition()

    @classmethod
    def _is_abstract(cls) -> bool:
        """Return whether the class was explicitly declared abstract.

        Only a marker declared in the class's own namespace counts, so that
        subclasses of abstract templates are treated as concrete.
        """
        return cls.__dict__.get("_abstract", False) is True

    @classmethod
    def _validate_definition(cls) -> None:
        """Validate the required class attributes for a concrete type."""
        if not cls.functional_type:
            raise NotImplementedError(
                "Variables must have a functional data type (e.g. 'distance', 'angle')"
            )
        if not cls.sql_data_type:
            raise NotImplementedError(
                "Variables must have an SQL data type such as INT"
            )
        if not cls.header_name:
            raise NotImplementedError("Variables must have a header name")


class BaseBoolVariable(BaseIntVariable):
    """Abstract type encoding boolean values as integers (0 or 1)."""

    _abstract: ClassVar[bool] = True
    functional_type: str = "bool"
    sql_data_type: SQLDataType = SQLDataType.BOOLEAN


class IsInferredVariable(BaseBoolVariable):
    """Whether a data point is inferred (from past values) or observed; 1 or 0."""

    header_name: str = "is_inferred"


class PhiVariable(BaseIntVariable):
    """The angle of a detected object, in degrees."""

    header_name: str = "phi"
    functional_type: str = "angle"


class Label(BaseIntVariable):
    """Discrete label identifying an object within a ROI."""

    header_name: str = "label"
    functional_type: str = "label"


class BaseDistanceIntVar(BaseIntVariable):
    """Abstract type encoding variables representing distances."""

    _abstract: ClassVar[bool] = True
    functional_type: str = "distance"


class mLogLik(BaseIntVariable):
    """Type representing a log likelihood.

    It should be multiplied by 1000 to be stored as an int.
    """

    header_name: str = "mlog_L_x1000"
    functional_type: str = "proba"


class XYDistance(BaseIntVariable):
    """Distance moved between two consecutive observations.

    Log10 x 1000 is used so that the floating point distance is stored as an int.
    """

    header_name: str = "xy_dist_log10x1000"
    functional_type: str = "relative_distance_1e6"


class WidthVariable(BaseDistanceIntVar):
    """The width of a detected object."""

    header_name: str = "w"


class HeightVariable(BaseDistanceIntVar):
    """The height of a detected object."""

    header_name: str = "h"


class BaseRelativeVariable(BaseDistanceIntVar):
    """Abstract type for distance variables expressed relative to an origin.

    Relative variables are converted to absolute coordinates using
    information from the ROI.
    """

    _abstract: ClassVar[bool] = True

    def to_absolute(self, roi: ROI) -> Self:
        """Convert a relative position to absolute image coordinates.

        Args:
            roi: The region of interest the variable was measured in.

        Returns:
            A new variable expressed relative to the top left of the parent image.
        """
        return self._get_absolute_value(roi)

    def _get_absolute_value(self, _roi: ROI) -> Self:
        raise NotImplementedError(
            "Relative variable must implement a `_get_absolute_value()` method"
        )


class XPosVariable(BaseRelativeVariable):
    """The X position of a detected object."""

    header_name: str = "x"

    @override
    def _get_absolute_value(self, roi: ROI) -> XPosVariable:
        ox, _ = roi.offset
        return XPosVariable(int(self) + ox)


class YPosVariable(BaseRelativeVariable):
    """The Y position of a detected object."""

    header_name: str = "y"

    @override
    def _get_absolute_value(self, roi: ROI) -> YPosVariable:
        _, oy = roi.offset
        return YPosVariable(int(self) + oy)
