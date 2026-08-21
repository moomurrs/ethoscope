"""Container for tracking variables keyed by header name."""

# author: quentin
# refactor: moomurrs

from collections.abc import Iterable
from typing import Self

from .variables import BaseIntVariable


class DataPoint(dict[str, BaseIntVariable]):
    """An insertion-ordered container of variables keyed by header name.

    Variables are accessible by header name, which is an individual identifier
    of a variable type (see :class:`~ethoscope.core.variables.BaseIntVariable`):

    >>> from ethoscope.core.variables import XPosVariable, YPosVariable, HeightVariable
    >>> point = DataPoint([XPosVariable(32), YPosVariable(18)])
    >>> point["x"]
    32
    >>> point.append(HeightVariable(3))
    >>> point["h"]
    3

    Insertion order is preserved and is relied upon downstream (e.g. the result
    writers derive SQL column order from it). Appending a variable whose header
    name already exists replaces it in place.

    """

    __slots__ = ()

    def __init__(self, data: Iterable[BaseIntVariable]) -> None:
        """Initialize the data point from an iterable of variables.

        Args:
            data: The variables to store; each is keyed by its ``header_name``.
                If several variables share a header name, the last one wins.
        """
        super().__init__((variable.header_name, variable) for variable in data)

    def copy(self) -> Self:
        """Return a new data point holding the same variables, in the same order.

        Copying with the ``=`` operator merely creates an alias to this
        ``DataPoint`` object. In contrast, this method returns an independent
        container. Because variables are immutable, their values are shared
        between the original and the copy.

        Returns:
            A copy of this object.
        """
        return type(self)(self.values())

    def append(self, item: BaseIntVariable) -> None:
        """Add a new variable to the data point; insertion order is preserved.

        If a variable with the same header name already exists, it is replaced.

        Args:
            item: The variable to be added.
        """
        self[item.header_name] = item
