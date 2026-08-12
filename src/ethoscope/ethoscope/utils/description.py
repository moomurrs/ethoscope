# author: quentin
# refactor: moomurrs

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


class DescribedObject:
    """Base class for objects exposing a web-interface-friendly description.

    The :pyattr:`description` attribute is a dictionary with the following keys:

    * ``overview`` (str): A short, user-friendly description of the object.
    * ``arguments`` (list[dict[str, object]]): One entry per ``__init__``
      parameter the web interface should expose. Each entry contains:

        * ``name`` (str): The argument name as it appears in ``__init__``.
        * ``description`` (str): A user-friendly description of the argument.
        * ``type`` (str): One of ``"number"``, ``"datetime"``, ``"daterange"``,
          ``"str"``, ``"select"`` or ``"boolean"``.
        * ``min``, ``max``, ``step`` (number, optional): Only relevant when
          ``type == "number"``. Define accepted limits and the UI increment.
        * ``default``: The default value.
        * ``hidden`` (bool, optional): When ``True``, hide the object from the
          web UI.
        * ``asknode``, ``required`` (optional): Node-side metadata.
        * ``options`` (list[dict[str, str]], optional): Only relevant when
          ``type == "select"``. Each entry maps a ``value`` to a display
          ``label``.
        * ``depends_on`` (dict[str, list[str]], optional): Values of other
          arguments that must be selected for this argument to apply.

    Every argument's ``name`` must correspond to a parameter of the subclass'
    ``__init__``.

    Subclasses set the description by assigning a dictionary to the
    ``_description`` class attribute.
    """

    _description: ClassVar[Mapping[str, object] | None] = None

    @property
    def description(self) -> Mapping[str, object] | None:
        """Return the description dictionary for the web interface.

        Returns:
            The description dictionary, or ``None`` if the subclass did not
            define one.
        """
        return self._description
