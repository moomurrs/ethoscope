# author: quentin
# refactor: moomurrs

from typing import override

import cv2
import numpy as np


class EthoscopeException(Exception):
    """A custom exception that can store a debugging image.

    Attributes:
        value: The exception message.
        img: An optional image snapshot attached to the exception, or None.
    """

    value: str
    img: np.ndarray | None

    def __init__(self, value: str, img: np.ndarray | None = None) -> None:
        """Create the exception, optionally attaching a copy of an image.

        Args:
            value: The exception message.
            img: An image to attach. Non-array values are ignored.
        """
        super().__init__(value)
        self.value = value
        self.img = np.copy(img) if isinstance(img, np.ndarray) else None

    @override
    def __str__(self) -> str:
        """Return the repr of the message for easy debugging."""
        return repr(self.value)


def show(im: np.ndarray, t: int = -1) -> None:
    """Display an image in a debug window and wait (debugging only).

    Args:
        im: The image to display.
        t: Wait time in milliseconds; lower than 1 waits for a key press.
    """
    cv2.imshow("debug", im)
    _ = cv2.waitKey(t)
