"""
Root test configuration for the ethoscope device package.

The ethoscope package imports picamera2 at module level in
``ethoscope.hardware.input.cameras``. picamera2 depends on the Raspberry Pi
camera stack (libcamera/pykms), which is not available on development and CI
machines. This conftest installs a minimal stub of picamera2 into
``sys.modules`` before the package is imported so tests can run without
camera hardware. When a real picamera2 is importable (e.g. on an actual
device), it is left untouched.
"""

import sys
import types


class _MappedArray:
    """Minimal stand-in for ``picamera2.MappedArray``."""

    def __init__(self, request, stream):
        self.request = request
        self.stream = stream
        self.array = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _Picamera2:
    """Minimal stand-in for ``picamera2.Picamera2``.

    Instantiating it raises so tests that require camera hardware fail
    loudly instead of silently succeeding against a fake.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "picamera2 is not available in this environment; camera hardware "
            "tests are disabled"
        )

    @staticmethod
    def set_logging(level):
        pass

    @staticmethod
    def load_tuning_file(path):
        raise RuntimeError("picamera2 is not available in this environment")


def _install_picamera2_stub():
    if "picamera2" in sys.modules:
        return
    try:
        __import__("picamera2")
        return
    except ImportError:
        pass

    module = types.ModuleType("picamera2")
    module.MappedArray = _MappedArray
    module.Picamera2 = _Picamera2
    sys.modules["picamera2"] = module


_install_picamera2_stub()
