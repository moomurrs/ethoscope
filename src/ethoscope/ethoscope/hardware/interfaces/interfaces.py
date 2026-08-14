"""Hardware interfaces for Ethoscope devices.

This module provides the abstractions used to connect Ethoscope devices to
external hardware (Arduino modules, servo controllers, sensors) and to
discover which modules are currently attached via USB.
"""

__author__ = "quentin"

import ast
import json
import logging
import queue
import time
import urllib.error
import urllib.request
from collections.abc import KeysView
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import serial
from serial.tools import list_ports

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import usb.core  # pyright: ignore[reportMissingTypeStubs]
except ImportError:
    usb = None

_LOGGER = logging.getLogger(__name__)

_NO_USB_DEVICE = "0000:0000"
_MODULE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    SyntaxError,
    TypeError,
)
_PORT_SELECTION_LOG_THRESHOLD = 2


class WrongSerialPortError(Exception):
    """Raised when an explicitly requested serial port is not valid."""


class NoValidPortError(Exception):
    """Raised when no usable serial port can be found."""


class ScanException(Exception):
    """Base error raised when an ethoscope sensor cannot be scanned."""


class EmptySensorResponse(ScanException):
    """Raised when a sensor URL responds with an empty body."""

    def __init__(self) -> None:
        super().__init__("No message back")


class InvalidSensorJSON(ScanException):
    """Raised when a sensor response cannot be parsed as JSON."""

    def __init__(self) -> None:
        super().__init__("Could not parse Json object")


class NoSerialConnectionError(RuntimeError):
    """Raised when interrogating hardware without an open serial connection."""

    def __init__(self) -> None:
        super().__init__("No serial connection available to the hardware module")


class InvalidModuleResponse(ValueError):
    """Raised when a module response is not a dictionary of capabilities."""

    def __init__(self) -> None:
        super().__init__("Module response is not a dictionary of capabilities")


class InvalidInstructionError(TypeError):
    """Raised when an instruction is not a dictionary."""

    def __init__(self) -> None:
        super().__init__("instructions should be dictionaries")


class BatchInstructionError(TypeError):
    """Raised when a batched instruction is not a list of dictionaries."""

    def __init__(self) -> None:
        super().__init__("batch instructions should be a list of dictionaries")


class DeviceInfo(TypedDict):
    """Description of a known USB device used by Ethoscope modules."""

    name: str
    id: list[str]
    family: NotRequired[str]
    model: NotRequired[str]
    used_for: NotRequired[list[str]]
    aka: NotRequired[str]


type DeviceCatalog = dict[str, DeviceInfo]
type Instruction = dict[str, object]
type InstructionPayload = Instruction | list[Instruction]


def _validate_instruction(instruction: object) -> InstructionPayload:
    """Validate that ``instruction`` is an instruction dict or a batch of dicts.

    Args:
        instruction: A dictionary of keyword arguments matching
            ``interface_class.send()``, or a list of such dictionaries.

    Returns:
        The validated instruction payload.

    Raises:
        InvalidInstructionError: If ``instruction`` is neither a dict nor a list.
        BatchInstructionError: If a batch contains a non-dict item.
    """
    if isinstance(instruction, list):
        if not all(
            isinstance(item, dict) for item in cast("list[object]", instruction)
        ):
            raise BatchInstructionError
        return cast("InstructionPayload", instruction)
    if isinstance(instruction, dict):
        return cast("InstructionPayload", instruction)
    raise InvalidInstructionError


def _detect_usb_devices() -> list[str]:
    """Return the ``vendor:product`` IDs of the USB devices attached to this machine.

    Returns the no-device sentinel when pyusb is unavailable or no USB device
    can be enumerated.
    """
    if usb is None:
        return [_NO_USB_DEVICE]
    try:
        devices = cast("Iterator[Any]", usb.core.find(find_all=True))
        return [f"{dev.idVendor:04x}:{dev.idProduct:04x}" for dev in (devices or [])]
    except (usb.core.NoBackendError, usb.core.USBError):
        return [_NO_USB_DEVICE]


def connectedUSB(
    optional_file: str = "/etc/ethoscope/modules.json",
) -> tuple[DeviceCatalog, DeviceCatalog]:
    """Return a dictionary of connected USB devices from a known selection.

    Known devices:
    #Arduino Micro
    Bus 001 Device 005: ID 2341:8037 Arduino SA Arduino Micro

    #Arduino Nano Every
    Bus 001 Device 006: ID 2341:0058 Arduino SA Arduino Nano Every

    #Lynxmotion SSC-32U
    Bus 001 Device 008: ID 0403:6001 Future Technology Devices International, Ltd
    FT232 Serial (UART) IC

    Args:
        optional_file: JSON file with user-defined interactors to merge into the
            known-device catalog.

    Returns:
        A tuple ``(known, found)`` with the full device catalog and the subset
        of currently attached devices.
    """
    known: DeviceCatalog = {
        "arduino_nano": {
            "name": "Arduino Nano",
            "family": "arduino",
            "model": "nano",
            "used_for": ["optomotor", "mAGO", "opto_LED"],
            "id": ["2341:0058"],
        },
        "arduino_micro": {
            "name": "Arduino Micro",
            "family": "arduino",
            "model": "micro",
            "used_for": ["optomotor", "mAGO", "opto_LED"],
            "id": ["2341:8037"],
        },
        "arduino_uno": {
            "name": "Arduino UNO",
            "family": "arduino",
            "model": "uno",
            "used_for": [],
            "id": ["2341:0043"],
        },
        "arduino_leonardo": {
            "name": "Arduino Leonardo",
            "family": "arduino",
            "model": "leonardo",
            "used_for": [],
            "id": ["2341:8036"],
        },
        "wemos_D1": {
            "name": "Wemos D1",
            "family": "ESP8266",
            "model": "D1",
            "used_for": [],
            "id": ["1a86:7523"],
            "aka": "CH340",
        },
        "lynxmotion_ssc32u": {
            "name": "LynxMotion SSC-32U",
            "family": "LynxMotion",
            "model": "SSC-32U",
            "used_for": ["servo", "AGO"],
            "id": ["0403:6001"],
            "aka": "FT232",
        },
        "noUSB": {
            "name": "python-usb not loaded",
            "id": ["0000:0000"],
        },
    }

    optional_path = Path(optional_file)
    if optional_path.exists():
        with optional_path.open() as modules_file:
            user_modules = json.load(modules_file)
            if isinstance(user_modules, dict):
                known.update(cast("DeviceCatalog", user_modules))

    devices = _detect_usb_devices()
    detected = set(devices)
    found = {
        name: info
        for name, info in known.items()
        if detected.intersection(info.get("id", []))
    }
    return known, found


class HardwareInterface(Protocol):
    """Protocol for hardware interfaces driven by :class:`HardwareConnection`."""

    def send(self, **kwargs: object) -> None:
        """Send an instruction to the hardware."""
        ...

    def interrogate(self, test: bool = False, command: str = "") -> dict[str, Any]:
        """Query the hardware for capability information."""
        ...

    def close(self) -> None:
        """Release the hardware resources held by this interface."""
        ...


class HardwareConnection(Thread):
    """Asynchronous, thread-safe connection to arbitrary hardware.

    Owns an instance of a :class:`HardwareInterface` subclass and delivers
    instructions to it on a dedicated worker thread, using a thread-safe queue
    for zero CPU usage when idle and near-zero latency when active.
    """

    def __init__(
        self,
        interface_class: type[HardwareInterface],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create a hardware connection and start its worker thread.

        Args:
            interface_class: The hardware interface class to instantiate (a
                subclass of :class:`SimpleSerialInterface` or compatible).
            args: Positional arguments forwarded to ``interface_class``.
            kwargs: Keyword arguments forwarded to ``interface_class``.
        """
        self._interface_class: type[HardwareInterface] = interface_class
        self._interface_args: tuple[Any, ...] = args
        self._interface_kwargs: dict[str, Any] = kwargs
        self._interface: HardwareInterface = interface_class(*args, **kwargs)
        self._instructions: queue.Queue[InstructionPayload | None] = queue.Queue()
        self._connection_open: bool = True
        super().__init__()
        self.start()

    @override
    def run(self) -> None:
        """Blocking loop that sends instructions to the hardware interface.

        Items may be single instruction dicts or batches (lists) of
        instructions sent atomically.
        """
        while self._connection_open:
            instruction = self._instructions.get()
            if instruction is None:
                self._instructions.task_done()
                _LOGGER.error("Hardware thread shut down!")
                break
            try:
                if isinstance(instruction, list):
                    for item in instruction:
                        self._interface.send(**item)
                else:
                    self._interface.send(**instruction)
            except Exception:
                _LOGGER.exception("Could not send instruction to hardware module")
            finally:
                self._instructions.task_done()
        self._interface.close()

    def send_instruction(self, instruction: object | None = None) -> None:
        """Stage an instruction to be sent to the hardware interface.

        Instructions are parsed sequentially, but asynchronously from the
        calling thread. Accepts a single instruction dict or a list of
        instruction dicts for atomic batch delivery (e.g. yoked stimuli).

        Args:
            instruction: A dictionary of keyword arguments matching
                ``interface_class.send()``, or a list of such dictionaries.
        """
        if instruction is None:
            instruction = {}
        self._instructions.put(_validate_instruction(instruction))

    def stop(self) -> None:
        """Shut the connection down and release the hardware interface.

        Idempotent: safe to call multiple times.
        """
        if not self._connection_open:
            return
        self._connection_open = False
        self._instructions.put(None)
        _LOGGER.error("Hardware thread shutting down...")

    def interrogate(self, test: bool = False, command: str = "") -> dict[str, Any]:
        """Interrogate the underlying hardware interface for capability information.

        Args:
            test: Whether to run diagnostic tests on the module.
            command: Optional custom AT command to run post-interrogation.

        Returns:
            The capability information reported by the module.
        """
        return self._interface.interrogate(test=test, command=command)

    def __enter__(self) -> Self:
        """Enter the runtime context; the connection is already running."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Stop the connection and wait for the worker thread to finish."""
        self.stop()
        self.join(timeout=5.0)
        if self.is_alive():
            _LOGGER.warning(
                "Hardware thread did not terminate within the join timeout"
            )

    @override
    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state of this connection."""
        return {
            "interface_class": self._interface_class,
            "interface_args": self._interface_args,
            "interface_kwargs": self._interface_kwargs,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the connection state, rebuilding the interface with no warm-up."""
        kwargs = dict(state["interface_kwargs"])
        kwargs["warmup"] = False
        interface_class = cast("type[HardwareInterface]", state["interface_class"])
        self.__init__(interface_class, *state["interface_args"], **kwargs)


class SimpleSerialInterface:
    """Abstract representation of a Serial hardware interface.

    Subclasses must define how the interface is connected and how information
    is communicated to the hardware by implementing :meth:`send` and
    :meth:`_warm_up`.
    """

    def __init__(
        self, port: str | None = None, baud: int = 115200, warmup: bool = False
    ) -> None:
        """Connect to a serial port, optionally running a warm-up sequence.

        Args:
            port: Serial port to use; auto-detected when ``None``.
            baud: Baud rate of the serial connection.
            warmup: Whether to run :meth:`_warm_up` upon connection.
        """
        _LOGGER.info("Connecting to Serial port...")
        self._serial: serial.Serial | None = None
        self._port: str = port if port is not None else self._find_port()
        try:
            self._serial = serial.Serial(self._port, baud, timeout=2)
            time.sleep(2)
            if warmup:
                self._warm_up()
        except (serial.SerialException, OSError, ValueError) as e:
            _LOGGER.warning("Could not connect to serial port %r: %s", self._port, e)
            if self._serial is not None:
                self._serial.close()
            self._serial = None

    def _find_port(self) -> str:
        """Detect and return the serial port used by the attached module.

        Returns:
            The path of the first detected ``ttyUSB``/``ttyACM`` port, or an
            empty string if no valid port is detected.
        """
        _LOGGER.info("listing serial ports")
        ports: set[str] = set()
        for port in list_ports.comports():
            if Path(port.device).name.startswith(("ttyUSB", "ttyACM")):
                ports.add(port.device)
                _LOGGER.info("\t%s", port.device)
        if not ports:
            _LOGGER.error(
                "No valid port detected!. Possibly, device not plugged/detected."
            )
            return ""
        ordered = sorted(ports)
        if len(ordered) > _PORT_SELECTION_LOG_THRESHOLD:
            _LOGGER.info("Several port detected, using first one: %s", ordered[0])
        return ordered[0]

    def close(self) -> None:
        """Close the underlying serial connection, if open."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> Self:
        """Enter the runtime context; the connection is established at construction."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the serial connection when leaving the context."""
        self.close()

    def _test_serial_connection(self) -> None:
        """Test the serial connection (legacy no-op hook)."""
        return

    def interrogate(self, test: bool = False, command: str = "") -> dict[str, Any]:
        """Interrogate the module for capability information, optionally running tests.

        Compatible with all new PCB firmware versions and firmware releases
        post-September 2020. Supports both capability discovery and functional
        validation through test execution.

        Args:
            test: Enable diagnostic test execution using device-reported
                parameters.
            command: Custom AT command to execute immediately after
                interrogation; an empty string disables it.

        Returns:
            Structured device information with device capabilities plus
            ``test`` and ``command`` execution status
            ('Success'/'Failed'/'Not attempted').

        Note:
            Automatically adapts to firmware version differences in response
            structures, maintaining backward compatibility with legacy firmware
            (v1.0) while supporting modern implementations (v>1.0).
        """
        connection = self._require_serial()
        _ = connection.write(b"T\r\n")
        time.sleep(0.1)
        response = connection.read_all()
        parsed = cast("object", ast.literal_eval(response.decode()))
        if not isinstance(parsed, dict):
            raise InvalidModuleResponse
        info = cast("dict[str, Any]", parsed)
        info.update({"test": "Not attempted", "command": "Not attempted"})

        if command:
            try:
                _LOGGER.info("Initiating custom command execution")
                _ = connection.write(f"{command}\r\n".encode())
                info["command"] = "Success"
            except (OSError, TypeError):
                _LOGGER.exception("Custom command transmission failed")
                info["command"] = "Failed"

        if test:
            try:
                _LOGGER.info("Executing diagnostic test sequence")
                cmd = self._get_test_button_command(info)
                _ = connection.write(f"{cmd}\r\n".encode())
                info["test"] = "Success"
            except (KeyError, TypeError, OSError):
                _LOGGER.exception("Diagnostic test execution failed")
                info["test"] = "Failed"

        return info

    def _get_test_button_command(self, info: dict[str, Any]) -> object:
        """Extract the test button command from a capability response.

        Tries the modern firmware layout first (``interface.test_button``),
        falling back to the legacy v1.0 layout.
        """
        try:
            return cast("object", info["interface"]["test_button"]["command"])
        except (KeyError, TypeError):
            return cast("object", info["test_button"]["command"])

    def _require_serial(self) -> serial.Serial:
        """Return the open serial connection, or raise if none is available."""
        if self._serial is None:
            raise NoSerialConnectionError
        return self._serial

    def _warm_up(self) -> None:
        """Run optional start-up instructions on the hardware."""
        raise NotImplementedError

    def send(self, **_kwargs: object) -> None:
        """Request the hardware interface to interact with the physical world.

        Args:
            _kwargs: Keyword arguments understood by the hardware interface.
        """
        raise NotImplementedError


class DefaultInterface(SimpleSerialInterface):
    """Dummy interface that does nothing, for when no hardware is to be used."""

    @override
    def _warm_up(self) -> None:
        """No-op warm-up for the dummy interface."""

    @override
    def send(self, **_kwargs: object) -> None:
        """No-op send for the dummy interface."""


class EthoscopeSensor:
    """Provide access to an ESP32 based WIFI ethoscope sensor."""

    _sensor_values: ClassVar[dict[str, str]] = {
        "temperature": "FLOAT",
        "humidity": "FLOAT",
        "light": "INT",
        "pressure": "FLOAT",
    }

    def __init__(self, sensor_url: str) -> None:
        """Create a sensor connection and read the initial values.

        Args:
            sensor_url: URL of the sensor's HTTP endpoint.
        """
        self._sensor_url: str = sensor_url
        self._last_read: float = 0
        self._sensor_data: dict[str, Any] = {}
        self._update(True)

    def _get_json_from_url(
        self, url: str, timeout: int = 5, post_data: bytes | None = None
    ) -> dict[str, Any]:
        """Fetch and parse the JSON payload published by the sensor URL.

        Args:
            url: Sensor URL; ``http://`` is prepended when missing.
            timeout: Request timeout in seconds.
            post_data: Optional body sent with the request.

        Returns:
            The parsed JSON payload.

        Raises:
            ScanException: When the request fails or the payload is invalid.
        """
        if not url.startswith("http://"):
            url = "http://" + url
        request = urllib.request.Request(
            url, data=post_data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                message = cast("bytes", response.read())
        except urllib.error.HTTPError as e:
            raise ScanException("Error" + str(e.code)) from e
        except urllib.error.URLError as e:
            raise ScanException("Error" + str(e.reason)) from e
        except (OSError, ValueError) as e:
            raise ScanException("Unexpected error" + str(e)) from e
        if not message:
            raise EmptySensorResponse
        try:
            return cast("dict[str, Any]", json.loads(message))
        except (ValueError, TypeError) as e:
            raise InvalidSensorJSON from e

    def _update(self, force: bool = False, freq: float = 5) -> None:
        """Refresh sensor values if the previous read is older than ``freq`` seconds.

        It is usually a good idea not to interrogate the sensors too often to
        avoid overheating.

        Args:
            force: Force a refresh regardless of age.
            freq: Maximum refresh interval in seconds.
        """
        if (time.time() - self._last_read) > freq or force:
            self._sensor_data = self._get_json_from_url(self._sensor_url)
            self._last_read = time.time()

    def read_all(self) -> tuple[Any, ...]:
        """Return the current value of every sensor property, in order."""
        self._update()
        return tuple(self._sensor_data[name] for name in self.sensor_properties)

    @property
    def sensor_properties(self) -> KeysView[str]:
        """The names of the sensor properties exposed by this sensor."""
        self._update()
        return self._sensor_values.keys()

    @property
    def sensor_types(self) -> dict[str, str]:
        """Mapping of sensor property names to their database types."""
        self._update()
        return self._sensor_values

    @property
    def temperature(self) -> Any:
        """The most recent temperature reading."""
        self._update()
        return self._sensor_data["temperature"]

    @property
    def humidity(self) -> Any:
        """The most recent humidity reading."""
        self._update()
        return self._sensor_data["humidity"]

    @property
    def light(self) -> Any:
        """The most recent light reading."""
        self._update()
        return self._sensor_data["light"]

    @property
    def pressure(self) -> Any:
        """The most recent pressure reading."""
        self._update()
        return self._sensor_data["pressure"]


def getModuleCapabilities(
    test: bool = False, shallow: bool = False, command: str = ""
) -> dict[str, Any]:
    """Try to get information regarding a possible attached module.

    Args:
        test: Whether to run diagnostic tests on the module.
        shallow: Skip interrogation when a device is detected.
        command: Optional custom AT command sent after interrogation.

    Returns:
        A dictionary describing the detected module and its capabilities.
    """
    found = connectedUSB()[1]

    if shallow and found:
        return {**found, "Smart": False, "Connected": True}

    if not found and "noUSB" not in found:
        return {
            "Error": "No known device is connected.",
            "Smart": False,
            "Connected": False,
        }

    try:
        module = SimpleSerialInterface()
        module_info = module.interrogate(test=test, command=command)
    except _MODULE_ERRORS:
        return {
            "Error": (
                "A known device is connected but could not open a connection with it."
            ),
            "found": False if "noUSB" in found else found,
            "Smart": False,
            "Connected": True,
        }
    else:
        module_info.update({"Smart": True, "Connected": True})
        return module_info
