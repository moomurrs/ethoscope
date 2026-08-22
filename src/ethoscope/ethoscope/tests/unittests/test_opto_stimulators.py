"""
Tests for the optogenetic LED stimulator classes and OptoMotor pulse_train routing.
"""

import unittest
from unittest.mock import patch

from ethoscope.hardware.interfaces.optomotor import OptoMotor


class MockSerial:
    """Mock serial port for OptoMotor tests."""

    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)
        return len(data)

    def close(self):
        pass


class TestOptoMotorPulseTrain(unittest.TestCase):
    """Test OptoMotor pulse_train method and send() routing."""

    def setUp(self):
        """Create an OptoMotor with mocked serial."""
        with patch.object(OptoMotor, "__init__", lambda self, *a, **kw: None):
            self.motor = OptoMotor()
            self.motor._serial = MockSerial()
            self.motor._n_channels = 20

    def test_pulse_train_sends_W_command(self):
        """Test that pulse_train sends the correct W command."""
        self.motor.pulse_train(channel=4, on_ms=100, off_ms=200, cycles=5)
        self.assertEqual(self.motor._serial.written[-1], b"W 4 100 200 5\r\n")

    def test_activate_sends_P_command(self):
        """Test that activate sends the correct P command."""
        self.motor.activate(channel=3, duration=1000, intensity=800)
        self.assertEqual(self.motor._serial.written[-1], b"P 3 1000 800\r\n")

    def test_send_routes_to_pulse_train(self):
        """Test that send() routes to pulse_train when on_ms/off_ms/cycles are given."""
        self.motor.send(channel=2, on_ms=50, off_ms=100, cycles=10)
        self.assertEqual(self.motor._serial.written[-1], b"W 2 50 100 10\r\n")

    def test_send_routes_to_activate(self):
        """Test that send() routes to activate when no pulse train params."""
        self.motor.send(channel=1, duration=500, intensity=900)
        self.assertEqual(self.motor._serial.written[-1], b"P 1 500 900\r\n")

    def test_send_default_routes_to_activate(self):
        """Test that send() with only channel routes to activate with defaults."""
        self.motor.send(channel=0)
        self.assertEqual(self.motor._serial.written[-1], b"P 0 10000 1000\r\n")

    def test_pulse_train_negative_channel_raises(self):
        """Test that pulse_train raises for negative channel."""
        with self.assertRaises(Exception):  # noqa: B017
            self.motor.pulse_train(channel=-1, on_ms=100, off_ms=100, cycles=5)

    def test_n_channels_is_20(self):
        """Test that _n_channels is 20 (not legacy 24)."""
        self.assertEqual(self.motor._n_channels, 20)


class TestOptoMotorMissingHardware(unittest.TestCase):
    """Regression tests for gilestrolab/ethoscope#216.

    Ensure OptoMotor can be instantiated on devices with no attached
    module (no serial port) without raising, and that send() calls are
    silently dropped instead of crashing.
    """

    def test_init_survives_missing_serial_port(self):
        """OptoMotor() must not raise when no serial port is available."""
        import serial as pyserial

        with patch.object(OptoMotor, "_find_port", return_value=""):
            with patch(
                "ethoscope.hardware.interfaces.optomotor.serial.Serial",
                side_effect=pyserial.SerialException(
                    "[Errno 2] could not open port ''"
                ),
            ):
                motor = OptoMotor()

        self.assertIsNone(motor._serial)

    def test_init_survives_file_not_found(self):
        """FileNotFoundError from os.open must not escape."""
        with patch.object(OptoMotor, "_find_port", return_value="/dev/ttyACM0"):
            with patch(
                "ethoscope.hardware.interfaces.optomotor.serial.Serial",
                side_effect=FileNotFoundError(2, "No such file or directory"),
            ):
                motor = OptoMotor()

        self.assertIsNone(motor._serial)

    def test_activate_is_noop_without_serial(self):
        """activate() must not raise AttributeError when _serial is None."""
        with patch.object(OptoMotor, "__init__", lambda self, *a, **kw: None):
            motor = OptoMotor()
            motor._serial = None

        # Should not raise; returns 0 to indicate no bytes written.
        self.assertEqual(motor.activate(channel=3, duration=1000, intensity=800), 0)

    def test_pulse_train_is_noop_without_serial(self):
        """pulse_train() must not raise AttributeError when _serial is None."""
        with patch.object(OptoMotor, "__init__", lambda self, *a, **kw: None):
            motor = OptoMotor()
            motor._serial = None

        self.assertEqual(
            motor.pulse_train(channel=4, on_ms=100, off_ms=200, cycles=5), 0
        )


if __name__ == "__main__":
    unittest.main()
