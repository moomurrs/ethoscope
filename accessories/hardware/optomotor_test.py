#!/usr/bin/env python3
"""Sweep-test every optomotor channel of an ethoscope.

Automates the manual interpreter check::

    from ethoscope.hardware.interfaces.optomotor import OptoMotor
    motor = OptoMotor()
    motor.activate(channel=1, duration=1000, intensity=1000)

By default the odd channels 1, 3, 5, ..., 19 are pulsed for one second
each, matching the manual test. Use ``--channel 7`` to test a single
channel (``--channel 3 7 11`` for several) or ``--all`` to sweep every
channel.

The serial port is always closed before the process exits, including on
Ctrl-C, and a short grace period is left afterwards so the USB CDC device
settles and the ethoscope tracking service can re-open the port for the
next experiment.

Usage on an ethoscope (Raspbian/Debian Trixie, ``/opt/ethoscope``
install)::

    sudo /usr/bin/python3 \
        /opt/ethoscope/accessories/hardware/optomotor_test.py

``sudo`` (or membership of the ``dialout`` group) is needed to open
``/dev/ttyACM*`` / ``/dev/ttyUSB*``. From a development checkout, run it
with the interpreter that has the ``ethoscope`` package installed.

Exit codes: 0 = all channels accepted their command, 1 = failure or no
module detected, 130 = interrupted with Ctrl-C.
"""

import argparse
import logging
import sys
import time

import serial

from ethoscope.hardware.interfaces.optomotor import OptoMotor

# Seconds given to the board to settle after the serial port is closed, so
# the tracking service can reliably re-open it.
PORT_RELEASE_GRACE_S = 2.0

# Same upper bound as documented by OptoMotor.activate().
MAX_INTENSITY = 1000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pulse each optomotor channel in turn to verify that the "
            "module is wired and communicating."
        )
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port to use (default: auto-detect).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1000,
        help="Pulse duration per channel in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        default=MAX_INTENSITY,
        help=f"PWM duty cycle 0-{MAX_INTENSITY} (default: {MAX_INTENSITY}).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.2,
        help="Extra pause after each pulse, in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "Channel(s) to test, e.g. --channel 7 or --channel 3 7 11 "
            f"(default: odd channels 1-{OptoMotor._n_channels - 1})."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test every channel (0-19), not only the odd ones.",
    )
    args = parser.parse_args(argv)

    if not 0 <= args.intensity <= MAX_INTENSITY:
        parser.error(f"--intensity must be between 0 and {MAX_INTENSITY}")
    if args.duration <= 0:
        parser.error("--duration must be a positive number of milliseconds")
    if args.gap < 0:
        parser.error("--gap cannot be negative")
    if args.all and args.channel is not None:
        parser.error("--all and --channel cannot be combined")
    if args.channel is not None:
        for channel in args.channel:
            if not 0 <= channel < OptoMotor._n_channels:
                parser.error(
                    f"--channel must be between 0 and {OptoMotor._n_channels - 1}"
                )

    return args


def _channels(test_all: bool, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        # Duplicates are preserved: --channel 7 7 fires channel 7 twice.
        return list(explicit)
    if test_all:
        return list(range(OptoMotor._n_channels))
    # Odd channels only: 1, 3, 5, ..., 19.
    return list(range(1, OptoMotor._n_channels, 2))


def _pulse(
    motor: OptoMotor,
    channel: int,
    duration: int,
    intensity: int,
) -> str | None:
    """Send one activation command.

    Return ``None`` on success, or an error description otherwise.
    """
    try:
        written = motor.activate(
            channel=channel,
            duration=duration,
            intensity=intensity,
        )
    except (serial.SerialException, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if not written:
        return "serial write returned 0"
    return None


def _close_serial(motor: OptoMotor) -> None:
    """Close the serial port and leave the board time to settle."""
    serial_obj = getattr(motor, "_serial", None)
    if serial_obj is None:
        return
    try:
        serial_obj.close()
    except (serial.SerialException, OSError) as exc:
        print(f"WARNING: serial close failed: {exc}", file=sys.stderr)
        return
    print("Serial port closed.")
    time.sleep(PORT_RELEASE_GRACE_S)


def main(argv: list[str] | None = None) -> int:
    # Surface the library's own diagnostics (port discovery, serial errors)
    # which would otherwise be dropped by an unconfigured root logger.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _parse_args(argv)
    channels = _channels(args.all, args.channel)
    cycle_s = args.duration / 1000.0 + args.gap

    exit_code = 0
    motor: OptoMotor | None = None
    try:
        print("Connecting to the optomotor module...")
        motor = OptoMotor(port=args.port)

        if getattr(motor, "_serial", None) is None:
            print(
                "ERROR: no optomotor module detected. Check that the board "
                "is plugged in and powered, then try again.",
                file=sys.stderr,
            )
            return 1

        if args.port is not None and motor._port != args.port:
            print(
                f"WARNING: requested {args.port}, but the interface is "
                f"using {motor._port}.",
                file=sys.stderr,
            )

        channel_list = ", ".join(str(channel) for channel in channels)
        pulses_s = len(channels) * cycle_s
        print(f"Connected on {motor._port}.")
        print(
            f"Channels: {channel_list} | {args.duration} ms at intensity "
            f"{args.intensity} | pulses take ~{pulses_s:.0f} s"
        )
        print()

        failures = 0
        for channel in channels:
            error = _pulse(motor, channel, args.duration, args.intensity)
            if error is None:
                print(f"ch {channel:2d} -> OK")
            else:
                failures += 1
                print(f"ch {channel:2d} -> FAILED ({error})")
            # Let the pulse finish before moving to the next channel.
            time.sleep(cycle_s)

        print()
        if failures:
            exit_code = 1
            print(
                f"FAILED: {failures} of {len(channels)} channel(s) did not "
                "respond.",
                file=sys.stderr,
            )
        else:
            print(f"ALL OK: {len(channels)} channel(s) tested.")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130
    finally:
        if motor is not None:
            _close_serial(motor)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
