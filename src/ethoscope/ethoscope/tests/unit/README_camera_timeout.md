# Camera Watchdog Tests

This document describes the unit tests for the camera initialization watchdog that prevents ethoscope devices from hanging indefinitely.

## Problem Solved

The ethoscope occasionally encountered errors during camera initialization (e.g. libcamera timeout, `picamera2` compatibility issues) that caused it to hang in `initialising` state indefinitely, requiring manual intervention.

## Solution

### Process-Level Failsafe (`tracking.py`)

- **2-minute watchdog timer** that monitors initialization status
- **Automatic process termination** if stuck in `initialising` state
- **Graceful error reporting** before termination

### Camera Acquisition (`cameras.py`)

- **30-second timeout** for first-frame acquisition (`OurPiCameraAsync` + `PiFrameGrabber`)
- **Single-attempt, fail-fast**: no retry, no legacy `picamera` fallback — Trixie + `picamera2` + Pi Camera v3 NoIR only
- **Detailed debugging logs** for troubleshooting

## Test Coverage

### TestCameraTimeoutMechanisms

- `test_timeout_handler_basic_functionality()`: Verifies timeout handler sets error state correctly
- `test_timeout_handler_no_trigger_when_not_initialising()`: Ensures timeout only triggers during initialization

Both tests use simplified mocks and have no hardware dependencies.

## Benefits

- **No Hardware Dependencies**: Tests run without requiring actual camera hardware
- **Fast Execution**: All tests complete in under 1 second
- **Maintainable**: Simple, focused tests that are easy to understand and modify

## Usage

Run the camera watchdog tests:

```bash
# Run just these tests
python -m pytest src/ethoscope/ethoscope/tests/unit/test_camera_timeout.py -v

# Run all unit tests
python -m pytest src/ethoscope/ethoscope/tests/unit/ -v
```

These tests ensure the watchdog works correctly, preventing ethoscopes from getting stuck in initialization loops.
