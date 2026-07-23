"""
Unit tests for stimulators/composed_stimulator.py.

Tests ComposedStimulator initialization, trigger/action wiring,
channel mapping, and _decide logic.
"""

import unittest
from unittest.mock import Mock, patch

from ethoscope.stimulators.composed_stimulator import ComposedStimulator
from ethoscope.stimulators.stimulators import HasInteractedVariable


def _make_mock_tracker(roi_id=1, last_time_point=200000, positions=None, times=None):
    """Create a mock tracker."""
    tracker = Mock()
    tracker._roi = Mock()
    tracker._roi.idx = roi_id
    tracker._roi.longest_axis = 100.0
    tracker.last_time_point = last_time_point
    tracker.positions = positions or [
        [{"xy_dist_log10x1000": -3000, "x": 50}],
        [{"xy_dist_log10x1000": -3000, "x": 50}],
    ]
    tracker.times = times or [last_time_point - 1000, last_time_point]
    return tracker


class TestComposedStimulatorInit(unittest.TestCase):
    """Test ComposedStimulator initialization."""

    def _create_stimulator(self, **kwargs):
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        defaults = {
            "hardware_connection": mock_hw,
            "trigger_type": "inactivity",
            "action_type": "motor_pulse",
        }
        defaults.update(kwargs)
        return ComposedStimulator(**defaults)

    def test_init_inactivity_motor(self):
        """Test init with inactivity trigger and motor action."""
        stim = self._create_stimulator(
            trigger_type="inactivity", action_type="motor_pulse"
        )
        self.assertIsNotNone(stim._trigger)
        self.assertIsNotNone(stim._action)
        self.assertIsNotNone(stim._roi_to_channel)

    def test_init_midline_crossing_led(self):
        """Test init with midline crossing trigger and LED action."""
        stim = self._create_stimulator(
            trigger_type="midline_crossing", action_type="led_pulse"
        )
        self.assertIsNotNone(stim._trigger)
        self.assertIsNotNone(stim._action)

    def test_init_periodic_led_pulse_train(self):
        """Test init with periodic trigger and LED pulse train."""
        stim = self._create_stimulator(
            trigger_type="periodic", action_type="led_pulse_train"
        )
        self.assertIsNotNone(stim._trigger)

    def test_init_time_restricted(self):
        """Test init with time-restricted trigger."""
        stim = self._create_stimulator(
            trigger_type="time_restricted", action_type="valve_pulse"
        )
        self.assertIsNotNone(stim._trigger)

    def test_init_invalid_trigger_raises(self):
        """Test ValueError for unknown trigger type."""
        with self.assertRaises(ValueError):
            self._create_stimulator(trigger_type="nonexistent")

    def test_init_invalid_action_raises(self):
        """Test ValueError for unknown action type."""
        with self.assertRaises(ValueError):
            self._create_stimulator(action_type="nonexistent")

    def test_init_with_module_interrogation(self):
        """Test init with successful module interrogation."""
        mock_hw = Mock()
        mock_hw.interrogate.return_value = {"capabilities": {"leds": 20, "motors": 10}}
        stim = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="led_pulse",
        )
        self.assertIsNotNone(stim._roi_to_channel)


class TestComposedStimulatorBindTracker(unittest.TestCase):
    """Test bind_tracker propagation."""

    def test_bind_tracker_propagates_to_trigger(self):
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        stim = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="motor_pulse",
        )
        tracker = _make_mock_tracker()
        stim.bind_tracker(tracker)
        self.assertIs(stim._trigger._tracker, tracker)


class TestComposedStimulatorDecide(unittest.TestCase):
    """Test ComposedStimulator._decide()."""

    def _create_bound_stimulator(self, roi_id=1, **kwargs):
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        defaults = {
            "hardware_connection": mock_hw,
            "trigger_type": "inactivity",
            "action_type": "motor_pulse",
            "min_inactive_time": 0,
            "stimulus_probability": 1.0,
        }
        defaults.update(kwargs)
        stim = ComposedStimulator(**defaults)
        tracker = _make_mock_tracker(
            roi_id=roi_id,
            positions=[
                [{"xy_dist_log10x1000": -3000}],
                [{"xy_dist_log10x1000": -3000}],
            ],
            times=[199000, 200000],
        )
        stim.bind_tracker(tracker)
        return stim

    def test_decide_unmapped_roi(self):
        """Test unmapped ROI returns no interaction."""
        stim = self._create_bound_stimulator(roi_id=99)
        out, dic = stim._decide()
        self.assertEqual(bool(out), False)

    def test_decide_real_stimulus(self):
        """Test real stimulus when trigger fires."""
        stim = self._create_bound_stimulator(roi_id=1)
        stim._trigger._t0 = 0
        out, dic = stim._decide()
        self.assertEqual(int(out), 1)
        self.assertIn("channel", dic)

    def test_decide_ghost_stimulus(self):
        """Test ghost stimulus (code 2)."""
        stim = self._create_bound_stimulator(roi_id=1, stimulus_probability=0.0)
        stim._trigger._t0 = 0
        out, dic = stim._decide()
        self.assertEqual(int(out), 2)
        self.assertEqual(dic, {})

    def test_decide_no_stimulus(self):
        """Test no stimulus when trigger doesn't fire."""
        stim = self._create_bound_stimulator(roi_id=1, min_inactive_time=9999)
        out, dic = stim._decide()
        self.assertEqual(int(out), 0)

    def test_decide_motor_instruction(self):
        """Test motor pulse instruction contains duration."""
        stim = self._create_bound_stimulator(
            roi_id=1, action_type="motor_pulse", pulse_duration=500
        )
        stim._trigger._t0 = 0
        out, dic = stim._decide()
        if int(out) == 1:
            self.assertIn("duration", dic)
            self.assertEqual(dic["duration"], 500)

    def test_decide_led_pulse_train_instruction(self):
        """Test LED pulse train instruction contains on/off/cycles."""
        stim = self._create_bound_stimulator(
            roi_id=1,
            action_type="led_pulse_train",
            pulse_on_ms=150,
            pulse_off_ms=250,
            pulse_cycles=10,
        )
        stim._trigger._t0 = 0
        out, dic = stim._decide()
        if int(out) == 1:
            self.assertIn("on_ms", dic)
            self.assertIn("off_ms", dic)
            self.assertIn("cycles", dic)


class TestComposedStimulatorChannelMaps(unittest.TestCase):
    """Test channel mapping for different action types."""

    def _create_stimulator(self, action_type):
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        return ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type=action_type,
        )

    def test_motor_uses_odd_channels(self):
        stim = self._create_stimulator("motor_pulse")
        # Motor channels should be odd
        if 1 in stim._roi_to_channel:
            self.assertEqual(stim._roi_to_channel[1] % 2, 1)

    def test_led_uses_even_channels(self):
        stim = self._create_stimulator("led_pulse")
        # LED channels should be even
        if 1 in stim._roi_to_channel:
            self.assertEqual(stim._roi_to_channel[1] % 2, 0)

    def test_valve_uses_even_channels(self):
        stim = self._create_stimulator("valve_pulse")
        if 1 in stim._roi_to_channel:
            self.assertEqual(stim._roi_to_channel[1] % 2, 0)


class TestComposedStimulatorYoking(unittest.TestCase):
    """Test the enable_yoking mechanism for focal and yoked ROIs."""

    YOKING_PAIRS = {1: 12, 3: 14, 5: 16, 7: 18, 9: 20}
    FOCAL_ROIS = [1, 3, 5, 7, 9]
    YOKED_ROIS = [12, 14, 16, 18, 20]

    def _create_bound_stimulator(self, roi_id=1, enable_yoking=False, **kwargs):
        """Create a stimulator bound to a tracker for the given ROI."""
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        defaults = {
            "hardware_connection": mock_hw,
            "trigger_type": "inactivity",
            "action_type": "motor_pulse",
            "min_inactive_time": 0,
            "stimulus_probability": 1.0,
            "enable_yoking": enable_yoking,
        }
        defaults.update(kwargs)
        stim = ComposedStimulator(**defaults)
        tracker = _make_mock_tracker(
            roi_id=roi_id,
            positions=[
                [{"xy_dist_log10x1000": -3000}],
                [{"xy_dist_log10x1000": -3000}],
            ],
            times=[199000, 200000],
        )
        stim.bind_tracker(tracker)
        return stim

    # ------------------------------------------------------------------ #
    # enable_yoking = False (default): normal independent behavior
    # ------------------------------------------------------------------ #

    def test_yoking_disabled_focal_roi_triggers_normally(self):
        """Focal ROI triggers normally with no yoked channel when yoking is disabled."""
        for roi_id in self.FOCAL_ROIS:
            with self.subTest(roi_id=roi_id):
                stim = self._create_bound_stimulator(roi_id=roi_id, enable_yoking=False)
                stim._trigger._t0 = 0
                out, instruction = stim._decide()
                self.assertEqual(int(out), 1, f"ROI {roi_id} should trigger")
                self.assertIn("channel", instruction)
                self.assertNotIn("_yoked_partner_channel", instruction)

    def test_yoking_disabled_yoked_roi_triggers_normally(self):
        """Yoked ROI triggers independently when yoking is disabled."""
        for roi_id in self.YOKED_ROIS:
            with self.subTest(roi_id=roi_id):
                stim = self._create_bound_stimulator(roi_id=roi_id, enable_yoking=False)
                stim._trigger._t0 = 0
                out, instruction = stim._decide()
                self.assertEqual(
                    int(out), 1, f"ROI {roi_id} should trigger independently"
                )
                self.assertIn("channel", instruction)
                self.assertNotIn("_yoked_partner_channel", instruction)

    # ------------------------------------------------------------------ #
    # enable_yoking = True: focal triggers both, yoked never self-triggers
    # ------------------------------------------------------------------ #

    def test_yoking_enabled_focal_roi_triggers_with_yoked_partner(self):
        """Focal ROI fires and includes the yoked partner channel in the instruction."""
        for focal_id, yoked_id in self.YOKING_PAIRS.items():
            with self.subTest(focal_roi=focal_id, yoked_roi=yoked_id):
                stim = self._create_bound_stimulator(
                    roi_id=focal_id, enable_yoking=True
                )
                stim._trigger._t0 = 0
                out, instruction = stim._decide()
                self.assertEqual(int(out), 1, f"Focal ROI {focal_id} should trigger")
                self.assertIn("channel", instruction)
                self.assertIn("_yoked_partner_channel", instruction)
                # The yoked channel must match the mapped channel for the paired ROI
                expected_yoked_channel = stim._roi_to_channel.get(yoked_id)
                self.assertEqual(
                    instruction["_yoked_partner_channel"],
                    expected_yoked_channel,
                    f"Yoked channel for ROI {focal_id}->{yoked_id} mismatch",
                )

    def test_yoking_enabled_yoked_roi_suppressed(self):
        """Yoked ROI never self-triggers when yoking is enabled."""
        for roi_id in self.YOKED_ROIS:
            with self.subTest(roi_id=roi_id):
                stim = self._create_bound_stimulator(roi_id=roi_id, enable_yoking=True)
                stim._trigger._t0 = 0
                out, instruction = stim._decide()
                self.assertEqual(
                    int(out), 0, f"Yoked ROI {roi_id} must not self-trigger"
                )
                self.assertEqual(instruction, {})

    def test_yoking_enabled_focal_roi_no_trigger_is_clean(self):
        """Focal ROI that doesn't meet trigger condition returns normally."""
        stim = self._create_bound_stimulator(
            roi_id=1, enable_yoking=True, min_inactive_time=9999
        )
        out, instruction = stim._decide()
        self.assertEqual(int(out), 0)
        self.assertNotIn("_yoked_partner_channel", instruction)

    # ------------------------------------------------------------------ #
    # _deliver with yoked partner channel
    # ------------------------------------------------------------------ #

    def test_deliver_sends_yoked_instruction(self):
        """_deliver batches the primary and yoked instructions into one atomic call."""
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        stim = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="motor_pulse",
            enable_yoking=True,
        )
        tracker = _make_mock_tracker(roi_id=1)
        stim.bind_tracker(tracker)

        yoked_channel = stim._roi_to_channel.get(12)
        # Call _deliver with a yoked partner
        stim._deliver(channel=1, duration=1000, _yoked_partner_channel=yoked_channel)

        # The primary and yoked instructions are sent as a single batched call
        # (one send_instruction with a list of two dicts), for atomic delivery.
        send_calls = mock_hw.send_instruction.call_args_list
        self.assertEqual(
            len(send_calls),
            1,
            "Expected a single batched send_instruction call (primary + yoked)",
        )

        # Inspect the single call's payload
        call = send_calls[0]
        payload = call.kwargs.get("instruction")
        if payload is None and call.args:
            payload = call.args[0]

        self.assertIsInstance(
            payload, list, "Batched call should pass a list of instructions"
        )
        self.assertEqual(len(payload), 2, "Batch should contain primary + yoked")

        primary, yoked = payload
        self.assertEqual(primary.get("channel"), 1)
        self.assertEqual(primary.get("duration"), 1000)
        self.assertEqual(
            yoked.get("channel"),
            yoked_channel,
            "Yoked instruction must target the yoked partner's channel",
        )
        # The yoked instruction should not leak the private key
        self.assertNotIn("_yoked_partner_channel", primary)
        self.assertNotIn("_yoked_partner_channel", yoked)

    def test_deliver_without_yoked_partner(self):
        """_deliver without _yoked_partner_channel sends only the primary instruction."""
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        stim = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="motor_pulse",
            enable_yoking=False,
        )
        tracker = _make_mock_tracker(roi_id=1)
        stim.bind_tracker(tracker)

        stim._deliver(channel=1, duration=1000)
        # Only the primary instruction from super()._deliver
        send_calls = mock_hw.send_instruction.call_args_list
        self.assertEqual(len(send_calls), 1, "Expected only the primary instruction")

    # ------------------------------------------------------------------ #
    # Yoking with LED action type
    # ------------------------------------------------------------------ #

    def test_yoking_enabled_led_action_focal_roi(self):
        """Yoking works correctly with LED pulse actions."""
        stim = self._create_bound_stimulator(
            roi_id=1, enable_yoking=True, action_type="led_pulse"
        )
        stim._trigger._t0 = 0
        out, instruction = stim._decide()
        self.assertEqual(int(out), 1)
        self.assertIn("_yoked_partner_channel", instruction)
        # LED channel map: ROI 12 → ch 10
        expected_yoked_channel = stim._roi_to_channel.get(12)
        self.assertEqual(instruction["_yoked_partner_channel"], expected_yoked_channel)

    def test_yoking_enabled_led_action_yoked_roi_suppressed(self):
        """Yoked ROI is suppressed regardless of action type."""
        stim = self._create_bound_stimulator(
            roi_id=12, enable_yoking=True, action_type="led_pulse"
        )
        stim._trigger._t0 = 0
        out, instruction = stim._decide()
        self.assertEqual(int(out), 0)
        self.assertEqual(instruction, {})

    # ------------------------------------------------------------------ #
    # Yoking configuration attributes
    # ------------------------------------------------------------------ #

    def test_yoking_attributes_initialized(self):
        """Yoking attributes are correctly set from constructor."""
        mock_hw = Mock()
        mock_hw.interrogate.side_effect = Exception("no module")
        stim_off = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="motor_pulse",
            enable_yoking=False,
        )
        self.assertFalse(stim_off._enable_yoking)
        self.assertEqual(set(stim_off._YOKED_ROIS), set(self.YOKED_ROIS))

        stim_on = ComposedStimulator(
            hardware_connection=mock_hw,
            trigger_type="inactivity",
            action_type="motor_pulse",
            enable_yoking=True,
        )
        self.assertTrue(stim_on._enable_yoking)
        self.assertEqual(stim_on._YOKING_PAIRS, self.YOKING_PAIRS)


if __name__ == "__main__":
    unittest.main()
