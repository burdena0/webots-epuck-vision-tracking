import unittest
import numpy as np
from visionlab.control import Policy
from visionlab.perception import Detection, HSVDetector

def target(x=0.5, confidence=1.0):
    return Detection(x, 0.1, confidence, (0, 0, 1, 1))

class SafetyTests(unittest.TestCase):
    def test_obstacle_overrides_off_center_turn(self):
        for sensor in (0, 1, 6, 7):
            readings = [65] * 8
            readings[sensor] = 190
            command = Policy().step(target(0.9), readings, 0.032)
            self.assertEqual((command.left, command.right, command.state), (0, 0, 'blocked'))
    def test_hysteresis_keeps_stop_at_noisy_threshold(self):
        policy = Policy()
        for reading, state in [(150, 'blocked'), (125, 'blocked'), (101, 'blocked'), (99, 'approach')]:
            self.assertEqual(policy.step(target(), [reading] * 8, 0.032).state, state)
    def test_loss_stops_immediately_then_only_searches_in_place(self):
        policy = Policy()
        for _ in range(20):
            policy.step(target(), [65] * 8, 0.032)
        first = policy.step(None, [65] * 8, 0.032)
        self.assertEqual((first.left, first.right), (0, 0))
        for _ in range(160):
            command = policy.step(None, [65] * 8, 0.032)
            self.assertAlmostEqual(command.left + command.right, 0)
        self.assertEqual(command.state, 'lost')
    def test_faults_stop(self):
        for readings in ([65] * 7, [float('nan')] * 8):
            self.assertEqual(Policy().step(target(), readings, 0.032).state, 'sensor_fault')
        self.assertEqual(Policy().step(target(float('nan')), [65] * 8, 0.032).state, 'vision_fault')
        self.assertEqual(Policy().step(target(confidence=float('nan')), [65] * 8, 0.032).state, 'vision_fault')
        self.assertEqual(Policy().step(target(), [65] * 8, float('nan')).state, 'sensor_fault')
    def test_low_confidence_never_drives_forward(self):
        policy = Policy()
        for _ in range(60):
            command = policy.step(target(confidence=0.4), [65] * 8, 0.032)
            self.assertAlmostEqual(command.left + command.right, 0)
    def test_turn_direction_and_speed_ramp(self):
        for x in (0.15, 0.85):
            policy, last = Policy(), (0, 0)
            for _ in range(40):
                command = policy.step(target(x), [65] * 8, 0.032)
                for old, new in zip(last, (command.left, command.right)):
                    self.assertLessEqual(abs(old - new), 6 * 0.032 + 1e-8)
                    self.assertLessEqual(abs(new), 3)
                last = command.left, command.right
            self.assertEqual(command.left > command.right, x > 0.5)

class VisionTests(unittest.TestCase):
    def test_tiny_specks_are_not_a_target(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[60, 80] = (0, 0, 255)
        self.assertIsNone(HSVDetector().detect(frame)[0])
        frame[160:220, :] = (0, 0, 255)
        self.assertIsNone(HSVDetector().detect(frame)[0])
    def test_largest_target_wins_over_small_distractor(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[80:160, 200:260] = (0, 0, 255)
        frame[80:90, 50:60] = (0, 0, 255)
        self.assertGreater(HSVDetector().detect(frame)[0].x, 0.65)

if __name__ == '__main__':
    unittest.main()
