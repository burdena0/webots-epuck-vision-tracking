from pathlib import Path
import unittest
import cv2
import numpy as np
from visionlab.neural import NeuralDetector

class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = NeuralDetector(Path(__file__).resolve().parents[1] / 'models/target_net.pt')
    def test_unseen_box_positions_and_shading(self):
        for center in (50, 160, 270):
            frame = np.full((240, 320, 3), 100, np.uint8)
            cv2.rectangle(frame, (center - 15, 60), (center + 15, 220), (8, 8, 120), -1)
            found, mask = self.detector.detect(frame)
            self.assertIsNotNone(found)
            self.assertLess(abs(found.x * 319 - center), 5)
            self.assertGreater(found.confidence, 0.65)
            self.assertEqual(mask.shape, frame.shape[:2])
    def test_gray_scene_and_tiny_specks_are_rejected(self):
        frame = np.full((240, 320, 3), 100, np.uint8)
        frame[50, 60] = (0, 0, 255)
        self.assertIsNone(self.detector.detect(frame)[0])

if __name__ == '__main__':
    unittest.main()
