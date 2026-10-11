import unittest
import numpy as np
import cv2
from visionlab.perception import HSVDetector
from visionlab.evaluation import score_trial, wilson
from scripts.evaluate_perception import scene, pixel_counts

class EvaluationTests(unittest.TestCase):
    def samples(self, gap=0.02, contact=False, forward=0, visible=True):
        return [{'time': i*0.032, 'left': forward, 'right': forward, 'state': 'blocked',
                 'inference_ms': 1, 'contacts': ['DISTRACTOR_A'] if contact else [],
                 'target_visible': visible, 'target_gap_m': gap} for i in range(1, 315)]

    def test_wrong_object_stop_is_not_success(self):
        self.assertFalse(score_trial(self.samples(gap=0.2), 'standard', 0.032, 10)['success'])

    def test_nearer_distractor_prevents_success(self):
        samples=self.samples()
        for row in samples:
            row['target_nearest']=False
        self.assertFalse(score_trial(samples,'standard',0.032,10)['success'])

    def test_nearby_wrong_object_is_rejected_by_identity_review(self):
        from scripts.check_arrival_identity import conservative_score
        samples=self.samples(gap=0.04)
        for row in samples:
            row['robot_position']=[0,0,0]
        objects={'DISTRACTOR_A':[0.10,0,0.1],'DISTRACTOR_B':[0.4,0.4,0.1]}
        self.assertTrue(score_trial(samples,'standard',0.032,10)['success'])
        self.assertFalse(conservative_score(samples,objects,'standard',10)['success'])

    def test_contact_invalidates_arrival(self):
        row = score_trial(self.samples(contact=True), 'standard', 0.032, 10)
        self.assertTrue(row['collision'])
        self.assertFalse(row['success'])

    def test_stable_stop_succeeds_but_transient_stop_does_not(self):
        rows = self.samples()
        self.assertTrue(score_trial(rows, 'standard', 0.032, 10)['success'])
        for i, row in enumerate(rows):
            if i%8==0:
                row['state']='approach'
        self.assertFalse(score_trial(rows, 'standard', 0.032, 10)['success'])

    def test_missing_requires_no_forward_command(self):
        self.assertTrue(score_trial(self.samples(visible=False), 'missing', 0.032, 10)['success'])
        self.assertFalse(score_trial(self.samples(visible=False, forward=0.01), 'missing', 0.032, 10)['success'])

    def test_incomplete_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            score_trial(self.samples()[:30], 'standard', 0.032, 10)

    def test_perfect_small_sample_does_not_have_certain_interval(self):
        low, high = wilson(9,9)
        self.assertLess(low, 0.8)
        self.assertAlmostEqual(high,1)

    def test_each_hsv_class_selects_matching_box(self):
        image = np.full((72,96,3), 100, np.uint8)
        for center, color in [(15,(0,0,180)), (48,(180,0,0)), (80,(0,180,0))]:
            cv2.rectangle(image, (center-6,15), (center+6,65), color, -1)
        for name, center in [('red_box',15),('blue_box',48),('green_box',80)]:
            detected, _ = HSVDetector(name).detect(image)
            self.assertAlmostEqual(detected.x*95, center, delta=1)

    def test_missing_generator_has_no_positive_labels(self):
        _, labels = scene(np.random.default_rng(123), 'missing')
        self.assertEqual(int(labels.max()),0)

    def test_occlusion_removes_truth_not_only_pixels(self):
        _, base = scene(np.random.default_rng(123), 'standard')
        image, labels = scene(np.random.default_rng(123), 'occlusion')
        self.assertLess(np.count_nonzero(labels), np.count_nonzero(base))
        self.assertEqual(pixel_counts(labels>0, labels>0)[0], np.count_nonzero(labels))

if __name__ == '__main__':
    unittest.main()
