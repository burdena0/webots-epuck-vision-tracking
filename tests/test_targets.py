import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scripts.train_targets import inside, load_data
from visionlab.targets import CatalogDetector

ROOT = Path(__file__).resolve().parents[1]

class TargetSelectionTests(unittest.TestCase):
    def frame(self, excluded=None):
        frame = np.full((240, 320, 3), 100, np.uint8)
        for name, center, color in [('red_box', 50, (8, 8, 160)), ('blue_box', 160, (160, 8, 8)), ('green_box', 270, (8, 160, 8))]:
            if name != excluded:
                cv2.rectangle(frame, (center - 15, 60), (center + 15, 220), color, -1)
        return frame
    def test_selected_class_ignores_other_visible_targets(self):
        for name, center in [('red_box', 50), ('blue_box', 160), ('green_box', 270)]:
            with self.subTest(target=name):
                detector = CatalogDetector(ROOT / 'models/targets', name)
                found, _ = detector.detect(self.frame())
                self.assertIsNotNone(found)
                self.assertLess(abs(found.x * 319 - center), 5)
    def test_missing_selected_class_is_not_replaced_by_a_distractor(self):
        for name in ('red_box', 'blue_box', 'green_box'):
            with self.subTest(target=name):
                detector = CatalogDetector(ROOT / 'models/targets', name)
                self.assertIsNone(detector.detect(self.frame(excluded=name))[0])
    def test_unknown_class_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'unknown target'):
            CatalogDetector(ROOT / 'models/targets', 'purple_box')

class TargetDataTests(unittest.TestCase):
    def test_paths_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'inside'):
                inside(Path(folder), '../elsewhere.png')
    def test_duplicate_images_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((72, 96, 3), 80, np.uint8)
            mask = np.zeros((72, 96), np.uint8)
            mask[20:50, 30:60] = 1
            cv2.imwrite(str(root / 'image.png'), image)
            cv2.imwrite(str(root / 'mask.png'), mask)
            samples = [{'image': 'image.png', 'mask': 'mask.png', 'split': split} for split in ('train', 'validation', 'test')]
            (root / 'manifest.json').write_text(json.dumps({'classes': ['red_box'], 'samples': samples}))
            with self.assertRaisesRegex(ValueError, 'different splits'):
                load_data(root)
    def test_mask_ids_outside_catalog_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / 'image.png'), np.full((72, 96, 3), 80, np.uint8))
            cv2.imwrite(str(root / 'mask.png'), np.full((72, 96), 2, np.uint8))
            samples = [{'image': 'image.png', 'mask': 'mask.png', 'split': 'train'}]
            (root / 'manifest.json').write_text(json.dumps({'classes': ['red_box'], 'samples': samples}))
            with self.assertRaisesRegex(ValueError, 'outside the target catalog'):
                load_data(root)

if __name__ == '__main__':
    unittest.main()
