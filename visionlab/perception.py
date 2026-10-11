from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class Detection:
    x: float
    area: float
    confidence: float
    box: tuple[int, int, int, int]

def component(mask, scores=None, min_area=0.003):
    # use the biggest blob so random red pixels dont steer the robot
    mask = np.asarray(mask, dtype=np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(clean)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[index, cv2.CC_STAT_AREA]) / mask.size
    if area < min_area:
        return None
    x, y, w, h, _ = map(int, stats[index])
    if h < w * 0.5:
        return None
    confidence = 1.0 if scores is None else float(scores[labels == index].mean())
    return Detection(float(centers[index, 0]) / (mask.shape[1] - 1), area, confidence, (x, y, w, h))

class HSVDetector:
    # Class-specific ranges are fixed before audit; hue uses OpenCV's 0..179 scale.
    HUES = {'red_box': ((0, 10), (170, 179)), 'blue_box': ((100, 130),), 'green_box': ((40, 85),)}

    def __init__(self, target='red_box', saturation=100, value=60):
        if target not in self.HUES:
            raise ValueError(f'unknown HSV target {target}')
        if not 0 <= saturation <= 255 or not 0 <= value <= 255:
            raise ValueError('HSV saturation and value must be in 0..255')
        self.target, self.saturation, self.value = target, saturation, value

    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        for low, high in self.HUES[self.target]:
            mask |= cv2.inRange(hsv, (low, self.saturation, self.value), (high, 255, 255))
        return component(mask), mask

class OriginalDetector:
    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
        mask |= cv2.inRange(hsv, (170, 100, 80), (179, 255, 255))
        m = cv2.moments(mask)
        if not m['m00']:
            return None, mask
        return Detection(m['m10'] / m['m00'] / (bgr.shape[1] - 1), float(np.count_nonzero(mask)) / mask.size, 1.0, (0, 0, 0, 0)), mask
