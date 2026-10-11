import hashlib
import json
from pathlib import Path
import re
import cv2
import numpy as np
import torch
from torch import nn
from .neural import WIDTH, HEIGHT, input_tensor
from .perception import component

def class_names(names):
    if not isinstance(names, list) or not 1 <= len(names) <= 254:
        raise ValueError('use 1 to 254 named target classes')
    if any(not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) for name in names) or len(set(names)) != len(names):
        raise ValueError('class names need unique lowercase names without spaces')
    return names

class TargetCatalogNet(nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, targets + 1, 1))
    def forward(self, image):
        return self.layers(image)

class CatalogDetector:
    def __init__(self, bundle, target):
        bundle = Path(bundle)
        metadata = json.loads((bundle / 'catalog.json').read_text())
        names = class_names(metadata['classes'])
        if target not in names:
            raise ValueError(f'unknown target {target}; choose from: {", ".join(names)}')
        if metadata['input_size'] != [WIDTH, HEIGHT] or metadata['threshold'] != 0.65:
            raise ValueError('this controller needs a 96 x 72 model with threshold 0.65')
        path = bundle / 'target_catalog.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['model_sha256']:
            raise ValueError('model and catalog metadata dont match; retrain the bundle')
        self.target, self.index, self.classes = target, names.index(target) + 1, names
        torch.set_num_threads(2)
        self.model = torch.jit.load(str(path), map_location='cpu').eval()
        with torch.inference_mode():
            output = self.model(input_tensor(np.zeros((HEIGHT, WIDTH, 3), np.uint8)))
        if tuple(output.shape) != (1, len(names) + 1, HEIGHT, WIDTH):
            raise ValueError('model outputs dont match the target catalog')
    def detect(self, bgr):
        with torch.inference_mode():
            scores = self.model(input_tensor(bgr)).softmax(1)[0, self.index].numpy()
        scores = cv2.resize(scores, (bgr.shape[1], bgr.shape[0]))
        mask = (scores >= 0.65).astype(np.uint8) * 255
        return component(mask, scores), mask
