from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from .perception import component

WIDTH, HEIGHT = 96, 72

class TargetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 1, 1))
    def forward(self, image):
        return self.layers(image)

def input_tensor(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255

class NeuralDetector:
    def __init__(self, model_path):
        torch.set_num_threads(2)
        if not Path(model_path).is_file():
            raise FileNotFoundError('train the model first: python scripts/train_model.py')
        self.model = torch.jit.load(str(model_path), map_location='cpu').eval()
        self.model(input_tensor(np.zeros((HEIGHT, WIDTH, 3), np.uint8)))
    def detect(self, bgr):
        with torch.inference_mode():
            scores = self.model(input_tensor(bgr)).sigmoid()[0, 0].numpy()
        scores = cv2.resize(scores, (bgr.shape[1], bgr.shape[0]))
        mask = (scores >= 0.65).astype(np.uint8) * 255
        return component(mask, scores), mask
