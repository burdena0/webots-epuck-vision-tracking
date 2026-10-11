import argparse
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['red_box', 'blue_box', 'green_box']

def sample(rng):
    top, bottom = rng.integers(40, 190), rng.integers(30, 160)
    image = np.broadcast_to(np.linspace(top, bottom, 72)[:, None, None], (72, 96, 3)).copy()
    image = np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((72, 96), np.uint8)
    # bright non-target colors and gray clutter stay background
    for _ in range(int(rng.integers(1, 5))):
        color = [(150, 150, 25), (25, 150, 150), (150, 25, 150), (80, 80, 80)][int(rng.integers(4))]
        x, y = int(rng.integers(96)), int(rng.integers(72))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(4, 25)), y + int(rng.integers(5, 30))), color, -1)
    for target in rng.permutation(3):
        if rng.random() < 0.25:
            continue
        w, h = int(rng.integers(5, 28)), int(rng.integers(12, 66))
        x, y = int(rng.integers(-w // 3, 90)), int(rng.integers(0, 62))
        color = rng.integers(0, 35, 3)
        color[[0, 2, 1][target]] = rng.integers(100, 256)
        cv2.rectangle(image, (x, y), (x + w, y + h), tuple(map(int, color)), -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), int(target) + 1, -1)
        cv2.rectangle(image, (x, y), (x + w // 3, y + h), tuple(int(c * 0.65) for c in color), -1)
        if rng.random() < 0.25:
            oy = y + h // 2
            cv2.rectangle(image, (x - 1, oy), (x + w + 1, oy + 3), (70, 70, 70), -1)
            cv2.rectangle(mask, (x - 1, oy), (x + w + 1, oy + 3), 0, -1)
    image = np.clip(image * rng.uniform(0.55, 1.2), 0, 255).astype(np.uint8)
    return image, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data/targets_demo')
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--validation', type=int, default=320)
    parser.add_argument('--test', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1:
        parser.error('each split needs images')
    if (args.output / 'manifest.json').exists():
        parser.error('that dataset already exists; use a new output folder')
    for folder in ('images', 'masks'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {'classes': NAMES, 'source': 'procedural rgb boxes with exact drawn masks', 'seed': args.seed, 'samples': []}
    for offset, (split, count) in enumerate([('train', args.train), ('validation', args.validation), ('test', args.test)]):
        rng = np.random.default_rng(args.seed + offset)
        for index in range(count):
            name = f'{split}_{index:05d}.png'
            rgb, mask = sample(rng)
            if not cv2.imwrite(str(args.output / 'images' / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError('couldnt save the image')
            if not cv2.imwrite(str(args.output / 'masks' / name), mask):
                raise OSError('couldnt save the mask')
            manifest['samples'].append({'image': 'images/' + name, 'mask': 'masks/' + name, 'split': split})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', newline='\n')
    print(f'created {len(manifest["samples"])} labeled images at {args.output}', flush=True)

if __name__ == '__main__':
    main()
