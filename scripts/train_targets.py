import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visionlab.targets import TargetCatalogNet, class_names
from visionlab.neural import WIDTH, HEIGHT

def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('dataset paths must stay inside the dataset folder')
    return path

def load_data(root):
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    names = class_names(manifest['classes'])
    splits = {name: ([], []) for name in ('train', 'validation', 'test')}
    seen = {}
    for row in manifest['samples']:
        split = row['split']
        if split not in splits:
            raise ValueError('split must be train, validation or test')
        bgr = cv2.imread(str(inside(root, row['image'])), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(inside(root, row['mask'])), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None or mask.ndim != 2 or mask.dtype != np.uint8 or mask.shape != bgr.shape[:2]:
            raise ValueError('each image needs a matching single-channel 8-bit png mask')
        if int(mask.max()) > len(names):
            raise ValueError('mask label is outside the target catalog')
        identity = hashlib.sha256(bgr.tobytes()).hexdigest()
        if identity in seen and seen[identity] != split:
            raise ValueError('the same image appears in different splits')
        seen[identity] = split
        rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)
        resized = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        splits[split][0].append(rgb)
        splits[split][1].append(resized)
    tensors = {}
    for split, (images, masks) in splits.items():
        if not images or set(np.unique(np.stack(masks))) - {0} != set(range(1, len(names) + 1)):
            raise ValueError(f'{split} must contain visible examples of every target class')
        tensors[split] = (torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).float() / 255,
                          torch.from_numpy(np.stack(masks).astype(np.int64)))
    return names, tensors, hashlib.sha256(manifest_bytes).hexdigest()

def score(model, x, y, names):
    intersection = np.zeros(len(names), dtype=np.int64)
    union = np.zeros(len(names), dtype=np.int64)
    misses = np.zeros(len(names), dtype=np.int64)
    false_positives = np.zeros(len(names), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 64):
            probabilities = model(x[start:start + 64]).softmax(1).numpy()
            truth = y[start:start + 64].numpy()
            for index in range(len(names)):
                pred, actual = probabilities[:, index + 1] >= 0.65, truth == index + 1
                intersection[index] += (pred & actual).sum()
                union[index] += (pred | actual).sum()
                area = pred.sum((1, 2))
                present = actual.sum((1, 2)) > 0
                misses[index] += ((area == 0) & present).sum()
                false_positives[index] += ((area >= 0.003 * HEIGHT * WIDTH) & ~present).sum()
    ious = intersection / np.maximum(union, 1)
    return {'macro_iou': round(float(ious.mean()), 4), 'images': len(x),
            'classes': {name: {'mask_iou': round(float(ious[i]), 4), 'empty_mask_misses': int(misses[i]),
                               'negative_image_false_positives': int(false_positives[i])} for i, name in enumerate(names)}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/targets')
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--lr', type=float, default=0.003)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error('epochs and learning rate must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    names, data, manifest_hash = load_data(args.data)
    model = TargetCatalogNet(len(names))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [2.5] * len(names)))
    best, best_state, selected_epoch, history = -1, None, None, []
    train_x, train_y = data['train']
    run = ROOT / 'runs/targets'
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in torch.randperm(len(train_x)).split(64):
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = score(model, *data['validation'], names)
        history.append({'epoch': epoch + 1, 'loss': round(float(np.mean(losses)), 5), 'validation': validation})
        if validation['macro_iou'] > best:
            best, selected_epoch = validation['macro_iou'], epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        scheduler.step()
        print(json.dumps(history[-1]), flush=True)
        (run / 'history.json').write_text(json.dumps(history, indent=2) + '\n', newline='\n')
    model.load_state_dict(best_state)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model_file = args.output / 'target_catalog.pt'
    torch.jit.trace(model, train_x[:1]).save(str(model_file))
    metadata = {'classes': names, 'input_size': [WIDTH, HEIGHT], 'threshold': 0.65,
                'model_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(), 'manifest_sha256': manifest_hash}
    (args.output / 'catalog.json').write_text(json.dumps(metadata, indent=2) + '\n', newline='\n')
    report = {'data': args.data.name, 'manifest_sha256': manifest_hash, 'classes': names,
              'split_images': {key: len(value[0]) for key, value in data.items()}, 'epochs': args.epochs,
              'seed': args.seed, 'lr': args.lr, 'parameters': sum(p.numel() for p in model.parameters()),
              'torch_version': torch.__version__, 'training_seconds': round(time.perf_counter() - started, 2),
              'checkpoint_selection': 'validation macro foreground mask iou only', 'selected_epoch': selected_epoch,
              'history': history, 'test': score(model, *data['test'], names), 'model_sha256': metadata['model_sha256']}
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'target_training.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    torch.save({'model': best_state, 'classes': names, 'selected_epoch': selected_epoch}, run / 'best_checkpoint.pt')
    print(json.dumps(report['test']), flush=True)

if __name__ == '__main__':
    main()
