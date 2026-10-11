import argparse
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
from visionlab.neural import TargetNet, WIDTH, HEIGHT

def dataset(count, seed):
    rng = np.random.default_rng(seed)
    images = np.empty((count, HEIGHT, WIDTH, 3), dtype=np.uint8)
    masks = np.zeros((count, HEIGHT, WIDTH), dtype=np.uint8)
    for index in range(count):
        # labels come from the shapes we drew. no hsv teacher labels
        top, bottom = rng.integers(35, 210, 3), rng.integers(30, 160, 3)
        image = np.broadcast_to(np.linspace(top, bottom, HEIGHT)[:, None, :], (HEIGHT, WIDTH, 3)).copy()
        image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        for _ in range(int(rng.integers(1, 7))):
            color = rng.integers(25, 225, 3)
            if color[0] > max(color[1:]) * 1.25:
                color[1] = color[0]
            x, y = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
            cv2.rectangle(image, (x, y), (x + int(rng.integers(3, 25)), y + int(rng.integers(3, 40))), tuple(map(int, color)), -1)
        if rng.random() < 0.78:
            w, h = int(rng.integers(5, 36)), int(rng.integers(10, 69))
            x, y = int(rng.integers(-w // 3, WIDTH - 2)), int(rng.integers(0, HEIGHT - 10))
            color = (int(rng.integers(85, 256)), int(rng.integers(0, 42)), int(rng.integers(0, 42)))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(masks[index], (x, y), (x + w, y + h), 1, -1)
            cv2.rectangle(image, (x, y), (x + w // 3, y + h), tuple(int(c * 0.65) for c in color), -1)
            if rng.random() < 0.3:
                oy = y + h // 2
                cv2.rectangle(image, (x - 2, oy), (x + w + 2, oy + 4), (70, 70, 70), -1)
                cv2.rectangle(masks[index], (x - 2, oy), (x + w + 2, oy + 4), 0, -1)
        for _ in range(5):
            cv2.circle(image, (int(rng.integers(WIDTH)), int(rng.integers(HEIGHT))), 0, (210, 15, 15), -1)
        images[index] = image
    return (torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 255,
            torch.from_numpy(masks[:, None].astype(np.float32)))

def metrics(model, x, y):
    model.eval()
    intersection = union = false_positive = missing = 0
    errors = []
    with torch.inference_mode():
        for start in range(0, len(x), 64):
            pred = (model(x[start:start + 64]).sigmoid() >= 0.65).numpy()[:, 0]
            truth = y[start:start + 64].numpy()[:, 0].astype(bool)
            intersection += int((pred & truth).sum())
            union += int((pred | truth).sum())
            for p, t in zip(pred, truth):
                if t.sum() == 0:
                    false_positive += int(p.sum() >= 0.003 * p.size)
                elif p.sum() == 0:
                    missing += 1
                else:
                    errors.append(abs(np.nonzero(p)[1].mean() - np.nonzero(t)[1].mean()))
    return {'mask_iou': round(intersection / max(union, 1), 4), 'centroid_mae_pixels_at_96_width': round(float(np.mean(errors)), 3) if errors else None,
            'missed_positive_images': missing, 'false_positive_negative_images': false_positive, 'images': len(x)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--train-images', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--run-name', default='default')
    args = parser.parse_args()
    if args.epochs < 1 or args.train_images < 64 or args.batch_size < 1 or args.lr <= 0:
        parser.error('use positive settings and at least 64 training images')
    if not args.run_name.replace('-', '').replace('_', '').isalnum():
        parser.error('run name needs letters, numbers, underscores or hyphens')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_x, train_y = dataset(args.train_images, args.seed)
    val_x, val_y = dataset(320, args.seed + 1)
    model = TargetNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
    best_iou, best_state, history = -1, None, []
    run = ROOT / 'runs' / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(train_x))
        losses = []
        for batch in indices.split(args.batch_size):
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        score = metrics(model, val_x, val_y)
        history.append({'epoch': epoch + 1, 'loss': round(float(np.mean(losses)), 5), **score})
        if score['mask_iou'] > best_iou:
            best_iou = score['mask_iou']
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
        (run / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        torch.save({'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'settings': vars(args)}, run / 'last_checkpoint.pt')
    torch.save({'model': best_state, 'settings': vars(args), 'best_validation_iou': best_iou}, run / 'best_checkpoint.pt')
    model.load_state_dict(best_state)
    model.eval()
    models, reports = ROOT / 'models', ROOT / 'reports'
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    torch.jit.trace(model, train_x[:1]).save(str(models / 'target_net.pt'))
    test_x, test_y = dataset(320, args.seed + 2)
    report = {'data': 'procedural rgb rectangles; exact visible-shape masks; synthetic only', 'train_seed': args.seed,
              'validation_seed': args.seed + 1, 'test_seed': args.seed + 2, 'train_images': len(train_x), 'validation_images': len(val_x),
              'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': round(time.perf_counter() - started, 2),
              'settings': vars(args), 'torch_version': torch.__version__, 'history': history, 'test': metrics(model, test_x, test_y)}
    intersection = union = 0
    for image, truth in zip(test_x, test_y):
        rgb = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        pred = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)) > 0
        pred |= cv2.inRange(hsv, (170, 100, 60), (179, 255, 255)) > 0
        actual = truth.numpy()[0] > 0
        intersection += int((pred & actual).sum())
        union += int((pred | actual).sum())
    report['hsv_test_mask_iou'] = round(intersection / max(union, 1), 4)
    (reports / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    (run / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    with torch.inference_mode():
        masks = model(test_x[:12]).sigmoid().numpy()[:, 0] >= 0.65
    montage = []
    for i in range(12):
        rgb = (test_x[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[masks[i]] = (0, 255, 0)
        montage.append(np.hstack([rgb, overlay]))
    preview = np.vstack([np.hstack(montage[i:i + 3]) for i in range(0, 12, 3)])
    cv2.imwrite(str(reports / 'synthetic_predictions.png'), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['loss'] for row in history])
    axes[0].set(title='training loss', xlabel='epoch', ylabel='weighted binary cross entropy')
    axes[1].plot(epochs, [row['mask_iou'] for row in history])
    axes[1].set(title='held-out validation', xlabel='epoch', ylabel='mask iou', ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports / 'learning_curves.png', dpi=150)
    fig.savefig(run / 'learning_curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report['test']), flush=True)

if __name__ == '__main__':
    main()
