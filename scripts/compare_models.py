import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_model import dataset, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        report = json.loads((ROOT / 'reports/model_comparison.json').read_text())
        plot_comparison(report['results'])
        return
    torch.set_num_threads(4)
    baseline = torch.jit.load(str(ROOT / 'models/baseline_target_net.pt')).eval()
    improved = torch.jit.load(str(ROOT / 'models/target_net.pt')).eval()
    rows = []
    specifications = [('fresh', 1000, 202610220), ('dim', 320, 202610221), ('blurred', 320, 202610222)]
    for name, count, seed in specifications:
        x, y = dataset(count, seed)
        if name == 'dim':
            x = x * 0.65
        elif name == 'blurred':
            images = x.numpy().transpose(0, 2, 3, 1)
            images = np.stack([cv2.GaussianBlur(image, (3, 3), 0.8) for image in images])
            x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
        # score both models on the exact same images
        row = {'case': name, 'seed': seed, 'images': count,
               'baseline': metrics(baseline, x, y), 'improved': metrics(improved, x, y)}
        row['iou_gain_percentage_points'] = round((row['improved']['mask_iou'] - row['baseline']['mask_iou']) * 100, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'protocol': 'fresh seeds after validation-only checkpoint selection; fixed 0.65 mask threshold',
              'development_note': 'earlier v2 audit became diagnostic after its dim-light regression was used to choose brightness augmentation; these seeds were reserved for v3',
              'model_sha256': {key: hashlib.sha256((ROOT / 'models' / name).read_bytes()).hexdigest()
                               for key, name in [('baseline', 'baseline_target_net.pt'), ('improved', 'target_net.pt')]},
              'scope': 'procedural synthetic images; dim and blur are stress checks, not webots accuracy', 'results': rows}
    (ROOT / 'reports/model_comparison.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    plot_comparison(rows)

def plot_comparison(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    index = np.arange(len(rows))
    for offset, key, label, color in [(-0.18, 'baseline', 'original training', '#718a9c'), (0.18, 'improved', 'improved training', '#146c74')]:
        values = [row[key]['mask_iou'] * 100 for row in rows]
        bars = axes[0].bar(index + offset, values, width=0.36, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.2f', fontsize=8)
    axes[0].set(xticks=index, xticklabels=[row['case'] for row in rows], ylim=(0, 106), ylabel='mask iou (%)', title='same images, two trained models')
    axes[0].legend(fontsize=8, loc='lower right')
    for path, label, color in [('baseline/training.json', 'original: 1,600 images', '#718a9c'), ('training.json', 'improved: 4,800 images', '#146c74')]:
        history = json.loads((ROOT / 'reports' / path).read_text())['history']
        axes[1].plot([row['epoch'] for row in history], [row['mask_iou'] for row in history], label=label, color=color)
        if path == 'training.json':
            axes[1].plot([row['epoch'] for row in history], [row['dim_validation_iou'] for row in history],
                         label='improved: dim validation', color=color, linestyle='--', alpha=0.7)
    axes[1].set(xlabel='epoch', ylabel='validation mask iou', ylim=(0.7, 1), title='checkpoint selection uses validation')
    axes[1].legend(fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/model_improvement.png', dpi=160)

if __name__ == '__main__':
    main()
