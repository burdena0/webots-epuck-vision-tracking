"""Matched, frozen-weight catalog CNN / validation-calibrated HSV evaluation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_target_data import NAMES, sample
from visionlab.targets import CatalogDetector
from visionlab.perception import HSVDetector, component

CASES = ('standard', 'dim', 'blur', 'occlusion', 'near_color', 'missing')
CALIBRATION_SEED = 810110001
AUDIT_SEEDS = {case: 810120000 + i * 1000 for i, case in enumerate(CASES)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scene(rng, case):
    rgb, labels = sample(rng)
    if case == 'dim':
        rgb = np.clip(rgb.astype(float) * 0.35, 0, 255).astype(np.uint8)
    elif case == 'blur':
        rgb = cv2.GaussianBlur(rgb, (5, 5), 1.2)
    elif case == 'occlusion':
        # Physical opaque strip: overwrite both appearance and visible-pixel truth.
        y = int(rng.integers(15, 50))
        h = int(rng.integers(8, 17))
        rgb[y:y+h] = (80, 80, 80)
        labels[y:y+h] = 0
    elif case == 'near_color':
        # Semantically background despite similar hues: oranges, cyans, purples.
        for _ in range(3):
            x, y = int(rng.integers(0, 75)), int(rng.integers(0, 50))
            color = [(180, 75, 15), (15, 150, 175), (110, 15, 180)][int(rng.integers(3))]
            cv2.rectangle(rgb, (x, y), (x+18, y+21), color, -1)
            cv2.rectangle(labels, (x, y), (x+18, y+21), 0, -1)
    elif case == 'missing':
        # No trained class is present; retain noisy background and near-color clutter.
        rgb[labels > 0] = (75, 75, 75)
        labels[:] = 0
    elif case != 'standard':
        raise ValueError(f'unknown case {case}')
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels

def generate(count, seed, case):
    rng = np.random.default_rng(seed)
    return [scene(rng, case) for _ in range(count)]

def pixel_counts(prediction, truth):
    p, t = prediction > 0, truth > 0
    return int((p & t).sum()), int((p | t).sum())

def calibrate(count):
    # No audit images are consulted. Calibration mixes normal and dim development data.
    data = generate(count, CALIBRATION_SEED, 'standard') + generate(count, CALIBRATION_SEED+1, 'dim')
    trials = []
    for saturation in (50, 75, 100):
        for value in (20, 40, 60, 80):
            inter, union = np.zeros(3), np.zeros(3)
            for index, name in enumerate(NAMES):
                detector = HSVDetector(name, saturation, value)
                for image, labels in data:
                    _, mask = detector.detect(image)
                    i, u = pixel_counts(mask, labels == index+1)
                    inter[index] += i
                    union[index] += u
            trials.append({'saturation': saturation, 'value': value, 'macro_iou': float(np.mean(inter / np.maximum(union, 1)))})
    best = max(trials, key=lambda row: row['macro_iou'])
    return {'protocol': '12 S/V candidates on normal+dim calibration only; fixed class hue bands; first candidate breaks ties',
            'seeds': [CALIBRATION_SEED, CALIBRATION_SEED+1], 'images': len(data),
            'selected': {key: best[key] for key in ('saturation', 'value')}, 'candidates': trials}

def summarize(counts, diagnostics, timings):
    # Counts keep the image as the resampling unit, with three matched classes.
    inter = counts[:, :, 0].sum(0)
    union = counts[:, :, 1].sum(0)
    classes = {}
    for index, name in enumerate(NAMES):
        d = diagnostics[index]
        classes[name] = {'mask_iou': None if union[index] == 0 else float(inter[index] / union[index]), **d,
                         'centroid_mae_pixels': float(np.mean(d['centroid_errors'])) if d['centroid_errors'] else None}
        classes[name].pop('centroid_errors')
    present = union > 0
    return {'macro_iou': float(np.mean(inter[present] / union[present])) if present.any() else None,
            'classes': classes,
            'latency_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)), 'samples': len(timings)}}

def paired_interval(cnn, hsv, seed, repetitions=1000):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(cnn), len(cnn))
        a, b = cnn[chosen].sum(0), hsv[chosen].sum(0)
        if np.any(a[:, 1] == 0) or np.any(b[:, 1] == 0):
            continue
        differences.append(float(np.mean(a[:, 0] / a[:, 1]) - np.mean(b[:, 0] / b[:, 1])))
    if not differences:
        return None
    return {'low': float(np.percentile(differences, 2.5)), 'high': float(np.percentile(differences, 97.5)),
            'resamples': len(differences), 'unit': 'paired synthetic scene, retaining all three classes'}

def run_case(data, detectors, case):
    counts = {method: np.zeros((len(data), 3, 2), np.int64) for method in detectors}
    diagnostics = {method: [{'positive_images': 0, 'negative_images': 0, 'detection_misses': 0,
                             'false_positive_detections': 0, 'centroid_errors': []} for _ in NAMES] for method in detectors}
    timings = {method: [] for method in detectors}
    examples = []
    digest = hashlib.sha256()
    for row, (image, labels) in enumerate(data):
        digest.update(image.tobytes())
        digest.update(labels.tobytes())
        masks = {}
        # Alternate method order to reduce systematic ordering bias in timing.
        for method in (('cnn', 'hsv') if row % 2 == 0 else ('hsv', 'cnn')):
            masks[method] = []
            for index, detector in enumerate(detectors[method]):
                start = time.perf_counter_ns()
                detected, mask = detector.detect(image)
                timings[method].append((time.perf_counter_ns() - start) / 1e6)
                actual = (labels == index+1).astype(np.uint8)*255
                counts[method][row, index] = pixel_counts(mask, actual)
                d = diagnostics[method][index]
                present = bool(np.count_nonzero(actual))
                d['positive_images' if present else 'negative_images'] += 1
                d['detection_misses'] += int(present and detected is None)
                d['false_positive_detections'] += int(not present and detected is not None)
                ground = component(actual)
                if detected is not None and ground is not None:
                    d['centroid_errors'].append(abs(detected.x - ground.x) * (image.shape[1]-1))
                masks[method].append(mask)
        if row < 3:
            palette = np.array([[0,0,0], [0,0,255], [255,0,0], [0,255,0]], np.uint8)
            truth_view = palette[labels]
            overlays = []
            for method in ('cnn','hsv'):
                view = np.zeros_like(image)
                for index, mask in enumerate(masks[method]):
                    view[mask>0] = palette[index+1]
                overlays.append(view)
            examples.append(np.hstack([image, truth_view, *overlays]))
    result = {'case': case, 'seed': AUDIT_SEEDS[case], 'images': len(data), 'dataset_sha256': digest.hexdigest(),
              **{method: summarize(counts[method], diagnostics[method], timings[method]) for method in detectors}}
    result['cnn_minus_hsv_iou_95pct_bootstrap'] = paired_interval(counts['cnn'], counts['hsv'], AUDIT_SEEDS[case]+99)
    return result, np.vstack(examples)

def report_markdown(report):
    lines = ['# Matched catalog perception evaluation', '',
             'Frozen 1,460-parameter CNN versus an HSV baseline calibrated on separate development scenes. Both methods see identical audit pixels and labels, select the same class, and share component filtering.', '',
             '| Synthetic case | Images | CNN macro IoU | HSV macro IoU | CNN median / p95 ms | HSV median / p95 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['results']:
        def fmt(v): return '-' if v is None else f'{v*100:.2f}%'
        c, h = row['cnn']['latency_ms'], row['hsv']['latency_ms']
        lines.append(f"| {row['case']} | {row['images']} | {fmt(row['cnn']['macro_iou'])} | {fmt(row['hsv']['macro_iou'])} | {c['median']:.3f} / {c['p95']:.3f} | {h['median']:.3f} / {h['p95']:.3f} |")
    lines += ['', '## Detection errors', '', '| Case | Method | Misses / positive class-images | False detections / negative class-images |', '| --- | --- | ---: | ---: |']
    for row in report['results']:
        for method in ('cnn', 'hsv'):
            classes = row[method]['classes'].values()
            lines.append(f"| {row['case']} | {method} | {sum(c['detection_misses'] for c in classes)} / {sum(c['positive_images'] for c in classes)} | {sum(c['false_positive_detections'] for c in classes)} / {sum(c['negative_images'] for c in classes)} |")
    lines += ['', '## Protocol and limits', '',
              '- Audit seeds differ from recorded training, validation, legacy test, and HSV calibration seeds. The checkpoint and CNN threshold (0.65) remain unchanged.',
              '- The script saves the full configuration and hashes before scoring audit data. HSV S/V thresholds are selected from 12 candidates using calibration only.',
              '- IoU pools intersection/union by foreground class before component filtering, then averages classes with nonzero union. Missing-target IoU is undefined when neither method predicts pixels; detection false positives are the useful measure.',
              '- Detection counts use the actual deployed component and aspect-ratio filters. A positive image has any visible ground-truth class pixel. Centroid error excludes missed detections and truth rejected by the component filter; report it with miss counts.',
              '- Latency covers one selected-class detector call including resizing, inference, thresholding, and component filtering. It excludes loading, camera acquisition, BGRA conversion, control, and telemetry. Thirty warmup calls per detector are excluded; two PyTorch CPU threads are used.',
              '- 95% paired bootstrap intervals in JSON resample synthetic scenes, retaining the three classes together. They describe this generated sample and do not establish real-world reliability.',
              '- Dim uses a 0.35 brightness multiplier; blur uses a 5x5 Gaussian with sigma 1.2; occlusion overwrites 8-16 rows and their visible-pixel labels; near-color adds orange/cyan/purple background rectangles; missing removes all trained classes.',
              '- Blur labels represent underlying visible geometry, so blurred boundaries can lower IoU. Near-color labels test semantic color boundaries absent from training; these are a deliberate distribution shift.',
              '- No real-camera or physical-robot performance is claimed. No model retraining or audit-driven parameter tuning is performed.', '',
              'Example columns: RGB image, visible-pixel ground truth, CNN masks, HSV masks. Red/blue/green overlays represent the respective catalog classes.', '']
    for row in report['results']:
        lines.append(f"![{row['case']} examples]({row['case']}_examples.png)")
    return '\n'.join(lines) + '\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/perception_audit')
    parser.add_argument('--images', type=int, default=600)
    parser.add_argument('--calibration-images', type=int, default=180)
    args = parser.parse_args()
    if min(args.images, args.calibration_images) < 3:
        parser.error('use at least three images per set')
    if args.output.exists():
        parser.error('output already exists; preserve prior audits and use a new output folder')
    torch.set_num_threads(2)
    calibration = calibrate(args.calibration_images)
    args.output.mkdir(parents=True)
    (args.output/'hsv_settings.json').write_text(json.dumps(calibration, indent=2)+'\n')
    protocol = {'version': 1, 'calibration': calibration, 'audit_seeds': AUDIT_SEEDS,
                'images_per_case': args.images, 'cnn_threshold': 0.65, 'torch_threads': 2,
                'model_sha256': sha(ROOT/'models/targets/target_catalog.pt'),
                'catalog_sha256': sha(ROOT/'models/targets/catalog.json'),
                'training_report_sha256': sha(ROOT/'reports/target_training.json'),
                'source_sha256': {p: sha(ROOT/p) for p in ['scripts/evaluate_perception.py', 'scripts/make_target_data.py', 'visionlab/perception.py', 'visionlab/targets.py']},
                'environment': {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                                'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__}}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    detectors = {'cnn': [CatalogDetector(ROOT/'models/targets', name) for name in NAMES],
                 'hsv': [HSVDetector(name, **calibration['selected']) for name in NAMES]}
    warmup = generate(30, CALIBRATION_SEED+2, 'standard')
    for method in detectors.values():
        for detector in method:
            for image, _ in warmup:
                detector.detect(image)
    rows = []
    for case in CASES:
        row, examples = run_case(generate(args.images, AUDIT_SEEDS[case], case), detectors, case)
        rows.append(row)
        cv2.imwrite(str(args.output/f'{case}_examples.png'), cv2.resize(examples, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        print(json.dumps({'case': case, 'cnn': row['cnn']['macro_iou'], 'hsv': row['hsv']['macro_iou']}), flush=True)
    report = {'protocol': protocol, 'results': rows}
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'README.md').write_text(report_markdown(report), encoding='utf-8')

if __name__ == '__main__':
    main()
