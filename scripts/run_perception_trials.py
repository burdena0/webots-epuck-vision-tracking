"""Paired selected-target trials; keep controller and world constant across detectors."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_benchmarks import world
from visionlab.evaluation import wilson

COLORS = {'red_box': '0.8 0.02 0.01', 'blue_box': '0.02 0.01 0.8', 'green_box': '0.01 0.8 0.02'}
CASES = ('standard', 'dim', 'occlusion', 'near_color', 'missing', 'same_color')
SEEDS = (910110001, 910110002, 910110003)

def object_text(name, position, color):
    return f'''\nDEF {name} Solid {{
  translation {position[0]} {position[1]} 0.1
  children [ Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }} geometry Box {{ size 0.12 0.12 0.2 }} }} ]
  name "{name}" boundingObject Box {{ size 0.12 0.12 0.2 }}
}}\n'''

def trial_world(case, target, seed, backend, duration, output, hsv_settings):
    rng = random.Random(seed)
    x, y = rng.uniform(0.40, 0.47), rng.uniform(-0.075, 0.075)
    heading = rng.uniform(-0.06, 0.06)
    trial_id = f'audit_{case}_{target}_{seed}_{backend}'
    config = {'case': case, 'backend': backend, 'target': target, 'seed': seed, 'duration': duration,
              'trial_id': trial_id, 'output': str(output.resolve()), 'object_defs': ['TARGET', 'DISTRACTOR_A', 'DISTRACTOR_B'],
              'initial_target': [x,y], 'initial_robot_heading': heading}
    text = world('center', 'neural', duration)
    original = json.dumps(json.dumps({'case': 'center', 'backend': 'neural', 'duration': duration}))
    text = text.replace('customData '+original, 'customData '+json.dumps(json.dumps(config)))
    text = text.replace('randomSeed 11', f'randomSeed {seed}')
    text = text.replace('translation 0.42 0.0 0.1', f'translation {x} {y} {"-1.0" if case == "missing" else "0.1"}')
    text = text.replace('baseColor 0.8 0.02 0.01', 'baseColor '+COLORS[target])
    text = text.replace('translation 0 0 0 controller "camera_view"', f'translation 0 0 0 rotation 0 0 1 {heading} controller "camera_view"')
    args = ' '.join(json.dumps(a) for a in ['--backend', backend, '--target', target, '--no-preview', '--hsv-settings', str(hsv_settings.resolve())])
    text = text.replace('"--backend" "neural" "--no-preview"', args)
    text = text.replace('customData "center-neural"', 'customData '+json.dumps(trial_id))
    text = text.replace('controller "benchmark_supervisor"', 'controller "audit_supervisor"')
    if case == 'dim':
        # Explicit low ambient + direct light, rather than manipulating camera pixels.
        text = text.replace('TexturedBackgroundLight { }', 'DirectionalLight { direction -0.6 -1 -1 ambientIntensity 0.05 intensity 0.12 castShadows TRUE }')
        text = text.replace('TexturedBackground { }', 'TexturedBackground { luminosity 0.12 }')
    others = [color for name, color in COLORS.items() if name != target]
    if case == 'near_color':
        others = ['0.7 0.28 0.05', '0.4 0.05 0.7'] if target != 'blue_box' else ['0.05 0.55 0.7', '0.4 0.05 0.7']
    if case == 'same_color':
        # An instance-identity challenge outside the color-class model's capability.
        others[0] = COLORS[target]
    text += object_text('DISTRACTOR_A', (0.34 if case == 'same_color' else x, -0.21), others[0])
    text += object_text('DISTRACTOR_B', (x, 0.21), others[1])
    return text, config

def summary_markdown(rows):
    lines = ['# Paired closed-loop Webots evaluation', '',
             'Each case and detector uses the same three scene seeds for each of red, blue, and green: nine trials per cell. The same steering and stopping policy is used for both detectors.', '',
             '| Case | Detector | Successes / trials | 95% Wilson interval | Object collision trials | Median perception ms |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    import statistics
    for case in CASES:
        for backend in ('catalog', 'hsv'):
            subset = [row for row in rows if row['case']==case and row['backend']==backend]
            if not subset:
                continue
            successes = sum(row['success'] for row in subset)
            low, high = wilson(successes, len(subset))
            latency = statistics.median(row['inference_median_ms'] for row in subset)
            lines.append(f'| {case} | {backend} | {successes} / {len(subset)} | {low*100:.1f}-{high*100:.1f}% | {sum(row["collision"] for row in subset)} | {latency:.3f} |')
    lines += ['', '## Scoring and limits', '',
              '- Approach success requires a continuous 0.5-second stationary proximity stop with the selected target visible, its approximate surface gap between 0 and 8 cm, and the target at least 1 mm nearer than either distractor. No observed target/distractor contact or approximate overlap beyond 1 mm is allowed throughout the run.',
              '- Missing-target success requires zero positive forward commands for the entire trial, no arrival, and no object collision. Turning in place is permitted.',
              '- Contact checks match global contact locations between e-puck descendants and each target/distractor to within 1 micrometer. Webots contact node IDs identify the queried body, not its partner, and are not used to infer collision partners. Ground/floor-only contacts are excluded. The geometric circle/box approximation flags additional overlap; it does not certify physical clearance.',
              '- Occlusion moves the selected box out of view during seconds 3-6, then restores it. Dim scenes use low ambient/direct lighting. Near-color scenes add orange/cyan/purple distractors. Same-color scenes introduce a nearer instance of the selected color and test instance ambiguity; neither detector is trained to identify a specific same-color object.',
              '- Identical seeds define identical starting positions and heading for both detectors. Trial order alternates by seed. Models, controller parameters, and HSV settings stay frozen.',
              '- Latency covers selected detector preprocessing/inference/postprocessing, excluding camera acquisition, control, and telemetry; first 30 samples are omitted. JSON includes median and p95.',
              '- These are nine fixed simulation configurations per cell, not nine independent real-world trials. Wilson intervals illustrate small sample uncertainty and do not establish deployment reliability.',
              '- Raw per-step telemetry and per-trial logs are saved for inspection. Incomplete telemetry and controller faults fail the runner instead of counting as successes.', '']
    return '\n'.join(lines)+'\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webots', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/closed_loop_verified')
    parser.add_argument('--hsv-settings', type=Path, default=ROOT/'reports/perception_audit/hsv_settings.json')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--targets', nargs='+', choices=list(COLORS), default=list(COLORS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--duration', type=float, default=25)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.duration < 10 or not args.hsv_settings.is_file():
        parser.error('need duration >= 10 and a saved calibration file')
    if args.output.exists() and not args.resume:
        parser.error('output already exists; use --resume for the unchanged protocol')
    args.output.mkdir(parents=True, exist_ok=True)
    hashed = ['scripts/run_perception_trials.py', 'controllers/audit_supervisor/audit_supervisor.py',
              'controllers/camera_view/camera_view.py', 'visionlab/control.py', 'visionlab/evaluation.py',
              'visionlab/perception.py', 'visionlab/targets.py', 'models/targets/target_catalog.pt']
    protocol = {'cases': args.cases, 'targets': args.targets, 'seeds': args.seeds, 'duration': args.duration,
                'hsv_settings_sha256': hashlib.sha256(args.hsv_settings.read_bytes()).hexdigest(),
                'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in hashed},
                'python': sys.version, 'webots_path': str(Path(args.webots).resolve())}
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('cannot resume with changed code, model, calibration, or trial plan')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    env = dict(os.environ, WEBOTS_PYTHON_COMMAND=sys.executable, PYTHONIOENCODING='utf-8')
    rows = []
    for case, target, seed in itertools.product(args.cases, args.targets, args.seeds):
        for backend in (('catalog','hsv') if seed%2 else ('hsv','catalog')):
            text, config = trial_world(case, target, seed, backend, args.duration, args.output, args.hsv_settings)
            trial = config['trial_id']
            result_path = args.output/f'{trial}.json'
            if result_path.exists() and args.resume:
                row = json.loads(result_path.read_text())
                if any(row[k] != config[k] for k in config):
                    raise ValueError('saved trial does not match protocol')
            else:
                world_path = ROOT/'worlds'/f'benchmark_{trial}.wbt'
                world_path.write_text(text, encoding='utf-8')
                print(f'running {trial}', flush=True)
                started = time.perf_counter()
                run = subprocess.run([args.webots, '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr', str(world_path)],
                                     env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
                (args.output/f'{trial}.log').write_text(run.stdout+run.stderr, encoding='utf-8')
                if run.returncode or not result_path.exists():
                    raise RuntimeError(f'{trial} failed: {run.stdout[-1800:]} {run.stderr[-1800:]}')
                row = json.loads(result_path.read_text())
                row['wall_seconds'] = time.perf_counter()-started
                row['world_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                result_path.write_text(json.dumps(row, indent=2)+'\n')
            rows.append(row)
            (args.output/'summary.json').write_text(json.dumps({'protocol':protocol, 'trials':rows}, indent=2)+'\n')
            (args.output/'README.md').write_text(summary_markdown(rows), encoding='utf-8')
            print(json.dumps({'trial':trial, 'success':row['success'], 'collision':row['collision'], 'arrival':row['arrival_sim_seconds']}), flush=True)

if __name__ == '__main__':
    main()
