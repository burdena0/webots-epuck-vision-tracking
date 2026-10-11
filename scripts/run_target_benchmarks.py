import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_benchmarks import world

COLORS = {'red_box': '0.8 0.02 0.01', 'blue_box': '0.02 0.01 0.8', 'green_box': '0.01 0.8 0.02'}
POSITIONS = {'red_box': -0.13, 'blue_box': 0, 'green_box': 0.13}

def selection_world(target, missing=False, duration=40):
    case = ('missing_' if missing else 'select_') + target
    text = world('center', 'neural', duration)
    config = json.dumps({'case': case, 'backend': 'catalog', 'duration': duration, 'target': target})
    old_config = json.dumps(json.dumps({'case': 'center', 'backend': 'neural', 'duration': duration}))
    text = text.replace('customData ' + old_config, 'customData ' + json.dumps(config))
    text = text.replace('translation 0.42 0.0 0.1', f'translation 0.42 {POSITIONS[target]} {"-1.0" if missing else "0.1"}')
    text = text.replace('baseColor 0.8 0.02 0.01', 'baseColor ' + COLORS[target])
    text = text.replace('"--backend" "neural" "--no-preview"', f'"--backend" "catalog" "--target" "{target}" "--no-preview"')
    text = text.replace('customData "center-neural"', f'customData "{case}-catalog"')
    for name in COLORS:
        if name == target:
            continue
        text += f'''\nSolid {{
  translation 0.42 {POSITIONS[name]} 0.1
  children [ Shape {{ appearance PBRAppearance {{ baseColor {COLORS[name]} roughness 1 metalness 0 }} geometry Box {{ size 0.12 0.12 0.2 }} }} ]
  name "distractor_{name}" boundingObject Box {{ size 0.12 0.12 0.2 }}
}}
'''
    return text

def demo_world():
    text = selection_world('red_box').replace('"--no-preview"', '')
    text = text.replace('customData "select_red_box-catalog"', 'customData ""')
    # the demo runs freely and doesnt replace saved benchmark captures
    return re.sub(r'\nRobot \{\n  supervisor TRUE.*?\n\}\n', '\n', text, flags=re.S)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webots', required=True)
    args = parser.parse_args()
    env = dict(os.environ, WEBOTS_PYTHON_COMMAND=sys.executable, PYTHONIOENCODING='utf-8')
    reports = ROOT / 'reports'
    rows = []
    for target in COLORS:
        for missing in (False, True):
            case = ('missing_' if missing else 'select_') + target
            path = ROOT / 'worlds' / f'benchmark_{case}_catalog.wbt'
            path.write_text(selection_world(target, missing), encoding='utf-8', newline='\n')
            output = reports / f'{case}-catalog.json'
            output.unlink(missing_ok=True)
            print('running ' + case, flush=True)
            result = subprocess.run([args.webots, '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr', str(path)],
                                    env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=150)
            (reports / f'{case}-catalog.log').write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
            if result.returncode or not output.exists():
                raise RuntimeError(f'{case} failed: {result.stdout[-2000:]} {result.stderr[-2000:]}')
            row = json.loads(output.read_text())
            rows.append(row)
            print(json.dumps(row), flush=True)
    (reports / 'target_benchmark_summary.json').write_text(json.dumps(rows, indent=2) + '\n', newline='\n')
    (ROOT / 'worlds/target_selection_lab.wbt').write_text(demo_world(), encoding='utf-8', newline='\n')

if __name__ == '__main__':
    main()
