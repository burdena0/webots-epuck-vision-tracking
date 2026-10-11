import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def world(case, backend, duration):
    config = json.dumps({'case': case, 'backend': backend, 'duration': duration})
    y = 0.13 if case == 'offset' else 0.0
    return f'''#VRML_SIM R2025a utf8
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/gctronic/e-puck/protos/E-puck.proto"
WorldInfo {{ basicTimeStep 32 randomSeed 11 }}
Viewpoint {{ position 1.2 0.8 1.4 orientation -0.4 0.2 0.9 3.2 }}
TexturedBackground {{ }}
TexturedBackgroundLight {{ }}
RectangleArena {{ floorSize 2 2 }}
DEF TARGET Solid {{
  translation 0.42 {y} 0.1
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.8 0.02 0.01 roughness 1 metalness 0 }} geometry Box {{ size 0.12 0.12 0.2 }} }} ]
  name "target" boundingObject Box {{ size 0.12 0.12 0.2 }}
}}
DEF OBSTACLE Solid {{
  translation -0.8 -0.8 0.05
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.25 0.25 0.25 roughness 1 metalness 0 }} geometry Box {{ size 0.04 0.18 0.1 }} }} ]
  name "obstacle" boundingObject Box {{ size 0.04 0.18 0.1 }}
}}
DEF EPUCK E-puck {{
  translation 0 0 0 controller "camera_view"
  controllerArgs [ "--backend" "{backend}" "--no-preview" ]
  customData "{case}-{backend}" window ""
  camera_width 320 camera_height 240
  turretSlot [ Emitter {{ name "telemetry" channel 42 }} ]
}}
Robot {{
  supervisor TRUE controller "benchmark_supervisor" window ""
  customData {json.dumps(config)}
  children [ Receiver {{ name "results" channel 42 }} ]
}}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webots', required=True)
    parser.add_argument('--cases', nargs='+', choices=['center', 'offset', 'moving', 'occlusion', 'obstacle'], default=['center', 'offset', 'moving', 'occlusion', 'obstacle'])
    parser.add_argument('--backends', nargs='+', choices=['original', 'hsv', 'neural'], default=['original', 'hsv', 'neural'])
    parser.add_argument('--duration', type=float, default=40)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('duration needs to be positive')
    env = dict(os.environ, WEBOTS_PYTHON_COMMAND=sys.executable, PYTHONIOENCODING='utf-8')
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    rows = []
    for case in args.cases:
        for backend in args.backends:
            path = ROOT / 'worlds' / f'benchmark_{case}_{backend}.wbt'
            path.write_text(world(case, backend, args.duration), encoding='utf-8')
            print(f'running {case}/{backend}', flush=True)
            output = reports / f'{case}-{backend}.json'
            output.unlink(missing_ok=True)
            command = [args.webots, '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr', str(path)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=150)
            (reports / f'{case}-{backend}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            if result.returncode or not output.is_file():
                raise RuntimeError(f'webots failed for {case}/{backend}: {result.stdout[-2500:]} {result.stderr[-2500:]}')
            rows.append(json.loads(output.read_text()))
            print(json.dumps(rows[-1]), flush=True)
    (reports / 'benchmark_summary.json').write_text(json.dumps(rows, indent=2) + '\n')

if __name__ == '__main__':
    main()
