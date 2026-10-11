import argparse
import json
from pathlib import Path
import sys
import time
from controller import Robot
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visionlab.control import Policy, OriginalPolicy
from visionlab.perception import HSVDetector, OriginalDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['hsv', 'neural', 'original', 'catalog'], default='neural')
    parser.add_argument('--target', default='red_box')
    parser.add_argument('--target-bundle', type=Path, default=ROOT / 'models/targets')
    parser.add_argument('--hsv-settings', type=Path)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    if args.backend not in ('catalog', 'hsv') and args.target != 'red_box':
        parser.error('named targets need --backend catalog or hsv')
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    motors = [robot.getDevice(name) for name in ('left wheel motor', 'right wheel motor')]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0)
    try:
        settings = {}
        if args.hsv_settings:
            path = args.hsv_settings if args.hsv_settings.is_absolute() else ROOT / args.hsv_settings
            settings = json.loads(path.read_text())['selected']
        if args.backend == 'catalog':
            from visionlab.targets import CatalogDetector
            bundle = args.target_bundle if args.target_bundle.is_absolute() else ROOT / args.target_bundle
            detector = CatalogDetector(bundle, args.target)
        elif args.backend == 'neural':
            from visionlab.neural import NeuralDetector
            detector = NeuralDetector(ROOT / 'models/target_net.pt')
        else:
            detector = OriginalDetector() if args.backend == 'original' else HSVDetector(args.target, **settings)
        policy = OriginalPolicy() if args.backend == 'original' else Policy()
        reference = HSVDetector(args.target, **settings)
        camera = robot.getDevice('camera')
        camera.enable(timestep)
        sensors = [robot.getDevice(f'ps{i}') for i in range(8)]
        for sensor in sensors:
            sensor.enable(timestep)
        emitter = robot.getDevice('telemetry')
        print(f'camera ready. backend={args.backend} target={args.target}. press q to stop', flush=True)
        step_index = 0
        while robot.step(timestep) != -1:
            image = camera.getImage()
            if image is None:
                for motor in motors:
                    motor.setVelocity(0)
                continue
            bgra = np.frombuffer(image, np.uint8).reshape(camera.getHeight(), camera.getWidth(), 4)
            frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            started = time.perf_counter()
            detection, mask = detector.detect(frame)
            inference_ms = (time.perf_counter() - started) * 1000
            command = policy.step(detection, [sensor.getValue() for sensor in sensors], timestep / 1000)
            for motor, speed in zip(motors, (command.left, command.right)):
                motor.setVelocity(speed)
            # this reference is just for the report. the policy never gets it
            ref, _ = reference.detect(frame)
            packet = {'time': robot.getTime(), 'state': command.state, 'left': command.left, 'right': command.right,
                      'x': None if detection is None else detection.x, 'reference_x': None if ref is None else ref.x,
                      'front': max(sensors[i].getValue() for i in (0, 1, 6, 7)), 'inference_ms': inference_ms,
                      'target': args.target}
            emitter.send(json.dumps(packet).encode())
            if step_index % 30 == 0:
                print(f"{command.state}: front={packet['front']:.1f} inference={inference_ms:.2f}ms", flush=True)
            if step_index == 20 and robot.getCustomData():
                capture = ROOT / 'reports' / f'{robot.getCustomData()}-camera.png'
                capture.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(capture), np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
            step_index += 1
            if not args.no_preview:
                preview = frame.copy()
                if detection:
                    x, y, w, h = detection.box
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview, args.target + ': ' + command.state, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow('camera - q to stop', preview)
                cv2.imshow('target mask', mask)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    finally:
        for motor in motors:
            motor.setVelocity(0)
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
