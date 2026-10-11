import json
import math
from pathlib import Path
import statistics
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
robot = Supervisor()
dt = int(robot.getBasicTimeStep())
config = json.loads(robot.getCustomData())
receiver = robot.getDevice('results')
receiver.enable(dt)
target, epuck, obstacle = (robot.getFromDef(x) for x in ('TARGET', 'EPUCK', 'OBSTACLE'))
target_translation = target.getField('translation')
packets, distance_travelled, last_position = [], 0.0, None
minimum_gap, arrival_time = float('inf'), None
while robot.step(dt) != -1:
    t, case = robot.getTime(), config['case']
    if case == 'moving':
        target_translation.setSFVec3f([0.42, 0.10 * math.sin(min(t, 5.0) * 0.5), 0.10])
    elif case == 'occlusion':
        target_translation.setSFVec3f([0.42, 0.0, -1.0 if 4 <= t < 7 else 0.10])
    elif case == 'obstacle':
        obstacle.getField('translation').setSFVec3f([0.18, 0, 0.05])
    position = epuck.getPosition()
    if last_position is not None:
        distance_travelled += math.dist(position[:2], last_position[:2])
    last_position = position
    center = target.getPosition()
    # robot circle against the target box, only when its in the arena
    gap = math.hypot(max(abs(position[0] - center[0]) - 0.06, 0), max(abs(position[1] - center[1]) - 0.06, 0)) - 0.037
    if center[2] > 0:
        minimum_gap = min(minimum_gap, gap)
    while receiver.getQueueLength():
        packet = json.loads(receiver.getString())
        packets.append(packet)
        if packet['state'] == 'blocked' and gap < 0.08 and arrival_time is None and case != 'obstacle':
            arrival_time = t
        receiver.nextPacket()
    if t >= config['duration']:
        break
errors = [abs(p['reference_x'] - 0.5) for p in packets if p['reference_x'] is not None]
agreement = [abs(p['x'] - p['reference_x']) * 320 for p in packets if p['x'] is not None and p['reference_x'] is not None]
latencies = [p['inference_ms'] for p in packets]
loss_motion = [abs(p['left'] + p['right']) / 2 for p in packets if case == 'occlusion' and 4.15 <= p['time'] < 6.9]
row = {'case': case, 'backend': config['backend'], 'samples': len(packets), 'duration_sim_seconds': robot.getTime(),
       'arrival_sim_seconds': arrival_time, 'minimum_target_gap_m': round(minimum_gap, 4) if math.isfinite(minimum_gap) else None, 'travelled_m': round(distance_travelled, 4),
       'mean_visible_alignment_error': round(statistics.mean(errors), 4) if errors else None,
       'mean_centroid_difference_vs_hsv_pixels': round(statistics.mean(agreement), 3) if agreement else None,
       'inference_median_ms': round(statistics.median(latencies), 3) if latencies else None,
       'inference_p95_ms': round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
       'max_forward_speed_during_occlusion_rad_s': round(max(loss_motion), 5) if loss_motion else None,
       'blocked_samples': sum(p['state'] == 'blocked' for p in packets), 'final_position': position,
       'max_forward_speed_rad_s': round(max([abs(p['left'] + p['right']) / 2 for p in packets], default=0), 5),
       'wrong_target_packets': sum(p.get('target') != config.get('target', 'red_box') for p in packets),
       'states': sorted(set(p['state'] for p in packets))}
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
(reports / f"{config['case']}-{config['backend']}.json").write_text(json.dumps(row, indent=2) + '\n', newline='\n')
print('benchmark ' + json.dumps(row), flush=True)
robot.simulationQuit(0 if packets else 2)
