import gzip
import json
import math
from pathlib import Path
import sys

from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visionlab.evaluation import circle_box_gap, score_trial

robot = Supervisor()
dt = int(robot.getBasicTimeStep())
config = json.loads(robot.getCustomData())
receiver = robot.getDevice('results')
receiver.enable(dt)
target, epuck = robot.getFromDef('TARGET'), robot.getFromDef('EPUCK')
translation = target.getField('translation')
initial_target = translation.getSFVec3f()
objects = {name: node for name in config['object_defs'] if (node := robot.getFromDef(name)) is not None}
samples = []
epuck.enableContactPointsTracking(dt, True)
for node in objects.values():
    node.enableContactPointsTracking(dt)
while robot.step(dt) != -1:
    t = robot.getTime()
    if config['case'] == 'occlusion':
        translation.setSFVec3f([initial_target[0], initial_target[1], -1.0 if 3 <= t < 6 else initial_target[2]])
    position, center = epuck.getPosition(), target.getPosition()
    # node_id identifies the queried body's descendant, NOT the collision partner.
    # A shared global contact location links the robot and object's contact lists.
    robot_contacts = epuck.getContactPoints(True)
    contacts = sorted(name for name, node in objects.items()
                      if any(math.dist(a.point,b.point) <= 1e-6
                             for a in robot_contacts for b in node.getContactPoints()))
    gap = circle_box_gap(position, center)
    nearest = gap + 0.001 < min(circle_box_gap(position,node.getPosition())
                               for name,node in objects.items() if name != 'TARGET')
    while receiver.getQueueLength():
        packet = json.loads(receiver.getString())
        packet.update({'target_gap_m': gap, 'target_visible': center[2] > 0, 'target_nearest': nearest,
                       'contacts': contacts, 'robot_position': position, 'selected_target_position': center})
        if packet.get('target') != config['target']:
            raise RuntimeError('controller target does not match scored target')
        samples.append(packet)
        receiver.nextPacket()
    if t >= config['duration']:
        break
result = {**config, **score_trial(samples, config['case'], dt/1000, config['duration'])}
output = Path(config['output'])
output.mkdir(parents=True, exist_ok=True)
with gzip.open(output/f"{config['trial_id']}.telemetry.jsonl.gz", 'wt', encoding='utf-8') as stream:
    for row in samples:
        stream.write(json.dumps(row)+'\n')
(output/f"{config['trial_id']}.json").write_text(json.dumps(result, indent=2)+'\n')
print('audit result '+json.dumps(result), flush=True)
robot.simulationQuit(0)
