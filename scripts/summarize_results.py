import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / 'reports'
training = json.loads((reports / 'training.json').read_text())
rows = json.loads((reports / 'benchmark_summary.json').read_text())
lines = ['# results', '', '## training', '',
         f"- {training['train_images']} training images, {training['validation_images']} validation images, 320 test images",
         f"- {training['parameters']} parameters, {len(training['history'])} epochs",
         f"- held-out synthetic mask iou: {training['test']['mask_iou']:.4f}",
         f"- hsv mask iou on the same synthetic images: {training['hsv_test_mask_iou']:.4f}",
         '', '![learning curves](learning_curves.png)', '', '## webots runs', '',
         'each row is one deterministic 40-second simulation, not a statistical reliability estimate.',
         'the moving target travels for five seconds, then holds still. in the occlusion scene',
         'the target is moved out of view from seconds four through seven. the obstacle scene',
         'puts a gray box between the robot and target. arrival requires a proximity stop',
         'within 8 cm of the target surface. empty arrival means the task was not completed.', '',
         '| scene | mode | arrival (sim s) | min target gap (m) | median perception (ms) | forward during occlusion (rad/s) |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
for row in rows:
    arrive = 'incomplete' if row['arrival_sim_seconds'] is None else f"{row['arrival_sim_seconds']:.3f}"
    speed = '-' if row['max_forward_speed_during_occlusion_rad_s'] is None else f"{row['max_forward_speed_during_occlusion_rad_s']:.5f}"
    lines.append(f"| {row['case']} | {row['backend']} | {arrive} | {row['minimum_target_gap_m']:.4f} | {row['inference_median_ms']:.3f} | {speed} |")
lines.extend(['', 'a negative gap means the approximate circle/box geometry overlaps. it is a',
              'failure signal, not a successful safety result. the gap is an approximation of',
              'the robot body and does not measure every wheel or protrusion.',
              '', 'arrival improvements combine changes to speed and control. they are not evidence',
              'that the neural model alone makes the robot faster. hsv reference agreement is',
              'a consistency check, not ground-truth segmentation accuracy on webots frames.',
              '', 'the obstacle scene can fully hide the target. staying stopped is useful but',
              'doesnt count as completing the route. this controller does not do obstacle avoidance.', ''])
(reports / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
