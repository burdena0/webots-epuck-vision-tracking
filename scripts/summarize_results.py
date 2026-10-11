import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / 'reports'
training = json.loads((reports / 'training.json').read_text())
comparison = json.loads((reports / 'model_comparison.json').read_text())
rows = json.loads((reports / 'benchmark_summary.json').read_text())
lines = ['# results', '', '## training', '',
         f"- {training['train_images']} training images, {training['validation_images']} validation images, 320 test images",
         f"- {training['parameters']} parameters, {len(training['history'])} epochs",
         f"- reused legacy synthetic test mask iou (diagnostic): {training['test']['mask_iou']:.4f}",
         f"- hsv mask iou on the same synthetic images: {training['hsv_test_mask_iou']:.4f}",
         '', '![learning curves](learning_curves.png)', '', '## model comparison', '',
         'both models use the exact same images and a fixed 0.65 threshold. these audit',
         'seeds were reserved before v3 training. dim images are scaled by 0.65; blurred',
         'images use a 3 x 3 gaussian kernel with sigma 0.8. the sets have different scenes.', '',
         '| case | images | original mask iou | v3 mask iou | gain (percentage points) |',
         '| --- | ---: | ---: | ---: | ---: |']
for result in comparison['results']:
    lines.append(f"| {result['case']} | {result['images']} | {result['baseline']['mask_iou']:.4f} | {result['improved']['mask_iou']:.4f} | {result['iou_gain_percentage_points']:+.2f} |")
lines.extend(['', 'v2 improved normal-light masks but lost dim-light accuracy. that audit became',
              'development data when it informed brightness augmentation. v3 fixes dim-light',
              'performance against the original on the new set, but blur is still slightly worse.',
              'the earlier reports are in `experiments`. these are synthetic masks, not real-world',
              'accuracy or statistical reliability estimates.', '', '![model comparison](model_improvement.png)',
              '', '## webots runs', '',
         'each row is one deterministic 40-second simulation, not a statistical reliability estimate.',
         'the moving target travels for five seconds, then holds still. in the occlusion scene',
         'the target is moved out of view from seconds four through seven. the obstacle scene',
         'puts a gray box between the robot and target. arrival requires a proximity stop',
         'within 8 cm of the target surface. empty arrival means the task was not completed.', '',
         '| scene | mode | arrival (sim s) | min target gap (m) | median perception (ms) | forward during occlusion (rad/s) |',
         '| --- | --- | ---: | ---: | ---: | ---: |'])
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
(reports / 'results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
