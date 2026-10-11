import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / 'reports'
training = json.loads((reports / 'target_training.json').read_text())
rows = json.loads((reports / 'target_benchmark_summary.json').read_text())
assert len(rows) == 6
for row in rows:
    assert row['samples'] == 1249 and row['wrong_target_packets'] == 0
    if row['case'].startswith('missing_'):
        assert row['arrival_sim_seconds'] is None
        assert row['max_forward_speed_rad_s'] == 0
    else:
        assert row['arrival_sim_seconds'] is not None
        assert row['minimum_target_gap_m'] > 0
lines = ['# named target results', '',
         'the catalog model has 1,460 parameters and was trained from random weights on',
         '3,000 labeled synthetic images for 24 epochs. 320 validation images selected',
         'the checkpoint. the untouched 400-image test split was scored afterward.',
         'this dataset differs from the red-only model audit; dont compare the scores',
         'as an improvement on the same task.', '',
         '| target class | test mask iou | empty-mask misses | negative-image false positives |',
         '| --- | ---: | ---: | ---: |']
for name, row in training['test']['classes'].items():
    lines.append(f"| {name} | {row['mask_iou']:.4f} | {row['empty_mask_misses']} | {row['negative_image_false_positives']} |")
lines.extend(['', f"macro foreground mask iou: {training['test']['macro_iou']:.4f}. pooled iou is calculated",
              'separately for each class, then averaged. masks use a fixed 0.65 threshold before',
              'component filtering. empty-mask misses are positive images with no predicted',
              'pixels. negative false positives require at least 0.003 of image area.', '',
              '![training and test](target_learning_curves.png)', '', '## selected-target simulation', '',
              'each row is one deterministic 40-second run with three fixed box positions.',
              'the selected color varies; the other two stay visible as distractors. missing',
              'cases hide only the selected box. arrival requires a proximity stop within',
              '8 cm of the selected surface. missing cases should not complete an arrival.', '',
              '| scene | arrival (sim s) | approximate min selected-target gap (m) | max forward command (rad/s) | median perception (ms) |',
              '| --- | ---: | ---: | ---: | ---: |'])
for row in rows:
    arrival = 'none' if row['arrival_sim_seconds'] is None else f"{row['arrival_sim_seconds']:.3f}"
    gap = '-' if row['minimum_target_gap_m'] is None else f"{row['minimum_target_gap_m']:.4f}"
    lines.append(f"| {row['case']} | {arrival} | {gap} | {row['max_forward_speed_rad_s']:.5f} | {row['inference_median_ms']:.3f} |")
lines.extend(['', 'all six runs kept the requested target name in every telemetry packet. the',
              'selection unit test also checks which box produces the predicted centroid.',
              'the red-only hsv reference fields in raw telemetry are not selected-target',
              'accuracy measurements for blue or green. they are not used for steering.', '',
              '![simulator camera and selected mask](target_selection_examples.png)', '',
              'there were no negative approximate selected-target gaps in these three approach',
              'runs. the blue and green gaps were small (2.1 and 3.9 mm). raw proximity readings',
              'vary with the scene; this is not a verified physical stopping margin. geometry',
              'approximates the robot body and does not cover every wheel or protrusion.',
              'missing-target travelled distance was about 0.1 mm of simulator settling while',
              'forward commands stayed zero. this does not measure general collision avoidance.',
              'same-class instances, more lighting conditions and real data need separate tests.', ''])
(reports / 'target_results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
history = training['history']
axes[0].plot([row['epoch'] for row in history], [row['validation']['macro_iou'] for row in history], color='#146c74')
axes[0].set(title='validation selects the model', xlabel='epoch', ylabel='macro foreground mask iou', ylim=(0, 1))
values = [training['test']['classes'][name]['mask_iou'] * 100 for name in training['classes']]
bars = axes[1].bar(training['classes'], values, color=['#bf4141', '#477dbb', '#408a61'])
axes[1].bar_label(bars, fmt='%.2f')
axes[1].set(title='400 held-out synthetic scenes', ylabel='class mask iou (%)', ylim=(0, 106))
for ax in axes:
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(reports / 'target_learning_curves.png', dpi=160)
plt.close(fig)
fig, axes = plt.subplots(3, 1, figsize=(9, 8))
for ax, name in zip(axes, training['classes']):
    path = reports / f'select_{name}-catalog-camera.png'
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(f'selected: {name} | camera (left), selected mask (right)')
    ax.axis('off')
fig.tight_layout()
fig.savefig(reports / 'target_selection_examples.png', dpi=140)
plt.close(fig)
print('6 target-selection runs checked; class report and figures saved')
