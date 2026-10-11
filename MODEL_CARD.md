# target net

this card covers the red-only model. the selectable multi-class model has its own
[data guide](docs/target_data.md) and [measurements](reports/target_results.md).

## job

take an image, predict a red target mask, then use its centroid for steering. the
network doesnt output wheel speeds or learn a driving policy. the controller
handles steering, speed limits and proximity stops separately.

## model and training

- three convolution layers: 3 -> 8 -> 12 -> 1 channels
- relu after the first two layers, sigmoid on the output
- 1,113 trainable parameters, 96 x 72 rgb input, cpu inference
- adam, starting learning rate 0.003, cosine decay to 0.0003
- weighted binary cross entropy with positive weight 3, plus 0.25 soft dice loss
- batch size 64, 36 epochs, deterministic seed 20261011
- per-image brightness multiplier sampled from 0.5 to 1.25 each training pass
- checkpoint selected by mean standard and dim validation iou, not audit scores
- fixed 0.65 pixel threshold for both models

`models/target_net.pt` is the active v3 model. `baseline_target_net.pt` keeps the
original 1,600-image, 12-epoch bce-only model. `improved_v2_target_net.pt` keeps the
intermediate run that showed a dim-light regression. it is not used for driving.

## data and split rules

4,800 training images and 320 validation images, generated with seeds 20261011
and 20261012. dim validation is the same 320 scenes scaled by 0.65, not another
independent split. the generator draws boxes and records their visible pixels as
labels. it varies position, size, intensity, shadows, clutter, partial occlusion
and pixel noise. some scenes have no target. augmentation changes pixels only.

these are procedural images, not photos or recorded webots frames. there are no
pretrained weights or hsv-generated labels. simulator measurements are separate.

the legacy 320-image test (seed 20261013) has been viewed across experiments. its
v3 score of 0.9550 is a diagnostic, not a new untouched result. v2's audit informed
the decision to add brightness augmentation, so that set also became development
data. its reports are retained in `reports/experiments`.

the final v3 audit seeds were reserved before training and scored after validation
checkpoint selection: 202610220 (1,000 normal scenes), 202610221 (320 dim scenes)
and 202610222 (320 blurred scenes). baseline and v3 see identical images within
each case. dim pixels are multiplied by 0.65. blur uses a 3 x 3 gaussian kernel,
sigma 0.8. all three cases have different scenes. no post-audit tuning was done.

## comparison

| synthetic audit | original mask iou | v3 mask iou | change |
| --- | ---: | ---: | ---: |
| normal, 1,000 images | 0.9374 | 0.9591 | +2.17 percentage points |
| dim, 320 images | 0.8755 | 0.9438 | +6.83 percentage points |
| blurred, 320 images | 0.9155 | 0.9121 | -0.34 percentage points |

normal-scene negative-image false positives dropped from 6 to 2, with no
empty-mask misses for either model. dim centroid mean absolute error dropped from
1.201 to 0.496 pixels at 96-pixel width. blur had one empty-mask miss for v3 and
none for the original. centroid error excludes negative and empty-mask images.

iou is pooled pixel intersection / union before component filtering. the negative
false-positive rule is a raw predicted area of at least 0.003 of image pixels.
an empty-mask miss means a positive image had no predicted pixels; it does not
measure every possible detection failure. see `reports/model_comparison.json` for
all counts. synthetic performance doesnt prove webots or hardware accuracy.

v2 reached higher normal-light accuracy on its earlier audit but regressed in dim
light. v3 prioritizes the balanced validation score. it improves normal and dim
results against the original, with a small blur tradeoff. dont call it better in
every condition. local closed-loop measurements are in `reports/results.md`.

## failure cases

multiple red objects can make the largest component win over the intended target.
broad red backgrounds can look like a target. the aspect-ratio filter can reject
a heavily occluded or very wide target. stopping can prevent contact without
completing the task; that does not count as an arrival. the obstacle scene can hide
the target completely. there is no route planner or physical robot verification.

## reproduce

run `scripts/train_model.py --run-name improved-v3`, then `scripts/compare_models.py`.
the default settings match the published run. optimizer checkpoints and settings
are saved in `runs/improved-v3`; each new run replaces the active model and report.

the original training settings were `--epochs 12 --train-images 1600 --schedule none
--dice-weight 0 --augmentation none --selection standard`. preserve the active v3
model before running that command, since training replaces it. torch versions are
recorded in the reports. inference timing includes preprocessing and postprocessing
and depends on the machine. if an audit informs another change, reserve new seeds.
