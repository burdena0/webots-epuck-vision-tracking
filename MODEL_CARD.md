# Target net

This card covers the red-only model. The selectable multi-class model has its own
[data guide](docs/target_data.md) and [measurements](reports/target_results.md).

## Job

Take an image, predict a red target mask, then use its centroid for steering. The
network doesn't output wheel speeds or learn a driving policy. The controller
handles steering, speed limits and proximity stops separately.

## Model and training

- Three convolution layers: 3 -> 8 -> 12 -> 1 channels
- ReLU after the first two layers, sigmoid on the output
- 1,113 trainable parameters, 96 x 72 RGB input, CPU inference
- Adam, starting learning rate 0.003, cosine decay to 0.0003
- Weighted binary cross entropy with positive weight 3, plus 0.25 soft dice loss
- Batch size 64, 36 epochs, deterministic seed 20261011
- Per-image brightness multiplier sampled from 0.5 to 1.25 each training pass
- Checkpoint selected by mean standard and dim validation IoU, not audit scores
- Fixed 0.65 pixel threshold for both models

`models/target_net.pt` is the active v3 model. `baseline_target_net.pt` keeps the
original 1,600-image, 12-epoch BCE-only model. `improved_v2_target_net.pt` keeps the
intermediate run that showed a dim-light regression. It is not used for driving.

## Data and split rules

4,800 training images and 320 validation images, generated with seeds 20261011
and 20261012. Dim validation is the same 320 scenes scaled by 0.65, not another
independent split. The generator draws boxes and records their visible pixels as
labels. It varies position, size, intensity, shadows, clutter, partial occlusion
and pixel noise. Some scenes have no target. Augmentation changes pixels only.

These are procedural images, not photos or recorded Webots frames. There are no
pretrained weights or HSV-generated labels. Simulator measurements are separate.

The legacy 320-image test (seed 20261013) has been viewed across experiments. Its
v3 score of 0.9550 is a diagnostic, not a new untouched result. V2's audit informed
the decision to add brightness augmentation, so that set also became development
data. Its reports are retained in `reports/experiments`.

The final v3 audit seeds were reserved before training and scored after validation
checkpoint selection: 202610220 (1,000 normal scenes), 202610221 (320 dim scenes)
and 202610222 (320 blurred scenes). Baseline and v3 see identical images within
each case. Dim pixels are multiplied by 0.65. Blur uses a 3 x 3 Gaussian kernel,
sigma 0.8. All three cases have different scenes. No post-audit tuning was done.

## Comparison

| synthetic audit | original mask IoU | v3 mask IoU | change |
| --- | ---: | ---: | ---: |
| normal, 1,000 images | 0.9374 | 0.9591 | +2.17 percentage points |
| dim, 320 images | 0.8755 | 0.9438 | +6.83 percentage points |
| blurred, 320 images | 0.9155 | 0.9121 | -0.34 percentage points |

Normal-scene negative-image false positives dropped from 6 to 2, with no
empty-mask misses for either model. Dim centroid mean absolute error dropped from
1.201 to 0.496 pixels at 96-pixel width. Blur had one empty-mask miss for v3 and
none for the original. Centroid error excludes negative and empty-mask images.

IoU is pooled pixel intersection / union before component filtering. The negative
false-positive rule is a raw predicted area of at least 0.003 of image pixels.
An empty-mask miss means a positive image had no predicted pixels; it does not
measure every possible detection failure. See `reports/model_comparison.json` for
all counts. Synthetic performance doesn't prove Webots or hardware accuracy.

V2 reached higher normal-light accuracy on its earlier audit but regressed in dim
light. V3 prioritizes the balanced validation score. It improves normal and dim
results against the original, with a small blur tradeoff. Don't call it better in
every condition. Local closed-loop measurements are in `reports/results.md`.

## Failure cases

Multiple red objects can make the largest component win over the intended target.
Broad red backgrounds can look like a target. The aspect-ratio filter can reject
a heavily occluded or very wide target. Stopping can prevent contact without
completing the task; that does not count as an arrival. The obstacle scene can hide
the target completely. There is no route planner or physical robot verification.

## Reproduce

Run `scripts/train_model.py --run-name improved-v3`, then `scripts/compare_models.py`.
The default settings match the published run. Optimizer checkpoints and settings
are saved in `runs/improved-v3`; each new run replaces the active model and report.

The original training settings were `--epochs 12 --train-images 1600 --schedule none
--dice-weight 0 --augmentation none --selection standard`. Preserve the active v3
model before running that command, since training replaces it. Torch versions are
recorded in the reports. Inference timing includes preprocessing and postprocessing
and depends on the machine. If an audit informs another change, reserve new seeds.
