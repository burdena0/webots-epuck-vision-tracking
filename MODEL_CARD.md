# target net

## job

take a camera image, predict a red target mask, then use its centroid for steering.
the network doesnt output wheel speeds or learn a driving policy. the controller
handles steering, speed limits and proximity stops separately.

## model

- three convolution layers: 3 -> 8 -> 12 -> 1 channels
- relu after the first two layers, sigmoid on the output
- 1,113 trainable parameters, 96 x 72 rgb input
- cpu inference; exported weights are in `models/target_net.pt`
- adam optimizer, learning rate 0.003, weighted binary cross entropy
- batch size 64, 12 epochs, deterministic seed 20261011

## data

1,600 training images, 320 validation images and 320 test images. each split uses
a different seed. the generator draws boxes and records their visible pixels as
labels. it varies position, size, color intensity, shadows, background clutter,
partial occlusion and pixel noise. some images have no target.

these are procedural images, not photos or recorded webots frames. there are no
pretrained weights or hsv-generated training labels. the simulator benchmarks are
separate from training and validation.

## measurements

the selected model scored 0.9375 mask iou on the 320-image synthetic test set.
its centroid mean absolute error was 0.265 pixels at 96-pixel width, for positive
images with a nonempty predicted mask. there were no empty-mask misses and one
false positive on a negative image using the raw-mask area rule.

the hsv mask baseline scored 0.5829 iou on those same images. both scores use raw
pixel masks before component filtering. synthetic performance doesnt prove the
same result in webots or on hardware. real webots camera captures and closed-loop
measurements are in `reports/results.md`.

## failure cases

multiple red objects can cause the largest component to win over the intended
target. broad red backgrounds can look like a target. the aspect-ratio filter can
reject a heavily occluded or very wide target. stopping can prevent a collision
without completing the task; that counts as a limitation, not a successful arrival.
there is no claim of collision-free navigation or physical robot reliability.

## reproduce

run `scripts/train_model.py` with the default settings. validation iou selects the
checkpoint, then the test split is scored once. settings and optimizer checkpoints
are saved in `runs/default`; the active model and report are replaced by a new run.
latencies depend on the machine and include preprocessing plus postprocessing.
