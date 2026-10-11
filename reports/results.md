# Results

## Training

- 4800 training images, 320 validation images, 320 test images
- 1113 parameters, 36 epochs
- Reused legacy synthetic test mask IoU (diagnostic): 0.9550
- HSV mask IoU on the same synthetic images: 0.5829

![learning curves](learning_curves.png)

## Model comparison

Both models use the exact same images and a fixed 0.65 threshold. These audit
seeds were reserved before v3 training. Dim images are scaled by 0.65; blurred
images use a 3 x 3 Gaussian kernel with sigma 0.8. The sets have different scenes.

| case | images | original mask IoU | v3 mask IoU | gain (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| fresh | 1000 | 0.9374 | 0.9591 | +2.17 |
| dim | 320 | 0.8755 | 0.9438 | +6.83 |
| blurred | 320 | 0.9155 | 0.9121 | -0.34 |

V2 improved normal-light masks but lost dim-light accuracy. That audit became
development data when it informed brightness augmentation. V3 fixes dim-light
performance against the original on the new set, but blur is still slightly worse.
The earlier reports are in `experiments`. These are synthetic masks, not real-world
accuracy or statistical reliability estimates.

![model comparison](model_improvement.png)

## Webots runs

Each row is one deterministic 40-second simulation, not a statistical reliability estimate.
The moving target travels for five seconds, then holds still. In the occlusion scene
the target is moved out of view from seconds four through seven. The obstacle scene
puts a gray box between the robot and target. Arrival requires a proximity stop
within 8 cm of the target surface. Empty arrival means the task was not completed.

| scene | mode | arrival (sim s) | min target gap (m) | median perception (ms) | forward during occlusion (rad/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| center | original | 31.424 | 0.0105 | 0.301 | - |
| center | HSV | 7.520 | 0.0167 | 0.597 | - |
| center | neural | 7.520 | 0.0163 | 1.259 | - |
| offset | original | 33.024 | 0.0126 | 0.300 | - |
| offset | HSV | 8.544 | 0.0190 | 0.610 | - |
| offset | neural | 8.416 | 0.0194 | 1.242 | - |
| moving | original | 32.064 | 0.0119 | 0.300 | - |
| moving | HSV | 7.872 | 0.0180 | 0.623 | - |
| moving | neural | 7.872 | 0.0179 | 1.315 | - |
| occlusion | original | 32.320 | 0.0106 | 0.302 | 0.50000 |
| occlusion | HSV | 10.848 | 0.0163 | 0.618 | 0.00000 |
| occlusion | neural | 10.688 | 0.0164 | 1.205 | 0.00000 |
| obstacle | original | incomplete | 0.1738 | 0.301 | - |
| obstacle | HSV | incomplete | 0.3229 | 0.503 | - |
| obstacle | neural | incomplete | 0.3229 | 1.071 | - |

A negative gap means the approximate circle/box geometry overlaps. It is a
failure signal, not a successful safety result. The gap is an approximation of
the robot body and does not measure every wheel or protrusion.

Arrival improvements combine changes to speed and control. They are not evidence
that the neural model alone makes the robot faster. HSV reference agreement is
a consistency check, not ground-truth segmentation accuracy on Webots frames.

The obstacle scene can fully hide the target. Staying stopped is useful but
doesn't count as completing the route. This controller does not do obstacle avoidance.
