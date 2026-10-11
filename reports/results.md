# results

## training

- 1600 training images, 320 validation images, 320 test images
- 1113 parameters, 12 epochs
- held-out synthetic mask iou: 0.9375
- hsv mask iou on the same synthetic images: 0.5829

![learning curves](learning_curves.png)

## webots runs

each row is one deterministic 40-second simulation, not a statistical reliability estimate.
the moving target travels for five seconds, then holds still. in the occlusion scene
the target is moved out of view from seconds four through seven. the obstacle scene
puts a gray box between the robot and target. arrival requires a proximity stop
within 8 cm of the target surface. empty arrival means the task was not completed.

| scene | mode | arrival (sim s) | min target gap (m) | median perception (ms) | forward during occlusion (rad/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| center | original | 31.424 | 0.0105 | 0.305 | - |
| center | hsv | 7.520 | 0.0167 | 0.600 | - |
| center | neural | 7.520 | 0.0165 | 1.346 | - |
| offset | original | 33.024 | 0.0126 | 0.307 | - |
| offset | hsv | 8.544 | 0.0190 | 0.618 | - |
| offset | neural | 8.416 | 0.0195 | 1.267 | - |
| moving | original | 32.064 | 0.0119 | 0.307 | - |
| moving | hsv | 7.872 | 0.0180 | 0.604 | - |
| moving | neural | 7.840 | 0.0180 | 1.295 | - |
| occlusion | original | 32.320 | 0.0106 | 0.300 | 0.50000 |
| occlusion | hsv | 10.848 | 0.0163 | 0.601 | 0.00000 |
| occlusion | neural | 10.592 | 0.0168 | 1.259 | 0.00000 |
| obstacle | original | incomplete | 0.1738 | 0.306 | - |
| obstacle | hsv | incomplete | 0.3229 | 0.512 | - |
| obstacle | neural | incomplete | 0.3229 | 1.042 | - |

a negative gap means the approximate circle/box geometry overlaps. it is a
failure signal, not a successful safety result. the gap is an approximation of
the robot body and does not measure every wheel or protrusion.

arrival improvements combine changes to speed and control. they are not evidence
that the neural model alone makes the robot faster. hsv reference agreement is
a consistency check, not ground-truth segmentation accuracy on webots frames.

the obstacle scene can fully hide the target. staying stopped is useful but
doesnt count as completing the route. this controller does not do obstacle avoidance.
