# CNN and HSV Robot Trials

There are 108 Webots trials: six conditions, three target colors, three starting configurations, and two detectors. CNN and HSV runs start from matching scenes and use the same steering and stopping logic. Each table row summarizes nine trials.

| Case | Detector | Successes / trials | 95% Wilson interval | Object collision trials | Median perception ms |
| --- | --- | ---: | ---: | ---: | ---: |
| standard | catalog | 9 / 9 | 70.1-100.0% | 0 | 1.151 |
| standard | HSV | 9 / 9 | 70.1-100.0% | 0 | 0.617 |
| dim | catalog | 9 / 9 | 70.1-100.0% | 0 | 1.235 |
| dim | HSV | 9 / 9 | 70.1-100.0% | 0 | 0.707 |
| occlusion | catalog | 9 / 9 | 70.1-100.0% | 0 | 1.846 |
| occlusion | HSV | 9 / 9 | 70.1-100.0% | 0 | 0.855 |
| near_color | catalog | 9 / 9 | 70.1-100.0% | 0 | 1.711 |
| near_color | HSV | 8 / 9 | 56.5-98.0% | 0 | 0.873 |
| missing | catalog | 9 / 9 | 70.1-100.0% | 0 | 1.021 |
| missing | HSV | 9 / 9 | 70.1-100.0% | 0 | 0.569 |
| same_color | catalog | 6 / 9 | 35.4-87.9% | 0 | 1.522 |
| same_color | HSV | 4 / 9 | 18.9-73.3% | 0 | 0.719 |

## Scoring and limits

- Approach success requires a continuous 0.5-second stationary proximity stop with the selected target visible, its approximate surface gap between 0 and 8 cm, and the target at least 1 mm nearer than either distractor. No observed target/distractor contact or approximate overlap beyond 1 mm is allowed throughout the run.
- Missing-target success requires zero positive forward commands for the entire trial, no arrival, and no object collision. Turning in place is permitted.
- Contact checks match global contact locations between e-puck descendants and each target/distractor to within 1 micrometer. Webots contact node IDs identify the queried body, not its partner, and are not used to infer collision partners. Ground/floor-only contacts are excluded. The geometric circle/box approximation flags additional overlap; it does not certify physical clearance.
- Occlusion moves the selected box out of view during seconds 3-6, then restores it. Dim scenes use low ambient/direct lighting. Near-color scenes add orange/cyan/purple distractors. Same-color scenes introduce a nearer instance of the selected color and test instance ambiguity; neither detector is trained to identify a specific same-color object.
- Identical seeds define identical starting positions and heading for both detectors. Trial order alternates by seed. Models, controller parameters, and HSV settings stay frozen.
- Latency covers selected detector preprocessing/inference/postprocessing, excluding camera acquisition, control, and telemetry; first 30 samples are omitted. JSON includes median and p95.
- These are nine fixed simulation configurations per cell, not nine independent real-world trials. Wilson intervals illustrate small sample uncertainty and do not establish deployment reliability.
- Raw per-step telemetry and per-trial logs are saved for inspection. Incomplete telemetry and controller faults fail the runner instead of counting as successes.
