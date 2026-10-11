# Webots Neural Vision and Robot Control

An e-puck robot that finds a target box and drives toward it in Webots. I started with HSV color tracking, then added a small CNN and improved the steering and stopping logic. This repo includes training code, weights, learning curves, simulator results, and tests.

## What's in here

- A 1,113 parameter PyTorch CNN that predicts the target mask from RGB pixels
- Generated training images with exact labels, shadows, clutter and missing targets
- Brightness augmentation, dice loss and a cosine learning-rate schedule
- Separate training, validation and test seeds
- HSV and original-controller baselines so the comparisons are reproducible
- Proportional steering, smoother wheel speeds and slower approaches
- Immediate stops on target loss or bad sensors, then a short scan in place
- Front proximity stops with hysteresis so it doesn't chatter at the threshold
- A separate supervisor that moves the scene and records results
- A trained red/blue/green target catalog with explicit target selection
- An image-and-mask dataset loader for training your own named targets

The controller only gets camera frames and proximity readings. Supervisor positions
are used for scoring, never steering. The HSV reference in telemetry is also only
for evaluation. Neural mode doesn't silently switch to HSV.

## Matched perception evaluation

The latest evaluation compares the 1,460-parameter catalog CNN with a calibrated
HSV detector on all three colors. The audit uses 3,600 fresh synthetic
scenes across normal light, dim light, blur, partial occlusion, near-color clutter,
and missing targets. The weights and CNN threshold stay fixed. HSV saturation/value
settings are chosen on separate calibration scenes before the audit is scored.

```powershell
python scripts/evaluate_perception.py
python scripts/run_perception_trials.py --webots "path\to\webots.exe"
```

Use the project's activated venv for these commands. The first command saves
calibration, protocol hashes, per-class results, paired bootstrap intervals,
latency distributions, and image/mask examples in `reports/perception_audit`.
The second compares both detectors using the same controller and scene seeds,
with repeated selected-target approaches in Webots. It records successful stops,
wrong-object stops, actual object contacts, missing-target behavior, latency,
raw per-step telemetry, and logs in `reports/closed_loop_verified`.

After the trials finish, `python scripts/verify_evaluation.py` regenerates the
image identities and recomputes every trial score from telemetry using the tested
scoring implementation.
`python scripts/check_arrival_identity.py` adds a conservative post-audit review:
the selected target must be nearer than both distractors throughout the stop.
The original frozen distance-based scores remain preserved separately.
`python scripts/run_scoring_probe.py --webots "path\to\webots.exe"` starts a
deliberate collision diagnostic to verify that object contacts are actually
observed and prevent a success. It is kept separate from the performance trials.

Read [the image audit](reports/perception_audit/README.md) and
[the robot trials](reports/closed_loop_verified/README.md) together. HSV is stronger
on clean colors and near-color clutter; the CNN's image-level advantage is in dim
light and blur. A segmentation improvement does not automatically improve robot
completion. Neither detector can identify a specific instance among same-color
objects. These measurements cover simulation and generated data only.

Each runner preserves existing results instead of silently replacing them. Use
`--output reports/new_audit` for a new image audit. Simulator trials can resume
using `--resume` only when code, model, calibration, and trial settings match the
saved protocol. Do not tune on an audit and keep calling it untouched.

## Setup

Use Webots r2025a and Python 3.12. In this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the Python command in Webots preferences to that venv's Python. Open
`worlds/camera_lab.wbt` and run it. Neural mode is the default. Press q in the camera
window to stop. The first world load needs internet for the Cyberbotics protos.

For a downloaded project, run `python scripts/configure_webots_python.py` from
the activated venv to set the three project controllers to your local interpreter.
Machine-specific `runtime.ini` files are not distributed or committed.

To compare modes, set the robot's controllerargs to `--backend hsv` or
`--backend original`. The original mode keeps the old steering and stopping rules.

## Train it

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py
```

This starts from random weights and trains on 4,800 images for 36 epochs. It varies
brightness from 0.5 to 1.25 each time an image is seen. The best checkpoint is picked
using the mean IoU on standard and dim validation views. The model used by Webots
is `models/target_net.pt`. Loss history, checkpoints and settings go in `runs/default`.
The plots and measurements go in `reports`.

For another experiment:

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --run-name improved-v3
.\.venv\Scripts\python.exe scripts/compare_models.py
```

Each training run replaces the active model and report. Keep the run folders if
you want to compare experiments. `models/baseline_target_net.pt` keeps the original
model for comparison. The 320-image legacy test was reused, so its a diagnostic now.
The final comparison uses new seeds reserved before v3 training. If you tune based
on those results, you need another untouched audit set.

The first longer run improved normal-light masks but regressed in dim light. V3
adds brightness variation to fix that. Both experiment reports are kept, including
the remaining small blur regression. Read `MODEL_CARD.md` for the full comparison.

## Test it

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_benchmarks.py --webots "path\to\webots.exe"
.\.venv\Scripts\python.exe scripts/summarize_results.py
```

The simulator runner checks five scenes in three modes. It writes JSON results and
camera captures. Read `reports/results.md` for the numbers and `MODEL_CARD.md` for
what the model was trained on. Unit tests run in GitHub actions; the checked-in
Webots measurements were run locally.

## Select a trained target

Open `worlds/target_selection_lab.wbt`. Set the robot controllerargs to
`--backend catalog --target blue_box` to approach blue while ignoring red and green.
Use `red_box` or `green_box` for the other trained classes. The camera preview
shows the selected target name. If its missing, the robot stops and only scans in
place. It doesn't pick a different color for you.

The catalog model was trained on 3,000 labeled synthetic images. Its held-out
400-image macro mask IoU is 0.9813. Read [target data](docs/target_data.md) for the
training commands, custom image/mask format and limits. Selection measurements
and class-level results are in `reports/target_results.md`.

## Limits

This is a simulation project. The training set is procedural, and the model learns
red-box appearance rather than general object recognition. Different colors,
lighting, shapes and multiple similar red objects still need more data and tests.
Blur remains a weakness: v3 scored slightly below the original in that stress check.
Proximity thresholds are raw simulator readings, not distances. Blocked means stop;
this project doesn't plan a route around obstacles. No physical robot results are
claimed. The world references Cyberbotics assets under their own license.
