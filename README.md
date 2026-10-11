# webots neural vision + robot control

an e-puck that finds a red box and drives toward it. i started with hsv color
tracking, then added a small cnn and a better controller. the repo has the training
code, trained weights, learning curves, simulator results and tests.

## whats in here

- a 1,113 parameter pytorch cnn that predicts the target mask from rgb pixels
- generated training images with exact labels, shadows, clutter and missing targets
- separate training, validation and test seeds
- hsv and original-controller baselines so the comparisons are reproducible
- proportional steering, smoother wheel speeds and slower approaches
- immediate stops on target loss or bad sensors, then a short scan in place
- front proximity stops with hysteresis so it doesnt chatter at the threshold
- a separate supervisor that moves the scene and records results

the controller only gets camera frames and proximity readings. supervisor positions
are used for scoring, never steering. the hsv reference in telemetry is also only
for evaluation. neural mode doesnt silently switch to hsv.

## setup

use webots r2025a and python 3.12. in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

set the python command in webots preferences to that venv's python. open
`worlds/camera_lab.wbt` and run it. neural mode is the default. press q in the camera
window to stop. the first world load needs internet for the cyberbotics protos.

to compare modes, set the robot's controllerargs to `--backend hsv` or
`--backend original`. the original mode keeps the old steering and stopping rules.

## train it

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py
```

this starts from random weights and trains for 12 epochs. it saves the best model
based on validation iou, then evaluates the test set once. the model used by webots
is `models/target_net.pt`. loss history, checkpoints and settings go in `runs/default`.
the plots and test results go in `reports`.

for another experiment:

```powershell
.\.venv\Scripts\python.exe scripts/train_model.py --epochs 20 --lr 0.001 --train-images 2400 --run-name longer-run
```

each training run replaces the active model and report. keep the run folders if
you want to compare experiments. dont keep changing settings based on the test set.

## test it

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_benchmarks.py --webots "path\to\webots.exe"
.\.venv\Scripts\python.exe scripts/summarize_results.py
```

the simulator runner checks five scenes in three modes. it writes json results and
camera captures. read `reports/results.md` for the numbers and `MODEL_CARD.md` for
what the model was trained on. unit tests run in github actions; the checked-in
webots measurements were run locally.

## limits

this is a simulation project. the training set is procedural, and the model learns
red-box appearance rather than general object recognition. different colors,
lighting, shapes and multiple similar red objects still need more data and tests.
proximity thresholds are raw simulator readings, not distances. blocked means stop;
this project doesnt plan a route around obstacles. no physical robot results are
claimed. the world references cyberbotics assets under their own license.
