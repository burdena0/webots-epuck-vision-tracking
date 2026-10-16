# Webots E-puck Vision Tracking

An e-puck that finds a colored box and drives toward it in Webots. I started with
HSV tracking, then added a small PyTorch CNN and improved the steering and stopping
logic. Everything here runs in simulation.

The controller uses camera frames and proximity sensors. It stops when the target
disappears or an obstacle gets too close. Supervisor positions are used to score
the runs, never to steer the robot.

## Run it

Use Webots R2025a and Python 3.12. From the repo folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/configure_webots_python.py
```

Open `worlds/target_selection_lab.wbt`. Set the robot's `controllerArgs` to:

```text
"--backend" "catalog" "--target" "blue_box"
```

Use `red_box` or `green_box` to select another color. For the original red-only
demo, open `worlds/camera_lab.wbt`. The first world load downloads Cyberbotics PROTOs.
Press Q in the camera window to stop.

## Train

The catalog CNN has 1,460 parameters and was trained on 3,000 synthetic images.
The older red-only CNN has 1,113 parameters. Each has its own weights.

```powershell
python scripts/make_target_data.py
python scripts/train_targets.py --data data/targets_demo
```

Training writes new weights to `models/targets`. Copy the existing weights first
if you want to keep the included model and reproduce its saved results.

## Tests and results

```powershell
python -m unittest discover -s tests -v
python scripts/verify_evaluation.py
```

The saved comparison covers 3,600 synthetic images and 108 paired Webots trials.
Both detectors use matching scenes and the same controller. HSV handles clean
colors and near-color clutter better. The CNN scored higher on blurred images;
the small dim-light difference was uncertain. Both struggle to select a specific
box when several boxes share its color.

- [Image results](reports/perception_audit/results.json)
- [Robot results](reports/closed_loop_verified/summary.json)
- [Image comparison](reports/perception_audit/comparison.png)
- [Robot comparison](reports/closed_loop_verified/comparison.png)

The report folders also contain settings, source hashes, camera examples, and
per-step telemetry. Tests cover detection, steering, stopping, and trial scoring.

To run another comparison from the activated environment:

```powershell
python scripts/evaluate_perception.py --output reports/new_images
python scripts/run_perception_trials.py --webots "C:\path\to\webots.exe" --hsv-settings reports/new_images/hsv_settings.json --output reports/new_trials
```

Use fresh output folders. These results are from generated images and simulation;
the project has not been tested on a physical robot. The controller stops for
obstacles but does not plan a route around them.
