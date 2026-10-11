# train named targets

the catalog model predicts one background class and one mask per named target.
`--target` picks the mask used for steering. other classes stay distractors. it
doesnt switch to another class if the selected one disappears.

## the included demo

```powershell
.\.venv\Scripts\python.exe scripts/make_target_data.py
.\.venv\Scripts\python.exe scripts/train_targets.py --data data/targets_demo
.\.venv\Scripts\python.exe scripts/run_target_benchmarks.py --webots "path\to\webots.exe"
```

the generator writes 3,000 training images, 320 validation images and 400 test
images. split seeds are 20261023, 20261024 and 20261025. boxes can overlap, have
shadows or be missing. their labels come from drawing geometry, not color masks.
brightness varies from 0.55 to 1.2. raw generated data is ignored by git and can be
recreated with the command above. use a new `--output` folder to regenerate data.

training starts from random weights. the 1,460-parameter cnn uses weighted cross
entropy, adam and cosine decay for 24 epochs. validation macro foreground iou
selects the checkpoint; test images are scored once afterward. masks use a fixed
0.65 probability threshold. reports keep all class scores and missed targets.

open `worlds/target_selection_lab.wbt`. change the e-puck's controllerargs to:

```text
"--backend" "catalog" "--target" "blue_box"
```

the included names are `red_box`, `blue_box` and `green_box`. the original red-only
model is still available with `--backend neural`. catalog mode loads its own
trained weights and metadata from `models/targets`.

## your own labeled data

make a folder with `manifest.json`, rgb images and matching masks. masks are
single-channel 8-bit pngs with integer class ids: 0 is background, 1 is the first
name in `classes`, 2 is the second, and so on. a mask needs the same width and
height as its image. use visible object pixels only. cover different lighting,
positions, backgrounds and negative examples where the target is absent.

```json
{
  "classes": ["my_target", "other_target"],
  "samples": [
    {"image": "images/train_01.png", "mask": "masks/train_01.png", "split": "train"},
    {"image": "images/val_01.png", "mask": "masks/val_01.png", "split": "validation"},
    {"image": "images/test_01.png", "mask": "masks/test_01.png", "split": "test"}
  ]
}
```

this is the format, not a large enough training set. every split needs examples
of every class. keep separate scenes or recording sessions in separate splits.
the loader rejects identical decoded images across splits and invalid mask ids;
it cannot find every near-duplicate or a recording-session leak. image and mask
paths must stay inside the dataset folder. class names are unique lowercase
identifiers without spaces.

```powershell
.\.venv\Scripts\python.exe scripts/train_targets.py --data "C:\path\to\my_data" --output models/my_targets
```

then use these controllerargs:

```text
"--backend" "catalog" "--target-bundle" "models/my_targets" "--target" "my_target"
```

the bundle contains `target_catalog.pt` and `catalog.json`. keep them together.
relative bundle paths start from the project root, not the controller folder.
unknown target names, inconsistent model metadata and failed sensor readings stop
the controller. custom data needs its own held-out evaluation and simulator runs.

## scope

the tested demo recognizes colored-box classes. it does not identify a unique
instance among several boxes of the same class. the largest selected component
wins. it is not a general object detector, and the small network has limited
shape context. supporting new objects in the dataset format does not mean those
objects have been trained or verified. no physical robot results are claimed.
