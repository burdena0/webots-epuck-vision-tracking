# Train named targets

The catalog model predicts one background class and one mask per named target.
`--target` picks the mask used for steering. Other classes stay distractors. It
doesn't switch to another class if the selected one disappears.

## The included demo

```powershell
.\.venv\Scripts\python.exe scripts/make_target_data.py
.\.venv\Scripts\python.exe scripts/train_targets.py --data data/targets_demo
.\.venv\Scripts\python.exe scripts/run_target_benchmarks.py --webots "path\to\webots.exe"
```

The generator writes 3,000 training images, 320 validation images and 400 test
images. Split seeds are 20261023, 20261024 and 20261025. Boxes can overlap, have
shadows or be missing. Their labels come from drawing geometry, not color masks.
Brightness varies from 0.55 to 1.2. Raw generated data is ignored by git and can be
recreated with the command above. Use a new `--output` folder to regenerate data.

Training starts from random weights. The 1,460-parameter CNN uses weighted cross
entropy, Adam and cosine decay for 24 epochs. Validation macro foreground IoU
selects the checkpoint; test images are scored once afterward. Masks use a fixed
0.65 probability threshold. Reports keep all class scores and missed targets.

Open `worlds/target_selection_lab.wbt`. Change the e-puck's controllerargs to:

```text
"--backend" "catalog" "--target" "blue_box"
```

The included names are `red_box`, `blue_box` and `green_box`. The original red-only
model is still available with `--backend neural`. Catalog mode loads its own
trained weights and metadata from `models/targets`.

## Your own labeled data

Make a folder with `manifest.json`, RGB images and matching masks. Masks are
single-channel 8-bit PNGs with integer class IDs: 0 is background, 1 is the first
name in `classes`, 2 is the second, and so on. A mask needs the same width and
height as its image. Use visible object pixels only. Cover different lighting,
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

This is the format, not a large enough training set. Every split needs examples
of every class. Keep separate scenes or recording sessions in separate splits.
The loader rejects identical decoded images across splits and invalid mask IDs;
it cannot find every near-duplicate or a recording-session leak. Image and mask
paths must stay inside the dataset folder. Class names are unique lowercase
identifiers without spaces.

```powershell
.\.venv\Scripts\python.exe scripts/train_targets.py --data "C:\path\to\my_data" --output models/my_targets
```

Then use these controllerargs:

```text
"--backend" "catalog" "--target-bundle" "models/my_targets" "--target" "my_target"
```

The bundle contains `target_catalog.pt` and `catalog.json`. Keep them together.
Relative bundle paths start from the project root, not the controller folder.
Unknown target names, inconsistent model metadata and failed sensor readings stop
the controller. Custom data needs its own held-out evaluation and simulator runs.

## Scope

The tested demo recognizes colored-box classes. It does not identify a unique
instance among several boxes of the same class. The largest selected component
wins. It is not a general object detector, and the small network has limited
shape context. Supporting new objects in the dataset format does not mean those
objects have been trained or verified. No physical robot results are claimed.
