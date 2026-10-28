# Learn from original and mirrored hand landmarks

Use `--mirror-training` when preparing a new experiment to train on both
original and geometrically mirrored development examples. This addresses the
hypothesis that the classifier learned an asymmetric preference for one hand.
It preserves all original letter classes, including J and Z, and is independent
of the optional index/middle fingertip weight.

## What is mirrored?

The classifier receives landmarks rather than photo pixels. This implementation
reflects the stored landmarks, then recomputes the selected features:

- Image coordinates: `x' = 1 - x`; y and relative depth z are unchanged.
- World coordinates: `x' = -x`; y and z are unchanged.
- Landmark indices and intended letter labels remain unchanged.
- A recorded `Left`/`Right` label is swapped in the virtual example.
- Cached original features are discarded for the mirrored example.

This is **landmark augmentation**, not a new MediaPipe analysis of flipped
photos. It cheaply supplies a reflected geometric pattern without duplicating
the image dataset. It does not reproduce differences in detection errors,
occlusion or estimated depth between real left and right hands. It also does
not turn a palm view into a hand-back view or add a new camera viewpoint.

For Shin inputs, ideal pairwise distances stay the same under reflection;
the x-axis direction cosines reverse sign. The combined 820-value input can
therefore learn from both orientations. A distance-only input contains no new
distance information under this transformation, although repeated fit rows can
still affect the classifier objective. More fit rows are not more independent
photos or participants. Review whether the reflected handshape retains the
intended label for your alphabet; a still image does not capture full J/Z motion.

## No leakage into validation or test

The experiment first freezes original photo IDs and grouped folds. Within each
fold, fitting uses its training originals plus their mirrors. Validation uses
only the original validation photos. A validation photo's mirrored row is never
included in that fold's fit. Reserved test photos are neither augmented nor
used for fitting. The final model fits all development originals plus mirrors.

Virtual IDs end in `::horizontal_mirror`, with the original ID retained in
`training_row_provenance.csv`. The source dataset and images remain unchanged.
Scaling is fitted on the augmented training rows only; optional distance
weighting still follows preprocessing. No automatic hand-detection switch or
input mirroring is required when using the resulting model with the webcam.

## Train the weight-4 candidate

Run from the project root in the `ktai-hand-gesture` environment:

```bash
python -m scripts.prepare_split --project data/asl_kaggle --out experiments/asl_mirror_w4_01 --model svm --grouping source --folds 3 --seed 42 --allow-no-test --split-from experiments/tip_tuning_01/f3/w4 --index-middle-tip-weight 4 --mirror-training
python -m scripts.train_model --experiment experiments/asl_mirror_w4_01 --variant shin2021_combined
```

This reuses the exact original photo/fold assignments from the existing
weight-4 candidate. The new model has the same SVM settings and input features;
the deliberate change is mirror augmentation. With 60,251 usable development
photos, the final fit uses 120,502 rows. Training time and memory use increase.
Existing output directories are refused; choose a new name for another run.

The full real-data model is not automatically trained during implementation.
Automated tests use explicitly synthetic data. Run the commands above locally
to create the candidate.

Inspect these files under `experiments/asl_mirror_w4_01/shin2021_combined/`:

- `training.json`: augmentation definition, original photo count, mirrored-row
  count, total fit rows, weight and development CV scores.
- `cv_folds.csv`: original training-photo and actual fit-row counts per fold.
- `cv_predictions.csv`: exactly one OOF prediction per original development photo.
- `training_row_provenance.csv`: original/virtual row mapping, source path and group metadata.
- `model.joblib`: the model for ordinary, unmirrored webcam input.

For a focused numerical comparison:

```bash
python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

for experiment in ("experiments/tip_tuning_01/f3/w4", "experiments/asl_mirror_w4_01"):
    folder = Path(experiment) / "shin2021_combined"
    info = json.loads((folder / "training.json").read_text())
    with (folder / "cv_predictions.csv").open() as handle:
        counts = Counter((r["label"], r["prediction"]) for r in csv.DictReader(handle))
    print(experiment, "pooled macro-F1:", info["cv_pooled"]["macro_f1"],
          "R->U:", counts["R", "U"], "U->V:", counts["U", "V"], "V->U:", counts["V", "U"])
PY
```

The original CV photos may not contain enough real left-hand recordings to
measure left-hand improvement. Use new, labeled left- and right-hand sessions
to assess that claim, keeping development exploration separate from the final test.
Training-set augmentation alone does not prove that recognition improved.

## Compare the old and augmented model live

Register both weight-4 models under new names, then open the A/B view:

```bash
python -m scripts.research register --model experiments/tip_tuning_01/f3/w4/shin2021_combined/model.joblib --experiment experiments/tip_tuning_01/f3/w4 --alphabet ASL --name "Weight 4, original training" --out exports/research_setup/mirror_control_w4_01.json
python -m scripts.research register --model experiments/asl_mirror_w4_01/shin2021_combined/model.joblib --experiment experiments/asl_mirror_w4_01 --alphabet ASL --name "Weight 4, original plus mirror training" --out exports/research_setup/mirror_augmented_w4_01.json
python -m apps.webcam ab --model-a exports/research_setup/mirror_control_w4_01.json --model-b exports/research_setup/mirror_augmented_w4_01.json --out exports/live_mirror_w4_01 --intended-label R --countdown 2
```

A uses original training data; B also learned from the reflected landmark rows.
Try both hands with R, then restart with `--intended-label U` or `V`. The intended
label remains assessment metadata; all 26 model answers stay available. Both
models receive the same current frame and one shared detector output. The app's
display mirroring remains separate from training augmentation.

To extend this model later with real development recordings, the existing
`scripts.research train` route preserves its selected mirror-training policy
and recomputes the row counts. Its test exclusion rules remain active. See
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for separate final-test collection.
