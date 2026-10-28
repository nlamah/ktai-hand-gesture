# Tune the index/middle fingertip distance

`--index-middle-tip-weight` is an optional SVM training parameter. It changes the
relative scale of existing fingertip-distance inputs, without adding features,
changing labels or restricting the model's original output classes.

## Which distance is weighted?

MediaPipe landmark 8 is the index fingertip and landmark 12 is the middle
fingertip. The training pipeline selects these existing columns by name:

| Column | Definition | Available in |
|---|---|---|
| `shin_dist_08_12` | Image-space 3D distance, followed by the existing per-photo z-score over all 190 Shin distances | `shin2021_distance`, `shin2021_combined`, `shin2021_with_finger_features`, `all_features` |
| `geom_tip_distance_index_middle` | World-landmark 3D distance divided by the wrist-to-middle-MCP distance | `finger_features_only`, `with_features`, `shin2021_with_finger_features`, `all_features` |

If both columns are present, the parameter applies to **both**. They are two
representations of the same fingertip pair with different coordinate systems
and normalization; they are not identical measurements. Other columns keep
their original values. The raw extraction functions and feature counts remain
unchanged. Variants without an explicit distance (`without_features` and
`shin2021_angle`) reject a nondefault weight. Nondefault weighting currently
supports SVM only.

## Meaning and placement

- `1` is the default and preserves the existing estimator and behavior.
- `2` multiplies each selected feature coordinate by two.
- `0` sets the selected distance coordinates to zero for an explicit ablation.
- Negative, nonnumeric or nonfinite weights are rejected.

The weight is applied **after all preprocessing**, immediately before SVM
fit/predict. Applying it before `StandardScaler` would let the scaler cancel
positive rescaling. Hybrid inputs retain their existing selective scaler and
column ordering; the weighting step follows its actual output order.

For an RBF SVM, the kernel uses squared Euclidean distance. Multiplying one
coordinate by `w` multiplies its squared-distance contribution by `w²`, at a
fixed gamma. With the existing `gamma="scale"` profile, gamma is still estimated
from each fold's weighted training matrix, so its numerical value can also
change. The weight is a relative input-scale setting, not a guarantee of better
classification or a percentage importance. See the official
[SVM kernel definition](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)
and [SVC gamma definition](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

The implementation is isolated in
`gesturelab/training/distance_weighting.py`. The frozen experiment, training
summary, saved model config, comparison reports and model registration record
the weight. The serialized sklearn pipeline contains the multiplication factors,
so ordinary webcam prediction applies the exact training weight automatically.
There is no live-only slider: a changed weight requires a newly trained model.

## Train one candidate on the same development split

Use a new experiment directory. This example keeps all 26 letters and reuses
the baseline's exact photo IDs and grouped folds. The value `2` is an example,
not a selected optimum:

```bash
python -m scripts.prepare_split --project data/asl_kaggle --out experiments/asl_tip_weight_2_01 --model svm --grouping source --folds 3 --seed 42 --allow-no-test --split-from experiments/shin2021_finger_kaggle_01/svm --index-middle-tip-weight 2
python -m scripts.train_model --experiment experiments/asl_tip_weight_2_01 --variant shin2021_combined
python -m scripts.research register --model experiments/asl_tip_weight_2_01/shin2021_combined/model.joblib --experiment experiments/asl_tip_weight_2_01 --alphabet ASL --name "Shin, tip distance weight 2" --out exports/research_setup/tip_weight_2.json
python -m apps.webcam ab --model-a exports/research_setup/baseline.json --model-b exports/research_setup/tip_weight_2.json --out exports/live_tip_weight_2 --intended-label R --countdown 2
```

For the 901-value hybrid, train `shin2021_with_finger_features` in the new
experiment instead, and update the registration path. That version weights both
distance columns. To compare weights on the same input, prepare separate new
experiments with `1` and your candidate weight using the same `--split-from`.
Inspect their saved `cv_predictions.csv`, overall/per-letter metrics and
confusions on the same IDs. No parameter sweep runs automatically.

Tune only on development data. Choosing the best weight from several trials can
make development CV optimistic; freeze the final choice before collecting and
evaluating new independent test observations. Follow
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for that final comparison.

Existing model files, source manifests and old results are not modified. Their
live input extraction remains compatible because the feature-extractor source
has not changed. Existing frozen experiments still enforce exact training-code
hashes: prepare a new experiment to retrain under this updated training code.
The weight-2 example above has not been trained on the real Kaggle data during
implementation; software tests use clearly marked synthetic data.

## Compare several weights and fold counts

Run this from the project root in the project environment. It explicitly trains
six SVM/Shin candidates: weights 1, 2 and 4, with 2 and 3 grouped folds. It keeps
all 26 classes. For a given fold count, the fixed seed, grouping and unchanged
dataset produce the same assignments for each weight. Different fold counts
intentionally produce different assignments. Do not use one three-fold
`--split-from` template when requesting two folds.

```bash
(
  set -e
  for folds in 2 3; do
    for weight in 1 2 4; do
      experiment="experiments/tip_tuning_01/f${folds}/w${weight}"
      python -m scripts.prepare_split --project data/asl_kaggle --out "$experiment" --model svm --grouping source --folds "$folds" --seed 42 --allow-no-test --index-middle-tip-weight "$weight"
      python -m scripts.train_model --experiment "$experiment" --variant shin2021_combined
    done
  done
)
```

Use a new root name, such as `tip_tuning_02`, for another sweep; existing runs are
preserved. The subshell stops if a command fails. Training all six candidates can
take a while on the full dataset. Change the lists only when you deliberately
want additional trials. Each label must occur in at least as many distinct
source groups as the requested fold count. These groups are filename blocks,
not verified participants.

Print a compact summary, including the main U/V/R confusions:

```bash
python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

print("folds weight pooled_F1 R_to_U U_to_R U_to_V V_to_U")
for path in sorted(Path("experiments/tip_tuning_01").glob("f*/w*/shin2021_combined/training.json")):
    info = json.loads(path.read_text())
    plan = json.loads((path.parent.parent / "split.json").read_text())
    with (path.parent / "cv_predictions.csv").open() as handle:
        counts = Counter((row["label"], row["prediction"]) for row in csv.DictReader(handle))
    print(plan["n_folds"], info["index_middle_tip_weight"],
          f'{info["cv_pooled"]["macro_f1"]:.6f}',
          counts["R", "U"], counts["U", "R"], counts["U", "V"], counts["V", "U"])
PY
```

Compare each candidate with weight 1 at the **same fold count**. Prefer an
improvement that is consistent across the chosen fold counts and check whether
reducing U/V/R confusion harms other letters. Pooled F1 uses all 26 classes.
These repeated CV setups reuse the same development photos and are not
independent replications. Do not select the single best fold count as proof of
improvement; use these results for development selection, then a new final test.

For example, if you choose weight 2 after this analysis, register its three-fold
run and compare it with the original baseline:

```bash
python -m scripts.research register --model experiments/tip_tuning_01/f3/w2/shin2021_combined/model.joblib --experiment experiments/tip_tuning_01/f3/w2 --alphabet ASL --name "Shin, tip weight 2" --out exports/research_setup/tip_tuning_01_selected.json
python -m apps.webcam ab --model-a exports/research_setup/baseline.json --model-b exports/research_setup/tip_tuning_01_selected.json --out exports/live_tip_tuning_01 --intended-label R --countdown 2
```

Replace `f3/w2` and the display name with your chosen run. Each run's final
`model.joblib` was refitted on all usable development photos; fold count controls
validation, not the proportion used by this final fit. The weight is already
inside that saved pipeline, so no weight flag is needed in the live command.
A is the original weight-1 baseline and B is your selected candidate. To inspect
U or V, restart with `--intended-label U` or `--intended-label V`; predictions
still cover the complete alphabet. Live observations are development evidence,
not an independent final performance measurement.

## Existing U / V / R confusion

If one hand is recognized better than the other, keep the selected weight and
run a separate [mirror-training experiment](MIRROR_TRAINING.md). This changes
training augmentation while retaining the same original validation assignments.

The saved grouped out-of-fold predictions from
`experiments/shin2021_finger_kaggle_01/svm` already show confusion. These are
development photos predicted by the fold model that did not train on that
photo's group, not predictions on the final all-development model's training
matrix. All 26 original output classes remain possible.

For `shin2021_combined`:

| Actual letter | Predicted U | Predicted V | Predicted R | Other letters | Total |
|---|---:|---:|---:|---:|---:|
| U | 2,447 | 8 | 40 | 18 | 2,513 |
| V | 12 | 2,510 | 5 | 25 | 2,552 |
| R | 202 | 1 | 2,326 | 14 | 2,543 |

The main confusion is R → U: 202/2,543 (7.94%). U → R occurs 40/2,513 times
(1.59%). In the existing 901-value Shin/finger model, R → U occurs 179 times and
U → R 38 times on the same IDs/folds. This does not establish that weighting
this distance will resolve those errors. Fingertip separation alone may not
distinguish crossing from adjacent fingers; existing geometry also includes
`geom_crossed_index_middle` and `geom_depth_index_middle`.

The read-only analysis is saved locally in `exports/uvr_confusion_01/`:
`SUMMARY.md`, `confusion_counts.csv`, `per_letter.csv`,
`confused_predictions.csv` and `summary.json`. It covers all four existing SVM
variants, verifies equal ID/label/fold mappings and records source file hashes.
It fitted no model and evaluated no reserved test. Confusion does not prove
wrong ground-truth labels, which require human image review. Detection failures
are outside these classifier scores, and filename-block grouping does not
establish participant-independent performance.
