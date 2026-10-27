# Train on the Kaggle ASL Alphabet dataset

The [external-data workflow](WORKFLOW.md#alternative-train-locally-with-external-asl-data)
explains how this route relates to the COCO import and the NGT recording workflow.

Run these commands from the project root with the recognition environment active:

```bash
conda activate ktai-hand-gesture
```

Install the environment using the [README instructions](../README.md#installation)
if needed. You do not need the Kaggle API, Roboflow SDK or an API key when the
photos are already extracted locally. A webcam is not needed for training.
The first landmark analysis downloads the MediaPipe model if it is missing.

## Complete run

The extracted download is expected here:

```text
data/external/archive-kaggle/
├── asl_alphabet_train/asl_alphabet_train/A/A1.jpg ... Z/Z3000.jpg
└── asl_alphabet_test/asl_alphabet_test/A_test.jpg ... Z_test.jpg
```

Start the full import, landmark analysis, training and development comparison:

```bash
python -u -m scripts.train_kaggle_dataset
```

This processes **all 78,000 A–Z training photos**, plus the 26 supplied letter
test photos. `del`, `nothing` and `space` are outside this letter experiment.
Exact duplicate pixels are stored once. In this download, **all 26 test photos
are exact copies of training photos**: A_test.jpg matches A2995.jpg, and the other
letters match their first training photo. The complete-run command excludes these
copies as test examples while preserving all training photos. It records each
overlap and reports that no independent final test remains. Conflicting labels
stop the import.
Training uses images with successful single-hand detections; detection failures
remain recorded in the dataset. No fixed subset is sampled.

The command creates a separate dataset at `data/asl_kaggle` and a Random Forest
experiment at `experiments/asl_kaggle_rf_01`. It leaves the original download and
the NGT and Roboflow datasets in place. Both input variants share three folds:
126 normalized image/world coordinates versus those same coordinates plus 81
finger features. The classifier has 150 trees, as configured in
`gesturelab/training/classifiers.py`.

## Progress and interruption

The terminal prints the current stage and progress. The latest status is also in
`data/asl_kaggle_run.json`; detection counts are written to
`data/asl_kaggle/reports/dataset_summary.json` when analysis finishes.

Import and analysis save progress every 1,000 photos. For a normal interruption,
completed work in the current batch is saved too. An abrupt process crash can
require reprocessing the last unsaved batch. Analysis stores landmarks without
redundant cached feature dictionaries; training and inspection calculate features
from those landmarks. This keeps the large manifest smaller.

Rerun the same command to resume import, analyze pending photos and skip finished
model variants. Keep the source files and options unchanged. Model fitting within
an unfinished variant restarts from the beginning. A partial variant output or a
changed frozen dataset/code is refused; use a new experiment directory after
reviewing the error. Do not run two jobs against the same output simultaneously.

For another experiment name:

```bash
python -u -m scripts.train_kaggle_dataset --out experiments/asl_kaggle_rf_02
```

This reuses the imported images and analyzed landmarks in `data/asl_kaggle`.
If the first run already completed, using the original command reuses its
finished models; use a new `--out` to fit them again. Keep the same `--source`
and `--project` values if you used custom locations. The complete-run command
always selects Random Forest; it has no `--model` option.

## Grouping and evaluation limits

The download has no reliable participant or recording-session metadata. The
participant is recorded as `unknown`. Consecutive numbered training photos are
kept together in blocks of 500 per letter: A1–A500 and B1–B500, for example, belong
to the same block. The six blocks are divided across three folds. These blocks
are filename groups, **not verified sessions or people**. Photos in different
blocks can still be related, so cross-validation is exploratory and cannot
establish accuracy on new people. The report includes this limitation.

The supplied test folder contains just **one letter photo per class**, and every
one duplicates a training photo. It cannot provide an independent final check.
Collect new test photos and freeze a new experiment to evaluate generalization.
For another download with genuinely distinct test photos, the importer keeps
those photos reserved; the complete-run command does not evaluate them.
Single photos also do not represent the full motion of dynamic letters.

## Results and webcam

After the command completes, open:

```text
experiments/asl_kaggle_rf_01/comparison_cv/report.html
experiments/asl_kaggle_rf_01/without_features/model.joblib
experiments/asl_kaggle_rf_01/with_features/model.joblib
```

The report is a local HTML file; open it in your browser. Each variant also has
`training.json` with feature counts and CV scores, `cv_predictions.csv` with
per-photo validation predictions, and `training_features.csv` with the model
inputs. `comparison_cv/metrics.csv` contains the paired comparison scores.
Scores in this report describe development CV on successful detections, not
accuracy on all downloaded images or on a new person using the webcam.

Use the trained model with the webcam:

```bash
python -m apps.webcam predict --project data/asl_kaggle --model experiments/asl_kaggle_rf_01/with_features/model.joblib
```

There are no independent final-test photos in this run. Do not interpret the
provided test folder as a final evaluation set.

The supplied app reference drawings are NGT. Use the [ASL chart](../assets/asl_alphabet.jpg)
for ASL pose guidance instead.

## Run individual stages

Use this alternative when you want to inspect each step:

```bash
python -m scripts.import_kaggle_dataset --source data/external/archive-kaggle --project data/asl_kaggle --exclude-overlapping-test
python -m scripts.analyze_dataset --project data/asl_kaggle --checkpoint-every 1000 --no-cache-features
python -m scripts.prepare_split --project data/asl_kaggle --out experiments/asl_kaggle_rf_01 --folds 3 --grouping source --model random_forest --allow-no-test
python -m scripts.train_without_features --experiment experiments/asl_kaggle_rf_01
python -m scripts.train_with_features --experiment experiments/asl_kaggle_rf_01
python -m scripts.compare_results --experiment experiments/asl_kaggle_rf_01
```

Without `--exclude-overlapping-test`, the individual import command stops on
overlapping test pixels. It never promotes test copies into independent examples.
Add `--resume` to the import command after an interrupted Kaggle import. Use an
existing frozen experiment only while its data and feature/training code remain
unchanged.

For an already fully imported dataset, skip the import command and reuse
`data/asl_kaggle`. Analysis processes only pending photos; an analyzed dataset
does not need to be processed again before each training run. Use a fresh
experiment name in `prepare_split` and every subsequent training/comparison
command when retraining.

To train another classifier, change `--model random_forest` to `--model svm`,
`--model knn`, `--model logistic` or `--model decision_tree` in the individual
`prepare_split` command. Keep `--grouping source` and `--allow-no-test` for this
download, and use a different experiment directory. Both input variants then
use that selected classifier on the same frozen folds.

## Compare against the Shin 2021 inputs

Reuse this analyzed dataset with the separate
[paper-method study](PAPER_SHIN2021.md). It trains five inputs with SVM and
LightGBM on shared assignments; it does not reuse or overwrite these Random
Forest models. Use `--split-from experiments/asl_kaggle_rf_01` with source
grouping, three folds and seed 42 to retain that experiment's exact membership.

## Troubleshooting local training

| Message or situation | What to do |
|---|---|
| Missing Kaggle folder | Point `--source` at the extracted directory containing `asl_alphabet_train` and `asl_alphabet_test`; do not pass the ZIP or only the inner `A` folder. |
| Output project already exists | Resume the same Kaggle import with `--resume --exclude-overlapping-test`, or use the complete-run command. A completed dataset can be reused without another import. |
| Resume requires the same source files | Restore the unchanged source files/options, or import the changed download into a new dataset directory. |
| `0 new images analyzed` | No photos are pending. Continue with splitting/training if the dataset is already analyzed. |
| Dataset/code changed since splitting | Choose a new experiment name and train both variants against a fresh split. |
| Experiment or variant output already exists | Reuse the saved results, or choose a new experiment name to retrain. |
| Fewer training examples than imported photos | Check detection counts in `reports/dataset_summary.json`; training requires one usable detected hand. |
| No separate final test | Expected for this download: all supplied test photos duplicate training photos. Collect independent test photos before claiming final-test accuracy. |
