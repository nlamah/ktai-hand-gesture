# Train on the downloaded ASL COCO dataset

For the Kaggle folder-based dataset, use [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).
The [external-data workflow](WORKFLOW.md#alternative-train-locally-with-external-asl-data)
compares both routes and explains source, dataset and experiment directories.

Use this route after downloading the dataset described in
[ASL_DOWNLOAD.md](ASL_DOWNLOAD.md). Run commands from the project root in the
recognition environment, not the optional download-only environment:

```bash
conda activate ktai-hand-gesture
```

You only need the recognition environment and the extracted files for local
training. The Roboflow SDK and API key belong to the optional download step.
The expected source layout is:

```text
data/external/asl_letters_v1/
├── train/
│   ├── _annotations.coco.json
│   └── ... image files named in the annotations
├── valid/
│   ├── _annotations.coco.json
│   └── ... image files named in the annotations
└── test/
    ├── _annotations.coco.json
    └── ... image files named in the annotations
```

If you already completed the import into `data/asl`, start at landmark analysis
or reuse its analyzed landmarks as described under [retraining](#reuse-the-dataset-and-train-another-experiment).
Keep downloaded ASL data separate from the NGT recording dataset.

## 1. Import images and labels

```bash
python -m scripts.import_coco_dataset --source data/external/asl_letters_v1 --project data/asl
```

The downloaded version contains 1,512 train images, 144 validation images and
72 test images, with letter labels A–Z. The importer maps COCO category IDs to
letters, saves original images and records source annotations and bounding boxes.
It uses the complete images for MediaPipe analysis; bounding boxes are retained
as provenance rather than silently changing the camera input by cropping it.

`data/asl` must be a new directory. The raw download remains unchanged.
The imported dataset does not receive the default NGT reference drawings.
Rerunning the importer on an existing output is refused; use the existing
imported dataset or choose a different output directory.
An interrupted COCO import cannot be resumed by this importer; choose a new
dataset directory for a complete retry. Do not treat a partially imported dataset
as the complete download.

## 2. Calculate hand landmarks

```bash
python -m scripts.analyze_dataset --project data/asl --checkpoint-every 1000 --no-cache-features
```

MediaPipe supplies the image and world landmarks required by the feature
calculations. COCO bounding boxes are not used as substitutes for landmarks.
Photos without a usable hand detection remain in the dataset. Development
training uses successful detections; final-test detection failures count as
errors. Review counts rather than assuming every imported image was detected.
These options save progress in batches and store landmarks without redundant
feature caches. Training calculates both input variants from those landmarks.
The first analysis downloads the MediaPipe model if it is missing. Rerunning
analysis processes only pending photos; `0 new images analyzed` is expected when
everything was already analyzed. It does not mean the dataset is empty; check
the dataset counts printed by the command.

## 3. Freeze a shared Random Forest experiment

```bash
python -m scripts.prepare_split --project data/asl --out experiments/asl_rf_01 --folds 3 --grouping source --model random_forest --allow-incomplete-test
```

Version 1 contains no test images for **E and L**, although both are present in
development data. `--allow-incomplete-test` explicitly acknowledges that gap;
training still includes all 26 letters. Final-test recall/F1 for missing letters
is undefined, and aggregate macro scores use only letters present in the test.
The report identifies the missing letters. Such a test cannot establish
recognition quality for E and L. The default split command rejects incomplete
test coverage when this flag is omitted.

The supplied dataset has no reliable participant or recording-session metadata.
The importer therefore records the participant as `unknown` and retains the
provided partition in `provenance.dataset_split`. It does not invent independent
participants or sessions.

For this experiment:

- The original **train and valid partitions become development data** for grouped
  cross-validation. The original valid partition is not a separate untouched test.
- The original **test partition remains the reserved final test**.
- `--grouping source` keeps augmented versions of the same original image together.
  For this Roboflow export, the filename before `.rf.` identifies the source group.
- The 1,512 training images represent 504 source images, with three versions each.
  Along with validation and test data, there are 720 source-image groups.
- Import rejects source groups or identical pixels that cross development/test
  roles, and it rejects ambiguous multi-letter image annotations.

This evaluates generalization to held-out source images, not necessarily to new
people or independent recording sessions. Different original photos can still
be correlated; missing participant/session metadata limits conclusions.
Do not use `--grouping participant` to claim person-independent evaluation here.

If the experiment directory exists, choose another name. Freeze a new experiment
after modifying data, feature code or model settings.

## 4. Train both input variants

```bash
python -m scripts.train_without_features --experiment experiments/asl_rf_01
python -m scripts.train_with_features --experiment experiments/asl_rf_01
```

Both use the same classifier, source groups and folds:

| Variant | Inputs | Values |
|---|---|---:|
| `without_features` | Normalized image and world landmarks | 126 |
| `with_features` | Same landmarks plus derived finger features | 207 |

The final models are saved in:

```text
experiments/asl_rf_01/without_features/model.joblib
experiments/asl_rf_01/with_features/model.joblib
```

These commands perform development cross-validation and final fitting on all
development images; they do not evaluate the reserved test images.

## 5. Inspect development results

```bash
python -m scripts.compare_results --experiment experiments/asl_rf_01
```

Open `experiments/asl_rf_01/comparison_cv/report.html`. Use these results to make
feature/model choices. Do not present them as final-test results.

After all choices are fixed, explicitly evaluate the reserved test set:

```bash
python -m scripts.compare_results --experiment experiments/asl_rf_01 --final-test
```

Open `experiments/asl_rf_01/final_test/report.html`. Do not tune using this score.

## 6. Try an ASL model with your webcam

```bash
python -m apps.webcam predict --project data/asl --model experiments/asl_rf_01/with_features/model.joblib
```

Replace `with_features` with `without_features` for the other model. Recognition
quality on a live camera may differ from the supplied images. This is a
still-pose classifier; a photograph does not represent a moving letter's full
execution. Labels are imported without automatically excluding letters.

For ASL pose guidance, open the [photographic alphabet chart](../assets/asl_alphabet.jpg)
from ASL University / Lifeprint.com. Author and usage information are recorded
in [assets/SOURCES.md](../assets/SOURCES.md).

The capture app's supplied reference images still show NGT. Do not use those
as ASL pose instructions. This import/training route is separate from the
NGT recording workflow in [WORKFLOW.md](WORKFLOW.md).

## Reuse the dataset and train another experiment

For the five-input SVM/LightGBM comparison based on the article, follow
[PAPER_SHIN2021.md](PAPER_SHIN2021.md). It reuses these stored landmarks and can
copy verified assignments with `--split-from` into a fresh experiment. No letters
are automatically excluded by that method.

Keep `data/asl` after a successful import and analysis. To train again, create a
new experiment instead of importing the photos again or overwriting a saved
model. For example, using the same analyzed data:

```bash
python -m scripts.prepare_split --project data/asl --out experiments/asl_rf_02 --folds 3 --grouping source --model random_forest --allow-incomplete-test
python -m scripts.train_without_features --experiment experiments/asl_rf_02
python -m scripts.train_with_features --experiment experiments/asl_rf_02
python -m scripts.compare_results --experiment experiments/asl_rf_02
```

Choose an unused experiment name and use it consistently in all four commands.
`--model` on `prepare_split` selects the classifier for both variants; the
individual training commands use that frozen choice. Supported values are
`random_forest`, `svm`, `knn`, `logistic` and `decision_tree`. To compare classifiers,
create a separate experiment per classifier with the same dataset, labels,
grouping, fold count and seed. Do not compare a score from different data splits
as if only the classifier had changed.

If you only want to use an existing model, skip training and run the webcam
command in step 6, or reopen its `comparison_cv/report.html`. Keep the imported
dataset and experiment directory: the raw download alone does not contain the
saved landmarks, model or reports.
