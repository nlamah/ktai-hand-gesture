# Shin 2021 feature method applied to this project

Shin, J., Matsuoka, A., Hasan, M. A. M., & Srizon, A. Y. (2021).
*American Sign Language Alphabet Recognition by Extracting Feature from Hand
Pose Estimation.* Sensors, 21(17), 5856.
[DOI and article](https://doi.org/10.3390/s21175856),
[full PDF](https://mdpi-res.com/d_attachment/sensors/sensors-21-05856/article_deploy/sensors-21-05856.pdf).
The article is published under CC BY 4.0. The full method, formulas, Tables 2–7
and Figures 3–5 were inspected, including the PDF page images.

This is an **application of the feature method to your dataset and grouped
evaluation**, not a reproduction of the published scores. The article leaves
some details unspecified and contains inconsistencies; the choices below are
part of the implementation. No letters are automatically excluded, including
J and Z. A still image does not capture their complete motion.

## Implementation map

| Component | File |
|---|---|
| Pure, typed distance/axis feature functions | `gesturelab/features/paper_shin2021.py` |
| Registered inputs, column order, input-code hashes | `gesturelab/features/training_inputs.py` |
| Classifier construction and parameter profiles | `gesturelab/training/classifiers.py` |
| Grouped folds and verified split-template reuse | `gesturelab/training/splits.py` |
| Fold fitting, final fitting, timings and saved model | `gesturelab/training/fitting.py` |
| Prediction alignment and fold-membership checks | `gesturelab/training/predictions.py` |
| CSV, figures and HTML comparisons | `gesturelab/training/comparison.py` |
| Five-input, two-classifier study | `gesturelab/training/paper_study.py`, `scripts/run_shin2021.py` |
| Shared live input reconstruction | `model_features()` and `gesturelab/camera/runtime.py` |

The existing two training shortcuts and their default comparison remain
available. The new method does not modify stored images or landmark arrays.

## Coordinates and feature definitions

### Landmark source

Section 3.2.1 and Figure 3 use 21 MediaPipe hand points. The implementation
reuses each sample's **`image_landmarks`**, in the stored order 0–20. Point 0 is
the wrist; the paper numbers the same points 1–21. It does not substitute
`world_landmarks`, even if those are available.

The selected coordinates are the MediaPipe image-space x/y/z values as stored:
x is normalized by width, y by height, and z has approximately the scale of x.
See the [MediaPipe result description](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results).
The paper does not report a MediaPipe version, exact coordinate units or an
aspect-ratio conversion. Figure axes alone do not resolve this. Using current
image-space output is an explicit application choice. It is closer to the
described RGB-image joint estimator than replacing it with metric world points,
but is not evidence that the 2021 implementation used identical units.

There is no extra pixel/aspect correction, palm alignment, handedness canonicalization
or mirroring in these inputs. That differs from the project's existing
126-value baseline, which normalizes both image and world coordinates, and its
207-value augmented input. The paper variants therefore differ in coordinate
representation as well as feature engineering. Conclusions should describe the
complete input method, not claim an isolated causal benefit from adding values.

### Distances: Equation 1 and Table 2

For each selected pair `(i, j)`, compute the Euclidean XYZ distance:

```text
d(i, j) = sqrt((x[j]-x[i])² + (y[j]-y[i])² + (z[j]-z[i])²)
```

The implementation follows the **literal Table 2 pairs** in increasing starting
point, then increasing ending point. There are 190 pairs. `raw_distances()`
returns the unnormalized values for checking; models receive `distance_features()`.
Names use zero-based indices, for example `shin_dist_00_02` corresponds to the
paper's pair `(1, 3)`.

Table 2 omits these 20 pairs, expressed in the paper's one-based numbering:

```text
(1,2), (1,5), (1,9), (1,13), (1,17),
(2,3), (3,4), (4,5), (6,7), (7,8), (8,9),
(10,11), (11,12), (12,13), (14,15), (15,16), (16,17),
(18,19), (19,20), (20,21)
```

This is **not the same as removing anatomical bone connections** from Figure 3.
For example, Table 2 excludes `(1,5)` but includes `(1,6)`. We retain the explicit
table rather than silently correcting it or reusing the skeleton-drawing edges.
Tests verify every row count: 15, 18, 17, 16, 16, 14, 13, 12, 12, 10, 9, 8, 8,
6, 5, 4, 4, 2, 1, 0, 0.

### Distance normalization: Equation 2

```text
normalized_distance = (distance - mean_of_this_photo's_190_distances)
                      / population_std_of_this_photo's_190_distances
```

The paper specifies a z-score but **does not specify its axis or standard-deviation
convention**. We choose per-photo normalization across its 190 distances with
`ddof=0`, matching the stated intent to remove a hand's uniform size factor.
This is an assumption, not a verified reproduction detail. It learns nothing
from other photos and is identical at training and prediction time.

A zero-variance distance vector is rejected. Raw distances are translation and
rotation invariant in the selected numeric XYZ space; normalized distances also
remove positive uniform scale. These properties do not establish invariance to
arbitrary camera viewpoint or anisotropic changes in normalized image axes.

### Axis features: Table 3 and Equations 3–6

All 210 pairs `i < j`, including adjacent points, are used. Each vector points
**from i to j**: `v = point[j] - point[i]`. Its three features are:

```text
cos_x = v.x / ||v||
cos_y = v.y / ||v||
cos_z = v.z / ||v||
```

These are the **direction cosines printed in Equations 4–6**, not joint angles
between three points. The article calls them angles but does not clearly state
whether inverse cosine was applied in code. We keep the printed cosine values,
including their signs; no inverse cosine, degree conversion or absolute value
is silently added. Names such as `shin_angle_00_01_cos_x` make that choice explicit.

The order is pair-major, x then y then z: 210 × 3 = **630 values**. No additional
normalization is applied. Translation and positive uniform scale leave these
values unchanged; rotation generally changes them, which preserves tilt information.

### Available input variants

| Registry name | Values | Exact composition |
|---|---:|---|
| `without_features` | 126 | Existing normalized image and world coordinates |
| `with_features` | 207 | Existing coordinates plus 81 finger features |
| `shin2021_distance` | 190 | Table 2 distances, per-photo z-score |
| `shin2021_angle` | 630 | Table 3 directed-pair axis cosines |
| `shin2021_combined` | 820 | All 190 normalized distances, then all 630 cosines |

All feature functions require finite 21×3 arrays and do not modify their input.
An axis vector of length at most `1e-12` is undefined and rejected, rather than
assigned a fabricated angle. Distance standard deviations at most `1e-12` are
also rejected. Training stops with the variant and photo ID if an input cannot
be constructed; it never silently drops different photos for different variants.
Review such a photo, correct the dataset consistently and create a new experiment.
The live camera clears its prediction and displays unusable landmarks for an
invalid feature input, then continues processing later frames.

## Classifiers, paper settings and controlled comparison

Two explicit profiles are supported by `--classifier-profile`:

- **`project` (default):** shared classifier hyperparameters across all five
  inputs. SVM keeps the project's RBF settings `C=10`, `gamma=scale` and balanced
  class weighting. LightGBM uses 31 leaves, learning rate 0.1, minimum child
  samples 20 and 100 estimators. These LightGBM choices are project defaults,
  not claimed article settings.
- **`shin2021_asl`:** fixed published ASL Alphabet parameters from Tables 5 and 7,
  shown below. It performs no search on your dataset. Because settings differ
  by featureset, its score changes cannot be attributed solely to features.

| Paper features | SVM C | SVM gamma | LightGBM leaves | Learning rate | Min. child samples | Estimators |
|---|---:|---|---:|---:|---:|---:|
| Distance | 1000 | 0.001 | 100 | 0.1 | 25 | 12 |
| Axis cosines | 1000 | 0.001 | 100 | 0.1 | 25 | 13 |
| Combined | 1000 | scale | 100 | 0.1 | 25 | 14 |

The paper has no corresponding preset for this project's existing 126/207-value
inputs. Under `shin2021_asl`, these use the combined row as an explicit fallback.
The paper does not specify class weights, seed, thread count or all LightGBM
options. The paper profile uses no class weighting; this project sets the
experiment seed, CPU execution, one LightGBM thread, deterministic mode and
column-wise training. Remaining library defaults are not claimed paper settings.
The exact library version and resolved parameters are recorded with the run.

The paper reports an RBF SVM, a 20% test split and five-fold CV for tuning.
Table 4 searches C in `{0.1, 1, 10, 100, 1000}` and gamma in
`{scale, 0.001, 0.0001}`. Table 6 searches leaves in
`{5, 10, 25, 50, 75, 100, 500, 1000}`, learning rates in
`{0.1, 0.01, 0.001, 0.0001, 0.00001}`, minimum child samples in
`{5, 10, 25, 50, 100, 500, 1000}` and estimators in `10–250` (step unspecified).
Table 5's Massey distance gamma 0.01 is outside Table 4's printed search space.
The optional preset here uses only the ASL Alphabet rows, not that inconsistent row.

Section 3.1.1/Table 9 print 780,000 ASL Alphabet images, although 26 classes ×
3,000 images gives 78,000. This project uses the actual imported file counts;
it does not reinterpret that inconsistency as an extra source of training data.

No hyperparameter search is implemented here. The original dataset's grouped
folds and reserved test roles take precedence over the paper's ungrouped 80/20
protocol. Any future tuning must use inner grouped validation inside each outer
training fold. The reserved test set must not guide feature or parameter choices.

The existing inputs use `StandardScaler` inside the SVM training pipeline, fitted
only on the training rows of each fold. Paper inputs already define their own
normalization; SVM receives them without a second scaler. LightGBM has no scaler.
The fitted pipeline, exact feature-column order, input definition, parameters and
input-code hashes are saved in `model.joblib`. Training and webcam inference both
call the same registered extractor. No image/world selection occurs in the camera.

## Run a complete development comparison

Run commands from the project root. Use an already imported, analyzed dataset;
see [WORKFLOW.md](WORKFLOW.md#alternative-train-locally-with-external-asl-data)
for external-data import. Install/update dependencies:

```bash
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

On macOS, LightGBM also needs the OpenMP runtime. In the Conda environment:

```bash
conda install -n ktai-hand-gesture -c conda-forge llvm-openmp
python -c "import lightgbm; print(lightgbm.__version__)"
```

See [LightGBM installation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
for other platforms. For your own recording sessions with a reserved test set:

```bash
python -m scripts.run_shin2021 --project data/ngt --out experiments/shin2021_own_01 --grouping session --folds 3
```

This includes every selected dataset label; it does not filter J or Z. Use an
ASL recording dataset in place of `data/ngt` for an ASL claim. Running NGT photos
through these features evaluates the method on NGT, not ASL.

For participant grouping, replace `--grouping session` with `--grouping participant`;
each label must have successful development detections from at least three people
for three folds, plus an independent reserved test person. Do not invent participant
identities for external datasets with unknown recording metadata.

For the imported Kaggle data, use source blocks and explicitly allow no final test:

```bash
python -m scripts.run_shin2021 --project data/asl_kaggle --out experiments/shin2021_kaggle_01 --grouping source --folds 3 --allow-no-test
```

The supplied Kaggle test photos duplicate training photos. Source blocks remain
filename groups, not verified people/sessions. Neither these CV scores nor the
paper's scores establish independent live-camera accuracy on your photos.

For the imported COCO export, acknowledge missing E/L test examples:

```bash
python -m scripts.run_shin2021 --project data/asl --out experiments/shin2021_coco_01 --grouping source --folds 3 --allow-incomplete-test
```

The command creates `svm/` and `lightgbm/` experiments, trains all five inputs and
writes `report.html`, `metrics.csv` and `shared_split.json` in the study directory.
It verifies identical photo IDs, labels and fold assignments across classifiers.
It does **not** evaluate the reserved test. Use a new output name for another run.
There is no automatic subsampling. RBF SVM can require substantial time and
memory on large datasets; `--models lightgbm` runs just that classifier with all
five inputs if you want to run the classifiers separately.
For a separate published-preset study, add `--classifier-profile shin2021_asl`
and choose a different `--out`.

### Keep the exact assignments of an existing experiment

Code changes invalidate previous experiment hashes. Existing datasets, model
files and reports are preserved; create a new experiment for this implementation.
To retain the exact old membership, add a verified assignment template:

```bash
python -m scripts.run_shin2021 --project data/asl_kaggle --out experiments/shin2021_kaggle_02 --grouping source --folds 3 --seed 42 --allow-no-test --split-from experiments/asl_kaggle_rf_01
```

This requires an identical dataset fingerprint, selected labels, development/test
IDs, grouping, fold count and seed. The template must validate each development
photo once, preserve label coverage and keep groups disjoint. Only membership is
copied into **new** plans with current code snapshots and a recorded template hash.
Old models are not loaded or continued, and their code-hash guards remain in force.
If data or labels changed, omit the template and make a fresh shared split.

### Split, train and compare individual stages

For example, prepare a fresh SVM experiment on the analyzed Kaggle dataset:

```bash
python -m scripts.prepare_split --project data/asl_kaggle --out experiments/shin2021_svm_01 --grouping source --folds 3 --model svm --classifier-profile project --allow-no-test
python -m scripts.train_without_features --experiment experiments/shin2021_svm_01
python -m scripts.train_with_features --experiment experiments/shin2021_svm_01
python -m scripts.train_model --experiment experiments/shin2021_svm_01 --variant shin2021_distance
python -m scripts.train_model --experiment experiments/shin2021_svm_01 --variant shin2021_angle
python -m scripts.train_model --experiment experiments/shin2021_svm_01 --variant shin2021_combined
python -m scripts.compare_results --experiment experiments/shin2021_svm_01 --variants without_features with_features shin2021_distance shin2021_angle shin2021_combined
```

Use `--model lightgbm` and a new experiment directory for LightGBM. Add
`--split-from experiments/shin2021_svm_01` to that preparation command to reuse
the exact SVM assignments; keep grouping, folds and seed identical. Choose
`--classifier-profile shin2021_asl` only for a separate preset experiment.

## Inspect results and evaluate the final test separately

Each classifier's comparison directory contains:

| File | Contents |
|---|---|
| `metrics.csv` | Pooled accuracy/macro-F1, CV means/spread, feature counts and timings |
| `cv_folds.csv` | Per-fold accuracy, macro-F1, balanced accuracy, counts, fit time and prediction timing for every input |
| `<variant>_per_label.csv` | Precision, recall, F1 and support for each letter |
| `<variant>_confusion.csv`, `.png` | Confusion counts and row-normalized figure |
| `<variant>_predictions.csv` | Predictions with photo ID and true label |
| `paired_predictions.csv` | Aligned per-photo predictions/correctness across all inputs |
| `pairwise_comparisons.csv` | Score differences and paired error counts for every pair |
| `detection_counts.csv` | Detection status counts per role and letter |
| `comparison.png`, `recall_per_letter.png`, `report.html` | Readable comparison and figures |

Each trained variant also retains its input matrix, `cv_predictions.csv`,
`cv_folds.csv`, `training.json` and `model.joblib`. Feature extraction time, CV fit
time, final fit time and total model-training time are measured separately.
Charts show development fold means and standard deviations, not confidence
intervals; the main table also exposes pooled held-out predictions. Neither a
small difference nor these error bars proves significance.

Development scores include successful detections only. Detection failures are
reported rather than silently disappearing. Final-test evaluation retains every
reserved test photo, predicting unknown for failed detections and counting these
as errors. An undefined feature extraction stops evaluation rather than discarding
that photo. Missing test-letter recall/F1 stays undefined. If no independent
test exists, `--final-test` is rejected even when `--allow-no-test` was used.

After feature/classifier choices are fixed, explicitly evaluate a reserved set,
for example in the COCO study:

```bash
python -m scripts.compare_results --experiment experiments/shin2021_coco_01/svm --final-test --variants without_features with_features shin2021_distance shin2021_angle shin2021_combined
```

The output uses a separate `final_test_...` directory. Repeating an identical
evaluation reuses it. Do not select the best method by testing many alternatives
on reserved data; make choices on development CV and report the final result
only afterward.

## Live webcam test

```bash
python -m apps.webcam predict --project data/asl_kaggle --model experiments/shin2021_kaggle_01/svm/shin2021_combined/model.joblib
```

Replace `svm` with `lightgbm`, or the input name with `shin2021_distance` /
`shin2021_angle`, to load another trained model. Existing 126/207-value models
trained under the current code use the same webcam path. Models whose input-code
hashes predate this update need retraining in a new experiment; no checks are
disabled for compatibility. Raw/stabilized prediction, mirroring and keyboard
controls retain their existing behavior.

## Executed checks and limitations

```bash
python -m unittest discover -s tests -v
python tests/smoke_shin2021.py --out examples/shin2021_software_test
python tests/smoke_modular.py --out examples/original_workflow_check
```

Use unused output directories. The full unit suite passed with **105 tests**.
New tests cover known Euclidean distances and signed axis cosines, the literal
Table 2 rows, feature counts/order, no input mutation, finite/shape validation,
zero variance/vectors, translation/scale invariance and rotation sensitivity.
They also test training-only fitted scaling, retained J/Z labels, participant
separation, LightGBM presets/version recording, exact split-template reuse and
rejection of changed data/folds, and real paper-model predictions through a fake
camera session. Hardware is not needed for these checks.

The complete paper smoke run trained **all ten combinations** (five inputs ×
SVM/LightGBM), compared their shared CV assignments, checked the saved input
columns against inference, generated CSV/PNG/HTML reports, and exercised a
separate, reusable final evaluation on **synthetic** data. One development and
one test detection failure were simulated and verified. The existing workflow
smoke run also passed.

These synthetic scores are software checks, not measured ASL recognition rates.
No new large real-data benchmark or physical webcam session was run as part of
this implementation. A proper comparison requires your analyzed ASL dataset,
its grouped development data and an independent test set where available.
