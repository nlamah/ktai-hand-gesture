# ASL alphabet: shared baseline and three research questions

This is the instructor's reproduction guide. The project recognizes **static A-Z images** using MediaPipe landmarks and a letter classifier. It does not model continuous signing or the movements of J and Z.

The numbering follows the supplied **group paper**:

| Component | Research | Implementation status |
|---|---|---|
| Shared baseline | Shin distances and directions: **DS, 820 inputs** | [baseline/](baseline/), available |
| RQ1 | Targeted additional training images | [RQ1/](RQ1/), methods/results supplied; executable code missing |
| RQ2 | Recovering hand detections using dilation, erosion and a Canny overlay | [RQ2 guide](RQ2/README.md), implemented; technical recovery and visual-review tools |
| RQ3 | Own hand-shape features and joint measurements, with/without Shin directions | [RQ3 guide](RQ3/README.md), implemented; includes brief practical explorations |

**The complete group study still requires the missing RQ1 code.** RQ2 is available through `python -m RQ2`; see its guide for raw-image inputs, running, verification and reporting. The shared baseline, RQ3 and supporting controls below are also available.

## 1. Install and check

Run from the repository root with Python 3.12 and the package constraints:

```bash
conda create -n ktai-hand-gesture python=3.12
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
python -m baseline check
python -m unittest discover -s tests -v
```

For an existing environment, activate it and install `requirements.txt`. If LightGBM cannot find `libomp` on macOS, use `conda install -c conda-forge llvm-openmp`.

`baseline check` reports dependencies, required files, local research inputs and contribution status. Missing research inputs are expected before restoring the bundles below. Unit tests run without `results/`, raw images or the input bundle. The dependency versions recorded by the frozen RQ3 protocols are pinned in `requirements.txt`.

Expected shared baseline on **60,251 accepted original single-hand detections**:

| Model | Macro-precision | Macro-F1 | Correct |
|---|---:|---:|---:|
| SVM, DS | 97.38815% | 0.97386235 | 58,827 |
| LightGBM, DS | 96.80215% | 0.96784341 | 58,502 |

These are our fixed model profiles using Shin features, not an exact replication of Shin's published scores. **DSJ has 835 inputs** and adds world-joint measurements. Its approximately 97.54% precision belongs to a separate reference.

## 2. Project structure and shared code

```text
baseline/         Shared Shin interface, classifier profiles, checks and input export
RQ1/              Integration slot for targeted data collection
RQ2/              Public interface and guide for image preprocessing and detection recovery
RQ3/              Public experiment workflows and the RQ3 guide
research/         Shared processing, live tools and fixed numerical implementations
research/studies/ Numerical implementations called by the RQ3 workflows
tests/            Tests for the active software
models/           Trusted trained models and the MediaPipe model file
assets/, config/  Workbook, examples and fixed measurement rules
results/          Local predictions, metrics and protocols; excluded from Git
data/, runs/      Local inputs and training checkpoints; excluded from Git
captures/         Local webcam captures; excluded from Git
```

`baseline.shin` imports the unchanged functions from `research/shin.py`. Source hashes and saved model classes require stable module paths. The mirror and G/H experiments also require a specific `research/live_model.py` implementation. That dependency is supplied in `baseline/frozen_sources/`; the RQ3 entry points use it in a temporary workspace. The current live app uses its own implementation. A complete supplied mirror cache is reused when available; otherwise mirror extraction needs the raw images.

Example for contributions:

```python
from baseline.shin import combined_features
from baseline.models import build_classifier
from research.data import Detector

# Use only detections with status == "ok".
features = combined_features(detection["image_landmarks"])  # 820 values
classifier = build_classifier("svm", training_images=len(X_train))
```

`baseline.data.load_shin_inputs(cache)` returns DS columns, labels, source IDs, groups and saved folds. Compare the same sources/folds and augment only training sources. Keep a source and all its derivatives in the same fold.

## 3. Inputs for the paper results

Git includes code, assets and deployment models. **The entire `results/` folder is ignored and is not included in a new checkout.** Raw Kaggle images, frozen landmark inputs, fold models and private captures are also excluded.

For the paper's RQ3 reproduction, transfer these two local files separately from Git:

- `rq3_inputs.zip`: the frozen manifest, landmark arrays, source IDs, five folds and checksums. It excludes raw images, mirror detections, private captures and trained fold models.
- `rq3_results.zip`: the reported RQ3 CSV/JSON tables, protocols and reference predictions, with checksums. It excludes photographs, captures, trained models and run checkpoints.

These files are created under `runs/reproduction/` on the research computer:

```bash
python -m baseline bundle --output runs/reproduction/rq3_inputs.zip
python -m baseline bundle-results --output runs/reproduction/rq3_results.zip
```

Use new output filenames when exporting again. Copy both archives outside the checkout before moving to another location. From the new repository root, restore and check them:

```bash
python -m zipfile -e /path/to/rq3_inputs.zip .
python -m zipfile -e /path/to/rq3_results.zip .
python -m baseline verify-data
python -m baseline verify-results
python -m baseline check
python -m baseline verify
```

`baseline verify-results` checks the restored reference archive before retraining. The RQ3 workflow verifiers check regenerated results. `baseline verify` recomputes both DS scores from the restored out-of-fold predictions; it does not fit models. Actual retraining follows in Sections 4-5. The result archive also supplies the frozen protocols/reference predictions required by the supplementary workflows. Recorded protocols reject changed inputs/settings; do not edit them to bypass a mismatch.

The main eight-way feature comparison and coordinate control need no raw photographs. Mirror extraction and the photographic occlusion control require the raw images described below. Replaying the four selected live cases requires their private capture folders separately; neither archive includes them.

A mirror cache can optionally be added to the input archive with `--with-mirrors` after all 78 chunks exist under `runs/mirror_01/detections/`. That cache is currently not part of `rq3_inputs.zip`. Redetecting landmarks is a separate run, not a substitute for the frozen classifier inputs.

### Raw images for redetection or photographic figures

**Skip this section for the main RQ3 feature comparison and coordinate control:** the frozen input bundle already supplies their landmarks. Raw photographs are needed for RQ2, mirror extraction without a supplied mirror cache, photographic controls or new detections.

The original photographs are **not committed**: `data/` is ignored by Git, and neither reproduction archive includes the photographs. Download [ASL Alphabet by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), save the downloaded ZIP as `asl-alphabet.zip` in your Downloads folder, and run from the repository root:

```bash
mkdir -p data/asl-alphabet-download
python -m zipfile -e ~/Downloads/asl-alphabet.zip data/asl-alphabet-download
```

The resulting source folder must contain `asl_alphabet_train/` and `asl_alphabet_test/`. For example, a training photograph is at `data/asl-alphabet-download/asl_alphabet_train/A/A1.jpg`; the extra nesting `asl_alphabet_train/asl_alphabet_train/A/A1.jpg` is also supported. This source path is also the default used by RQ2.

If raw images are needed alongside the frozen inputs, **import the images first, then extract the input bundle**:

```bash
python -m research.dataset import --source data/asl-alphabet-download --project data/asl_kaggle
python -m zipfile -e /path/to/rq3_inputs.zip .
```

Only the 78,000 A-Z training images are in scope; `space`, `del` and `nothing` are excluded. Supplied test images that duplicate training images are not an independent test. The original inventory contains 60,251 accepted single hands, 17,598 `no_hand` results and 151 multiple-hand detections.

## 4. Retrain the shared baseline

There are two routes. Use **4.1** to rerun classification on the exact saved paper landmarks. Use **4.2** to check the complete baseline, starting with the original photographs and estimating landmarks again.

### 4.1 Classification from the frozen paper landmarks

After restoring both archives as described in Section 3:

```bash
python -m baseline train --dry-run
python -m baseline train
```

This trains DS only, using SVM and LightGBM across the five fixed folds. Matching RQ3 baseline checkpoints are reused. `python -m baseline verify` checks the published DS scores; the full RQ3 export also checks checkpoint predictions.

### 4.2 Complete baseline from the original photographs

First complete installation in Section 1, then download and extract the photographs into `data/asl-alphabet-download/` as described under **Raw images for redetection or photographic figures** in Section 3. Run all commands below from the repository root. This route needs no frozen input or result archives.

Use separate paths to avoid replacing the paper inputs. Execute these five steps in order; detection and model training can take substantial time:

```bash
# 1. Import the original A-Z photographs and their labels.
python -m research.dataset import --source data/asl-alphabet-download --project runs/new_detection
# 2. Estimate landmarks with the baseline detector settings.
python -m research.dataset analyze --project runs/new_detection --detection-confidence 0.5 --presence-confidence 0.5
# 3. Inspect acceptance counts and rejected photographs before training.
python -m research.detection_report --project runs/new_detection --out runs/new_detection_report
# 4. Build feature rows and five source-group folds from accepted detections.
python -m research.prepare --project runs/new_detection --out runs/new_cache
# 5. Train and evaluate the DS baseline with SVM and LightGBM.
python -m baseline train --cache runs/new_cache --out runs/new_baseline
```

Step 3 prints overall detection counts and writes `runs/new_detection_report/by_letter.csv` and `rejected_images.csv`. The historical counts for comparison are 78,000 A-Z training images, 60,251 accepted detections, 17,598 `no_hand` and 151 multiple hands. Wait until `pending` is zero before preparing features.

Step 5 fits ten models (two classifiers across five folds). It writes held-out predictions to `runs/new_baseline/predictions.csv`, scores to `runs/new_baseline/metrics.csv`, plus models and a protocol. Compare the rows with `fold=pooled` to the baseline table in Section 1; precision and F1 in the CSV are fractions (multiply by 100 for percentages). `baseline verify` checks restored historical predictions, not this fresh run.

This is a new method replication with its own hashes, counts, folds, predictions and scores; exact historical counts and scores are not assumed. The output protocol rejects reuse with different inputs. Import supports `--resume`; detection resumes unfinished images by default. `research.prepare` requires a new output directory.

## 5. Reproduce RQ3

[RQ3/README.md](RQ3/README.md) explains the question, feature blocks, findings and supporting experiments.

```bash
python -m RQ3 list
python -m RQ3 features --dry-run
python -m RQ3 features
```

`features` runs input checks, training, permutation tests, export, independent verification and reporting. It fits **80 fold models**: eight representations x two learners x five folds. D is always present; S, E and J are switched on/off: DS, D, DSE, DE, DSJ, DJ, DSEJ and DEJ. The full representation has 905 inputs.

Classification uses 60,251 accepted images. Permutation uses 13,000 fixed held-out images and 3,900 group interventions, without refitting. The optimized SVM route checks parity with ordinary classifier predictions. Matching checkpoints under `runs/feature_factorial_01/` are resumable; the complete run is computationally expensive.

Outputs in `results/feature_factorial_01/` include `metrics.csv`, `per_letter.csv`, `predictions.csv`, `paired_changes.csv`, `permutation*.csv`, protocols and verification. The verifier checks hashes, shared splits, training-only preprocessing, reference parity and saved fold-model predictions. To verify an existing run:

```bash
python -m RQ3 features --action verify
```

Model replay requires the inputs and fold models; published CSVs alone are insufficient.

### Coordinate control and brief practical explorations

The coordinate control supports the main feature comparison. Left/right and G/H attempts are short exploratory follow-ups motivated by camera use, not extra main research questions. Their detailed commands are retained for reproduction. Run dependencies in the order below; each supports `--dry-run` and `--action verify`.

| Command | Inputs and dependencies |
|---|---|
| `python -m RQ3 coordinate_control` | Joint measurements from Shin's image points; frozen cache and manifest |
| `python -m RQ3 mirrors` | Raw photographs, or a separately supplied complete mirror cache; restores the matched reference and mirror models |
| `python -m RQ3 normalization` | Detector-Right normalization; run `mirrors` first |
| `python -m RQ3 gh_rotation` | Separate G/H rotation training; frozen cache and manifest |
| `python -m RQ3 occlusion` | Coarse rendering, point noise and artificial rotation testing; run `gh_rotation` first; raw images needed for the 64-photo detection control |
| `python -m RQ3 live_examples` | Replay four specific local captures, which must be supplied separately |

The orientation comparisons retain their **835-input DSJ reference**. Primary classification scores exclude rejected detections. The artificial mirror mixture is a different test presentation. A high artificial G/H rotation score does not establish better live recognition. Four selected webcam examples do not estimate precision/F1 or error frequency.

The 22 withdrawn guided G/H captures are not current training/test data. Private captures are excluded from Git and the input bundle; drawing their figures again requires those local files. Figure assets are in the separate paper project; measurement provenance is in `results/`.

## 6. Live recognition and feature inspection

Use the supplied deployment models without retraining:

```bash
python -m research.live_camera
```

Choose **1** for DS, **2** for DSJ, **9** for G/H rotation, **7/8** for the matched mirror reference/mirror-trained models and **0** for right normalization. The default is DSJ. **Space -> intended letter -> Enter** saves a capture under `captures/confusions/`. **L** toggles landmarks, **N** starts a round and **Q/Esc** exits. Saved records include the model hash, exact detector input, landmarks and prediction. Free captures are error analysis, not automatic training data.

For the browser feature tool:

```bash
python -m streamlit run app.py
```

Select a model, enable **Camera on**, optionally enter the intended letter and use **Save capture**. The intended letter does not affect recognition. Only one tool can use the camera at a time; macOS may require camera permission. See the [live test guide](docs/LIVE_TESTPLAN.md).

## 7. Separate paper project

The paper is maintained in a separate Overleaf repository, not in a `paper/` directory in this software checkout. This computer's paper checkout is `~/Documents/GitHub/6aba6ebfe26cf023a6c2efc0`; compile its `main.tex`. Obtain that source project separately when reproducing the complete submission.

Published classification evidence is restored from `rq3_results.zip` under `results/feature_factorial_01/`. Results, frozen inputs, raw images and private captures are distributed separately as described above. Executable RQ1 code and its raw experiment inputs still need to be supplied. RQ2 is implemented; its historical run outputs are distributed separately from Git. See each contribution guide for its exact scope and required inputs.

## 8. RQ1/RQ2 contributions

RQ1 still requires `RQ1/experiment.py` with a `main()` function. RQ2 is available: use `python -m RQ2 list` and the [RQ2 guide](RQ2/README.md) for reproduction, verification and reporting. `python -m RQ1 --status` and `python -m RQ2 --status` show the current status.

Reuse the shared detector/baseline where relevant. RQ1 outputs belong under `results/RQ1/`; RQ2 retains its existing `experiments/hand_detection_recovery/runs/<name>/` layout. Include source IDs, train/test splits, settings, software versions, predictions or detection statuses and a reproduction command. RQ2 must distinguish technical recovery from visually plausible landmarks. State any different dataset selection or baseline explicitly.

## 9. Documentation

- [RQ3 guide](RQ3/README.md): question, variants, findings and limitations.
- [Live test guide](docs/LIVE_TESTPLAN.md): free comparisons and optional collection protocols.
