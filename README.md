# ASL alphabet: shared baseline and three research questions

This is the instructor's reproduction guide. The project recognizes **static A–Z images** using MediaPipe landmarks and a letter classifier. It does not model continuous signing or the movements of J and Z.

The numbering follows the supplied **group paper**:

| Component | Research | Implementation status |
|---|---|---|
| Shared baseline | Shin distances and directions: **DS, 820 inputs** | [baseline/](baseline/), available |
| RQ1 | Targeted versus general data augmentation | [RQ1/](RQ1/), contribution not supplied |
| RQ2 | Recovering hand detections using dilation, erosion and a Canny overlay | [RQ2/](RQ2/), paper text supplied; executable code missing |
| RQ3 | Own hand-shape features and joint measurements, with/without Shin directions | [RQ3 guide](RQ3/README.md), implemented; includes brief practical explorations |

**The complete group study requires the missing RQ1/RQ2 code.** The shared baseline, RQ3 and supporting controls below are available. The supplied RQ2 paper results are not an executable implementation in this repository.

## 1. Install and check

Run from the repository root with Python 3.12 and the pinned package versions:

```bash
conda create -n ktai-hand-gesture python=3.12
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
python -m baseline check
python -m unittest discover -s tests -v
python -m baseline verify
```

For an existing environment, activate it and install `requirements.txt`. If LightGBM cannot find `libomp` on macOS, use `conda install -c conda-forge llvm-openmp`.

`baseline check` reports dependencies, required files, local research inputs and contribution status. `baseline verify` independently recomputes the DS scores from published out-of-fold predictions, without a camera, raw photographs or retraining.

Expected shared baseline on **60,251 accepted original single-hand detections**:

| Model | Macro-precision | Macro-F1 | Correct |
|---|---:|---:|---:|
| SVM, DS | 97.38815% | 0.97386235 | 58,827 |
| LightGBM, DS | 96.80215% | 0.96784341 | 58,502 |

These are our fixed model profiles using Shin features, not an exact replication of Shin's published scores. **DSJ has 835 inputs** and adds world-joint measurements. Its approximately 97.54% precision belongs to a separate reference.

## 2. Project structure and shared code

```text
baseline/         Shared Shin interface, classifier profiles, checks and input export
RQ1/              Integration slot for targeted/general augmentation
RQ2/              Integration slot for image preprocessing and detection recovery
RQ3/              Public experiment workflows and the RQ3 guide
research/         Shared processing, live tools and fixed numerical implementations
research/studies/ Numerical implementations called by the RQ3 workflows
tests/            Tests for the active software
models/           Trusted trained models and the MediaPipe model file
assets/, config/  Workbook, examples and fixed measurement rules
results/          Predictions, metrics, protocols and verification
paper/            LaTeX sources, tables, figures and PDFs
data/, runs/      Local inputs and training checkpoints; excluded from Git
captures/         Local webcam captures; excluded from Git
```

`baseline.shin` imports the unchanged functions from `research/shin.py`. Source hashes and saved model classes require stable module paths. The mirror and G/H experiments also require a specific `research/live_model.py` implementation. That dependency is supplied in `baseline/frozen_sources/`; the RQ3 entry points use it in a temporary workspace. The current live app uses its own implementation. A complete mirror cache is reused.

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

Git includes code, results and deployment models, but **not raw Kaggle images, the frozen landmark cache, fold models or private captures**. Exact classifier reproduction requires the frozen inputs. Redetection is a separate run: not all detector settings were retained in the original cache, so identical detection output is not assumed.

Supply `paper_inputs_20261108.zip` separately (approximately 689 MiB, created under `runs/reproduction/`). It contains the dataset manifest with estimated landmarks, `inputs.npz`, source IDs, five fixed folds, independently detected mirror inputs and SHA256 checksums. It excludes raw photographs, webcam captures, credentials and fold models.

Extract into a clean checkout from the repository root:

```bash
python -m zipfile -e /path/to/paper_inputs_20261108.zip .
python -m baseline verify-data
python -m baseline check
```

Recorded protocols check input hashes and reject changed settings. Do not alter a protocol to accept different inputs. To create a new input bundle on the research computer:

```bash
python -m baseline bundle --output runs/reproduction/paper_inputs_new.zip --with-mirrors
```

### Raw images for redetection or photographic figures

Download [ASL Alphabet by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The importer accepts both `asl_alphabet_train/A/A1.jpg` and the doubly nested layout; the source directory must contain the train and test directories.

If raw images are needed alongside the frozen inputs, **import the images first, then extract the input bundle**:

```bash
python -m research.dataset import --source /path/to/asl-alphabet --project data/asl_kaggle
python -m zipfile -e /path/to/paper_inputs_20261108.zip .
```

Only the 78,000 A–Z training images are in scope; `space`, `del` and `nothing` are excluded. Supplied test images that duplicate training images are not an independent test. The original inventory contains 60,251 accepted single hands, 17,598 `no_hand` results and 151 multiple-hand detections.

## 4. Retrain the shared baseline

With the **frozen paper inputs**:

```bash
python -m baseline train --dry-run
python -m baseline train
```

This trains DS only, using SVM and LightGBM across the five fixed folds. Matching RQ3 baseline checkpoints are reused. `python -m baseline verify` checks the published DS scores; the full RQ3 export also checks checkpoint predictions.

### Separate replication from fresh detections

Use separate paths to avoid replacing the paper inputs:

```bash
python -m research.dataset import --source /path/to/asl-alphabet --project runs/new_detection
python -m research.dataset analyze --project runs/new_detection --detection-confidence 0.5 --presence-confidence 0.5
python -m research.detection_report --project runs/new_detection --out runs/new_detection_report
python -m research.prepare --project runs/new_detection --out runs/new_cache
python -m baseline train --cache runs/new_cache --out runs/new_baseline
```

This is a new method replication with its own hashes, counts, folds, predictions and scores. The output protocol rejects reuse with different inputs. Import supports `--resume`; detection resumes unfinished images by default. `research.prepare` requires a new output directory.

## 5. Reproduce RQ3

[RQ3/README.md](RQ3/README.md) explains the question, feature blocks, findings and supporting experiments.

```bash
python -m RQ3 list
python -m RQ3 features --dry-run
python -m RQ3 features
```

`features` runs input checks, training, permutation tests, export, independent verification and reporting. It fits **80 fold models**: eight representations × two learners × five folds. D is always present; S, E and J are switched on/off: DS, D, DSE, DE, DSJ, DJ, DSEJ and DEJ. The full representation has 905 inputs.

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
| `python -m RQ3 mirrors` | Original/mirror training; complete mirror detections are reused |
| `python -m RQ3 normalization` | Detector-Right normalization; run `mirrors` first |
| `python -m RQ3 gh_rotation` | Separate G/H rotation training; frozen cache and manifest |
| `python -m RQ3 occlusion` | Coarse rendering, point noise and artificial rotation testing; run `gh_rotation` first; raw images needed for the 64-photo detection control |
| `python -m RQ3 live_examples` | Replay four specific local captures, which must be supplied separately |

The orientation comparisons retain their **835-input DSJ reference**. Primary classification scores exclude rejected detections. The artificial mirror mixture is a different test presentation. A high artificial G/H rotation score does not establish better live recognition. Four selected webcam examples do not estimate precision/F1 or error frequency.

The 22 withdrawn guided G/H captures are not current training/test data. Private captures are excluded from Git and the input bundle; drawing their figures again requires those local files. Existing figures and provenance are in `paper/figures/` and `results/`.

## 6. Live recognition and feature inspection

Use the supplied deployment models without retraining:

```bash
python -m research.live_camera
```

Choose **1** for DS, **2** for DSJ, **9** for G/H rotation, **7/8** for the matched mirror reference/mirror-trained models and **0** for right normalization. The default is DSJ. **Space → intended letter → Enter** saves a capture under `captures/confusions/`. **L** toggles landmarks, **N** starts a round and **Q/Esc** exits. Saved records include the model hash, exact detector input, landmarks and prediction. Free captures are error analysis, not automatic training data.

For the browser feature tool:

```bash
python -m streamlit run app.py
```

Select a model, enable **Camera on**, optionally enter the intended letter and use **Save capture**. The intended letter does not affect recognition. Only one tool can use the camera at a time; macOS may require camera permission. See the [live test guide](docs/LIVE_TESTPLAN.md).

## 7. Paper and evidence

Use the integrated English [paste-ready source](paper/group_integration/group_report_paste_ready.tex). Its [PDF preview](paper/group_integration/group_report_paste_ready_preview.pdf) explicitly marks missing coauthor assets. The [transfer package](paper/group_integration_ton.zip) contains the contribution and figures. Source-preservation checks and integration notes are in `paper/group_integration/` and `results/group_integration_20261107/`. The contribution uses six references; the bibliography check is in `results/bibliography_cleanup_20261108/verification.json`.

The complete group paper still needs the original RQ2 figure, code appendices, group bibliography, RQ1 contribution and shared abstract. Local previews use labelled placeholders for missing assets.

The standalone [working paper](paper/report.pdf) and [technical supplement](paper/appendix.pdf) can be built with Tectonic:

```bash
tectonic --only-cached -o paper paper/report.tex
tectonic --only-cached -o paper paper/appendix.tex
```

A fresh Tectonic installation may first need a run without `--only-cached` to download standard TeX resources. In Overleaf, paste the integrated source into `main.tex` within the existing group project.

## 8. Add RQ1/RQ2 contributions

Supply `RQ1/experiment.py` or `RQ2/experiment.py` with a `main()` function. This enables `python -m RQ1` or `python -m RQ2`; until then they report the missing contribution. `python -m RQ1 --status` and `python -m RQ2 --status` show the current status.

Reuse the shared detector/baseline and write outputs under `results/RQ1/` or `results/RQ2/`. Include source IDs, train/test splits, settings, software versions, predictions or detection statuses and a reproduction command. RQ2 must distinguish technical recovery from visually plausible landmarks. State any different dataset selection or baseline explicitly.

## 9. Documentation

- [RQ3 guide](RQ3/README.md): question, variants, findings and limitations.
- [Live test guide](docs/LIVE_TESTPLAN.md): free comparisons and optional collection protocols.
- [Figure guide](docs/APPENDIX_VISUALS.md): appendix figures and required source files.
- [Integration checklist](paper/INTEGRATION_TODO.md): outstanding group contributions.
