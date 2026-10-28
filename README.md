# Hand-gesture Lab 4 — separate steps, inspectable code

Read [WORKFLOW.md](docs/WORKFLOW.md) for the complete workflow and commands.
For the practical guide to the SVM/Shin webcam study, development/test
capture, A/B comparison and a selected improvement, read
[RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).
The read-only experiment audit and one-variant replay are in
[VALIDATION.md](docs/VALIDATION.md).
This project compares landmark representations for manual-alphabet pose
classification. It supports your own Dutch manual-alphabet photos, imported
ASL datasets and a separate webcam study using a fixed SVM/Shin baseline.

| Component | Command or file |
|---|---|
| Collect training photos using alphabet references | `python -m streamlit run apps/collect_app.py -- --project data/ngt` |
| Inspect and assess finger features and thresholds | `python -m streamlit run apps/feature_check_app.py -- --project data/ngt` |
| Read or modify feature calculations | `gesturelab/features/finger_features.py` |
| Export precision/recall of manual finger assessments | `python -m scripts.evaluate_features --project data/ngt` |
| Freeze a shared experiment split | `python -m scripts.prepare_split --project data/ngt --out experiments/ngt_01` |
| Train the baseline | `python -m scripts.train_without_features --experiment experiments/ngt_01` |
| Train with additional finger features | `python -m scripts.train_with_features --experiment experiments/ngt_01` |
| Compare cross-validation results | `python -m scripts.compare_results --experiment experiments/ngt_01` |
| Evaluate both models on reserved test photos | Add `--final-test` to the comparison command |
| Try a trained model with the webcam | `python -m apps.webcam predict --project data/ngt --model experiments/ngt_01/with_features/model.joblib` |

In the basic two-variant experiment, both receive the same 21 image landmarks and 21 estimated world
landmarks: 126 normalized coordinates. The second receives 81 additional
geometric features, for 207 inputs. Images, splits, classifier, settings and
normalization are otherwise identical. SVM is the default;
`scripts/prepare_split.py --model` selects another supported classifier.

The 26 individual reference images are stored in `assets/alphabet/` as
`A.png` through `Z.png`. The filenames define the letter labels; no crop JSON
is needed. They come from the Dutch manual alphabet; see
[assets/SOURCES.md](assets/SOURCES.md). References guide recording and are not
training images. Start with eight letters and expand when ready. Photographs
capture poses; they do not capture the complete movement of dynamic letters.

To reopen a dataset export, extract it and use its `dataset` directory instead
of `data/ngt`. Features are recomputed from stored landmarks.
Refresh references with `python -m scripts.init_dataset --project <dataset-directory>`.

See [TRAINING_INPUTS.md](docs/TRAINING_INPUTS.md) for the separation between landmark
normalization, finger calculations and extensible model input variants.

## Code organization

```text
ktai-hand-gesture/
├── apps/           # Streamlit apps and the local webcam app
├── scripts/        # Dataset, training, comparison and demo commands
├── docs/           # Workflow, feature definitions and input-variant guide
├── gesturelab/     # Shared implementation
│   ├── datasets/
│   ├── features/
│   ├── training/
│   ├── camera/
│   ├── research/   # Protocol capture, live A/B, selected refit and fixed tests
│   └── validation/ # Independent feature/split/score audit and one-run replay
├── tests/          # Unit tests and complete synthetic workflow check
├── assets/         # Alphabet references and attribution
├── config/         # Configuration defaults
└── models/         # Downloaded MediaPipe model
```

Terminal entry points live in `scripts/`, the Streamlit and webcam apps in `apps/`, and
the detailed guides in `docs/`. Run all documented commands from the project root. Generated datasets and
experiment results are stored in `data/` and `experiments/`.
The input variants are defined in `gesturelab/features/training_inputs.py`;
training runs in `gesturelab/training/fitting.py`, and reports are produced by
`gesturelab/training/comparison.py`.

The comparison measures whether the additional geometric finger features improve
recognition. Both variants otherwise use the same data and classifier settings.

## Installation

Open Terminal in the `ktai-hand-gesture` project directory.

```bash
conda env create -f environment.yml
conda activate ktai-hand-gesture
```

For an existing environment:

```bash
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

The apps run locally. Guided recording opens a separate local camera window;
Python must run on the computer with the webcam. On macOS, Terminal/Python may
need camera permission. The first hand analysis downloads the MediaPipe model.
Minimal Linux installations may need `libgles2`.

On macOS, LightGBM also needs OpenMP in the Conda environment:

```bash
conda install -n ktai-hand-gesture -c conda-forge llvm-openmp
```

## Train locally with external ASL data

You can train on downloaded photos without recording your own dataset first.
Extract the download into `data/external/`, activate `ktai-hand-gesture`, and use
the importer for its format:

| Download | Raw files | Local training guide |
|---|---|---|
| Roboflow COCO export | `data/external/asl_letters_v1/` with `train/`, `valid/` and `test/` | [ASL_TRAINING.md](docs/ASL_TRAINING.md) |
| Kaggle ASL Alphabet | `data/external/archive-kaggle/` with the alphabet train/test folders | [KAGGLE_TRAINING.md](docs/KAGGLE_TRAINING.md) |

For the extracted Kaggle dataset, this command imports all 78,000 letter photos,
calculates landmarks, trains both Random Forest variants and creates a comparison:

```bash
conda activate ktai-hand-gesture
python -u -m scripts.train_kaggle_dataset
```

The imported dataset is saved in `data/asl_kaggle/`; models and the HTML report
are saved in `experiments/asl_kaggle_rf_01/`. Raw downloads, imported datasets and
experiments are separate. If you already have a fully imported dataset with
`manifest.json`, reuse it; do not initialize it with NGT references or import it again from scratch.
The Kaggle command supports resuming the same run. For retraining after changes,
choose a new experiment directory.

See [the external-data workflow](docs/WORKFLOW.md#alternative-train-locally-with-external-asl-data)
for both command sequences, output files and how to reuse existing data. These
models use ASL labels; keep them separate from NGT recordings. The supplied
Kaggle test photos duplicate training photos, so that run reports development
cross-validation without an independent final test.

If you still need the Roboflow files, [ASL_DOWNLOAD.md](docs/ASL_DOWNLOAD.md)
explains the optional download step. Local import and training do not require a
Roboflow API key. The first landmark analysis downloads the MediaPipe model if
it is not already present.

## Compare the Shin 2021 paper method

Three independent inputs implement the distance/axis method from Shin et al.:
190 distance values, 630 axis cosines and 820 combined values. They can be
compared with the existing 126/207-value inputs on identical photo IDs and
grouped folds, using SVM and LightGBM. No letters are automatically excluded.
The comparison also includes finger features alone (81), Shin plus finger
features (901), and all three sources together (1,027). The 820-versus-901
comparison tests whether the finger features add value beyond the Shin inputs.

For the already imported and analyzed Kaggle data:

```bash
python -m scripts.run_shin2021 --project data/asl_kaggle --out experiments/shin2021_kaggle_01 --grouping source --folds 3 --allow-no-test
```

Open `experiments/shin2021_kaggle_01/report.html` afterward. Use a new output
directory. This is an application of the article's method, not a reproduction
of its scores. See [PAPER_SHIN2021.md](docs/PAPER_SHIN2021.md) for formulas,
explicit assumptions, parameter profiles, exact split reuse, individual commands,
final testing and webcam use. Old experiment/model files are preserved; updated
input-code hashes require new training rather than disabling checks.

## Collaborating with dataset ZIPs

Export the dataset, let the next student extract it and add photos under a unique
participant code, then pass on the updated ZIP. Use successive versions: automatic
merging of independently extended datasets is not implemented. See the
[step-by-step ZIP collaboration guide](docs/WORKFLOW.md#sharing-a-dataset-zip-with-classmates)
for commands and directory examples.

## Tests and demonstration

```bash
python -m unittest discover -s tests -v
python tests/smoke_modular.py --out examples/my_software_test
python tests/smoke_shin2021.py --out examples/shin2021_software_test
```

Use a new output directory. The smoke test creates explicitly synthetic skeleton
images and exercises the entire workflow without a camera. Its scores do not
measure recognition of real NGT photos. `scripts/demo.py` generates synthetic data
used by the automated tests.

`projects/demo_synthetic` is optional generated example data. You can skip or
delete that directory when using your own photos; keep `scripts/demo.py` for the tests.
The smoke test creates its own dataset. See the optional demo instructions in
[WORKFLOW.md](docs/WORKFLOW.md#optional-what-is-projectsdemo_synthetic-for) for the file
layout, an inspection example and regeneration commands.

[FINGER_FEATURES.md](docs/FINGER_FEATURES.md) explains the geometry. [WORKFLOW.md](docs/WORKFLOW.md)
covers recording, inspection, training, evaluation, comparison and reporting.
