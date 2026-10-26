# Hand-gesture Lab 4 — separate steps, inspectable code

Read [WORKFLOW.md](docs/WORKFLOW.md) for the complete workflow and commands.
This project investigates whether explicit geometric finger features improve
recognition of Dutch manual-alphabet poses from your own photographs.

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

Both variants receive the same 21 image landmarks and 21 estimated world
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
│   └── camera/
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

## Tests and demonstration

```bash
python -m unittest discover -s tests -v
python tests/smoke_modular.py --out examples/my_software_test
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
