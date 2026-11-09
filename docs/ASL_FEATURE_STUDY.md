# Static ASL landmark feature study

This minimal research repository compares a Shin-inspired SVM baseline with interpretable hand-shape features and continuous joint angles, using SVM and LightGBM on one Kaggle ASL Alphabet dataset.

- [English interim report, ICML 2021 format](../paper/report.pdf)
- [Supplementary appendix](../paper/appendix.pdf)
- [Editable LaTeX](../paper/report.tex)
- [All setup and reproduction steps](../REPRODUCE.md)
- [Repository-relative paths and portability checks](../docs/REPOSITORY_PATHS.md)
- [Start from scratch: Kaggle download, landmark detection, rejected images and threshold experiments](../REPRODUCE.md#3-from-downloaded-images-to-accepted-landmarks-start-from-scratch)
- [Frozen results and verification](../results/)
- [RQ2: PowerShell reproduction, analysis and figure interpretation](#rq2-detection-recovery)

## RQ2 detection recovery

Use the [RQ2 experiment guide](../experiments/hand_detection_recovery/README.md) for setup, a small check, complete runs, resume, analysis and figures. Run its commands from the repository root with the existing Conda environment active.

For the exact package versions recorded for this feature study, optionally install `requirements-research.txt` in a separate research environment. The shared project continues to use `environment.yml` and `requirements.txt`.

### Why the filter percentages do not add up to 27.27%

The `test_all` results contain 17,598 original misses. Dilation recovers 3,560
images (20.23%), erosion 2,018 (11.47%) and the Canny overlay 1,391 (7.90%).
An image may appear in two or all three of those groups. These are overlapping
success rates, not disjoint parts of a total.

For example, a photo accepted after both dilation and erosion contributes to
both filter bars but contributes only once to the **Any variant** bar. Therefore,
the sum of the filter counts is 6,969 successful **image–filter pairs**, while
the union contains 4,799 **distinct images**, or 27.27% of the original misses.
Adding the rounded percentages gives 39.60%, which counts some images repeatedly.
The combined rate is computed as:

```text
unique recovery rate = images with at least one successful variant / original misses
                     = 4,799 / 17,598 = 27.27%
```

The final bar represents trying all three filters and accepting any technical
success; it is not the performance of a single selected filter. Inspect
`figures/recovery_overlap_data.csv` for the number of images recovered by
exactly zero, one, two or three variants:

| Successful variants per image | Number of images in `test_all` |
| --- | ---: |
| Zero | 12,799 |
| Exactly one | 3,024 |
| Exactly two | 1,380 |
| All three | 395 |

Thus, `3,024 + 1,380 + 395 = 4,799` distinct images recover, whereas
`3,024 + 2 × 1,380 + 3 × 395 = 6,969` counts every successful image–filter pair.
A photo that reaches `ok` still requires
visual review before its landmarks are described as plausible. The figures
measure detection recovery, not improvement in letter-classification accuracy.

### Evidence files and implementation

| File within the run | Research use |
| --- | --- |
| `settings.json` | Source selection, detector/filter settings, versions and checksums |
| `baseline.csv`, `no_hand.csv` | Original detections and the subset sent to preprocessing |
| `per_image.csv`, `summary.csv` | Paired outcomes and aggregate technical recovery |
| `recovered.csv` | Recovered filenames and their successful variants |
| `still_no_hand.csv` | Images with zero detected hands under every variant |
| `unrecovered.csv` | Images with no accepted retry, including other rejection states |
| `retry_outcomes.csv`, `by_letter.csv` | Status counts per variant and results per letter |
| `processed.csv`, `processed/` | Original-to-variant mapping, checksums and processed PNGs |
| `retry_landmarks.jsonl` | Stored retry outputs and hand coordinates |
| `n47_example.csv` | Measured statuses for the development example |
| `analysis.md`, `review_gallery.html` | Readable overview and visual inspection |

The entry point is [run.py](../experiments/hand_detection_recovery/run.py).
It calls [recovery_experiment.py](../research/recovery_experiment.py) for each stage.
Filters are in `variants()` in [morphology_pilot.py](../research/morphology_pilot.py),
and detection and acceptance are in `Detector` in [data.py](../research/data.py).
[analyze.py](../experiments/hand_detection_recovery/analyze.py) generates the analysis;
[plot_results.py](../experiments/hand_detection_recovery/plot_results.py) generates the figures.
Archive the reported run before another clean start. N47 and earlier development
images are included by default; the complete run is a descriptive dataset
evaluation, not an untouched holdout evaluation.

## Included artifacts and language

The six deployed live-model `.joblib` files under `models/`, their provenance JSON files, the four G/H diagnostic models and the MediaPipe model asset are included in version control. No retraining is required when those files are present after cloning. The pinned Python/scikit-learn environment is still required to load them.

Confusion matrices, OOF prediction tables and analysis reports under `results/` are versioned. Raw webcam captures under `captures/` remain local and ignored, as requested; they can contain participant information. Analysis reports may describe captures that are therefore unavailable in a fresh clone, and capture-replay commands require those local files. The Kaggle images under `data/` and new experiment runs under `runs/` are also ignored; see the download and extraction steps in `REPRODUCE.md`.

Documentation, application text, feature labels and the active workbook use English. Legacy serialized category/viewpoint IDs, recorded session names, original filenames and Dutch import aliases are retained for compatibility with existing models and evidence. They do not require participants to use Dutch labels. Documentation files are `docs/ALPHABET_CAPTURE.md` and `docs/GH_CAPTURE.md`.

## Simple live camera (recommended)

```bash
conda activate ktai-hand-gesture
python -m research.live_camera
```

A single fixed-size window displays the camera image and current letter. Select a trained model using the buttons below the image or keys **1-6**, without restarting:

| Button | Model |
|---|---|
| **Reference [1]** | Shin reference: distances and directions (820 inputs) |
| **Joint angles [2]** | Reference plus joint angles (835 inputs) |
| **All-letter tilt [3]** | Joint-angle model with all-letter ±15° tilt training |
| **C-only tilt [4]** | Joint-angle model with ±15° tilt training for C only |
| **G/H rotation [5]** | Joint-angle model with larger direction rotations for G/H only |
| **Image + depth [6]** | Reference plus 3D and projected 2D joint angles (849 inputs; no augmentation) |

The default is Joint angles. The augmented models are experimental comparisons, not generally better models. Model selection is locked during a guided round. Snapshot exports preserve the actual model hash and its 820/835/849-input schema.

The camera window keeps a fixed layout while models switch. No workbook, feature panels or browser reruns are involved. The snapshot button opens a frozen-frame letter annotation only when requested. Hand landmarks are shown by default; press **L** to toggle them. Overlays never change the detector input. A dash means no accepted hand detection. Press **Q** or **Esc** to close. Close other camera apps first. For a different camera, use `--camera 1`; on macOS allow camera access for your terminal/Python if requested.

### Save a confusion for analysis

Click **Bewaar verwarring** below the camera or press **Space** when you see a mistake. The currently displayed frame freezes. Type the intended letter (A-Z), then press **Enter** to save it; **Esc** cancels. The last intended letter is prefilled for the next capture, and Backspace clears it if unknown. Q is a valid letter while annotating; it only quits in live mode.

Each tool launch starts a new round with a unique ID. **New round / N** starts another round without restarting the camera. Switching models stays within the current round, allowing comparisons. The round label/number appears below the buttons and is saved as `capture_round`, alongside `session_id` and the exact active model in each `capture.json`. Previous captures remain untouched; older ones already contain a `session_id` but no friendly round label.

Optionally name the round when starting:

```bash
python -m research.live_camera --round-name "GH-back-of-hand"
```

Each snapshot creates a new folder under `captures/confusions/` containing:

- `frame.png`: original unmirrored camera photo.
- `model_input.png`: exact unmirrored image passed to the landmark detector.
- `preview.png`: mirrored preview with the predicted letter and visible landmarks.
- `capture.json`: intended/predicted letters, timestamps, landmarks and detection status, 820, 835 or 849 raw and preprocessed features matching the chosen model, joint angles when used, all class decision scores, image checksums and model SHA256/provenance.

Nothing is recorded until you request and confirm a snapshot. Captures stay local and are ignored by Git. Select another destination with `--output /absolute/path/to/confusions`. These manually chosen examples are for error analysis; they do not estimate live accuracy and are not automatically added to training.

The model recipe scored **97.748% accuracy** and **0.97524 macro-F1** on the shared five-fold accepted-image evaluation. The deployment model was then fitted on all 60,251 accepted images; it has no separate final-test or webcam accuracy measurement.

The local model has already been trained. After a fresh clone, generate the ignored artifact with `python -m research.live_model` if needed. The tool accepts `--model /absolute/path/to/model.joblib`.

## Guided tilt experiment: C, D and O

You do not need to manually edit or add training images. First collect planned webcam comparisons with the current model, separately from training:

```bash
python -m research.live_camera --guided --role development
```

The window guides 27 attempts: C/D/O × wrist neutral/up/down × three repetitions. Keep your finger shape and camera position fixed; tilt at the wrist only, and relax/reform the hand between repetitions. The tilt names are qualitative instructions, not measured angle targets. Press Space or click the button for each planned attempt, inspect the frozen pose and press Enter. Esc retries the current pose; Q exits in live mode. Predictions are hidden during collection to reduce selection of only correct examples. Save every planned attempt, including incorrect predictions and detection failures.

At a separate time, collect an evaluation session:

```bash
python -m research.live_camera --guided --role evaluation
```

The terminal prints the new session directory under `captures/guided/`. Resume an interrupted round with:

```bash
python -m research.live_camera --guided --resume captures/guided/YOUR_SESSION_DIRECTORY
```

The app resumes from finalized capture records, locks the baseline model version, and keeps development/evaluation roles separate. Both roles remain outside training. Use the development session to choose augmentation of the existing Kaggle training data, then compare the fixed baseline and chosen variant on the reserved evaluation session. Each saved record contains the planned letter, qualitative tilt, repetition, session role and the usual image/landmark/feature data. Collection does not automatically retrain a model or change its reported scores. A second session from the same signer tests session transfer, not signer-independent performance.

## Try the trained tilt-augmentation variant

```bash
python -m research.live_camera --model models/asl_shin_angles_tilt15.joblib
```

The original default model remains `models/asl_shin_angles.joblib`. The new experiment augments the 630 Shin direction features with -15°/+15° camera-x rotations, preserving distances and joint angles. The candidate has already been trained locally; fresh clones can reproduce it with `python -m research.studies.tilt_augmentation --stage train`.

Results are mixed: development captures improve from 17/27 to 18/27 (one additional O; C unchanged). On the same five Kaggle folds, accuracy changes from 97.74776% to 97.70294% and macro-F1 from .975244 to .974686. This does not establish a general improvement. [Full results and reproduction](../results/tilt_augmentation_01/report.md).

## C-only tilt experiment

```bash
python -m research.live_camera --model models/asl_shin_angles_C_tilt15.joblib
```

This separate candidate augments only the 1,912 C training images with -15°/+15° direction rotations; the other 58,339 originals remain single examples. It uses 64,075 fitting rows, with original total weight per source image preserved. On the 27 development captures its predictions are unchanged from baseline (17/27 correct; C 3/9). Kaggle five-fold accuracy is also unchanged at 97.74776%; macro-F1 differs by less than 0.000001. [Details and reproduction](../results/tilt_C_only_01/report.md). The default model remains the original baseline.

## Feature inspection app (optional)

```bash
conda activate ktai-hand-gesture
python -m streamlit run app.py
```

Enable **Webcam on**; the app displays **Recognized letter** using the full-data SVM with 820 Shin descriptors and 15 joint-angle measurements. A typed target letter is optional and only controls the Excel-feature comparison. Predictions use each current frame; no letter persists when no hand is detected. Alternatives are SVM rankings, not calibrated probabilities.

The trained artifact is `models/asl_shin_angles.joblib` (generated locally, ignored by Git). If it is missing, train it once before starting the app:

```bash
python -m research.live_model
```

Use `--force` to retrain or `python -m streamlit run app.py -- --model /absolute/path/to/model.joblib` to select another compatible local bundle. The Excel reference remains the sole original workbook in `assets/`.

The best primary comparison improves SVM accuracy from 97.637% to 97.748% on 60,251 accepted detections. This is a small development-set gain; the report documents selection, dataset and paper-comparability limitations. Remaining files are necessary implementation, tests, data, evidence or report assets. Superseded work is in the reversible sibling archive recorded in `CLEANUP.json`.

## Experimental 2D + 3D angle model

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_2d.joblib --round-name "2D-3D-test"
```

The new **Image + depth [6]** model adds 14 aspect-correct projected joint angles to Joint angles. The unchanged five-fold comparison scores 97.693% accuracy and .97465 macro-F1, slightly below Joint angles (97.748%, .97524). It corrects T in the latest 11 selected webcam errors, while G remains wrong; this is an experimental comparison, not a demonstrated general improvement. See `results/angles_2d_01/report.md` for all-letter results, fold variation and development replays.

## Live G/H-only diagnosis

```bash
conda activate ktai-hand-gesture
python -m research.live_gh --variant middle
```

This separate native window always chooses G or H; other letters are outside its scope. Buttons/keys 1-4 select all 3D angles, all 2D angles, both, or middle-PIP-only. L toggles the landmark overlay; Q/Esc closes. Close the ordinary camera window before starting this one. The selected development examples motivated the middle-only model; recognizing them does not establish independent live accuracy. The ordinary six-model tool remains `python -m research.live_camera`.

## Guided G/H viewpoint collection

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture
```

Three participants, G/H, four views (palm, oblique palm, oblique back, back), five repetitions: **120 planned photos, 40 per person**. This supersedes the earlier 60/90-photo three-view proposal. Predictions are hidden; participant transitions require Enter. Use `--participants 2` for 80 photos. Photos are separate from training. P01 is development; P02/P03 are reserved evaluation. Full instructions, the four-view illustration and resume/status commands are in [docs/GH_CAPTURE.md](../docs/GH_CAPTURE.md).

## Named alphabet test with an example for every letter

```bash
conda activate ktai-hand-gesture
python -m research.alphabet_capture
```

The participant form records name/code, development/new-person status, ASL fingerspelling experience and hand. Five A-Z rounds give 130 photos per person. A fixed dataset example appears beside the camera; **Space saves directly**, **H** marks extra guidance, and Q/Esc pauses. Predictions are hidden. The same stored images can later be scored by both frozen models, separately for new and development participants. See [the complete instructions](../docs/ALPHABET_CAPTURE.md), including resume commands and guidance rules. Dataset examples are not independently validated ASL instructions; this is an example-guided static reproduction task.
