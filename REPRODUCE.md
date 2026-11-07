# Reproducing the interim ASL study

Run every command from the repository root. The sole retained dataset is **Kaggle ASL Alphabet**, `grassknoted/asl-alphabet`, with 78,000 A-Z training images. Dataset files are local and ignored by Git. The active user-authored Excel file is `assets/asl_feature_references.xlsx`; its labels and categories have been translated to English without changing the allowed feature values. The 49 historical slash relaxations are in `config/proposal_changes.json` and are applied in memory, not saved as a second workbook.

The report is `paper/report.pdf`; editable source is `paper/report.tex`. It follows the supplied ICML 2021 accepted template, with at most five main-text pages, then references. The separate `paper/appendix.pdf` contains the extended supplementary material. Author/student details remain to be filled in. This is a course report, not an ICML publication.

## 1. Environment

Use Python 3.12. The existing local Conda environment can be activated with:

```bash
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

For a fresh environment:

```bash
conda create -n ktai-hand-gesture python=3.12
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

On macOS, if LightGBM reports a missing `libomp` library:

```bash
conda install -c conda-forge llvm-openmp
```

For the simplest live test, use the native camera tool:

```bash
python -m research.live_camera
```

It opens one fixed 640x480 camera area with the current letter overlaid and model buttons below. The camera is opened once; each capture is analyzed without browser/component reruns. Display mirroring does not change the model input. A dash replaces the previous letter when no valid hand is found. Press Q/Esc or close the window to release the camera. Close other camera apps first; use `--camera 1` for a second camera and `--model PATH` for another compatible trusted bundle. On macOS, grant camera access to Terminal/Python if required. Images are recorded only after an explicit confirmed snapshot. This is raw per-frame prediction, without temporal smoothing or a new accuracy claim.

The compact panel exposes six current choices: Reference [1], World joints [2], G/H y180 only [9], Fresh original [7], Mirror trained [8] and Right normalized [0]. **More models [A]** opens four older exploratory models; all historical digit shortcuts remain valid. **N** starts a new round and **L** toggles landmarks. Compare [2] with [9] for the standalone G/H rotation, not the older G/H rotation [5].

To collect an example, click **Save example** or press Space. The selected displayed frame is frozen; type the intended letter A-Z and press Enter. Esc cancels and resumes the camera. Backspace clears the intended label if unknown. The intended label does not enter the model and the original prediction is retained.

The browser alternative is `python -m streamlit run app.py`: use the sidebar model selector, **Camera aan**, **Bedoelde letter** and **Meetmoment bewaren**. The last analyzed browser frame is saved with that frame's detection and the active model; a failed detection is also a valid saved attempt. The webcam's continuously displayed video can have advanced beyond the analyzed frame. Saved snapshots have the same format as native snapshots, with `camera: browser` and `source_frame_id`. They remain selected diagnostic examples. See [the practical test and handoff plan](docs/LIVE_TESTPLAN_NL.md) for planned evaluation.

The default destination is `captures/confusions/` (ignored by Git). Each unique capture folder contains lossless `frame.png` (raw unmirrored camera image), `model_input.png` (exact detector input), `preview.png` (mirrored visual context) and `capture.json`. The JSON preserves image hashes, capture/save UTC timestamps, human intended letter, prediction/alternatives, detection status/settings/version, image/world landmarks, the selected model's 820 or 835 feature names and raw/preprocessed values, joint angles when included in that model, 26 SVM decision values and model hash/metadata. The save operation uses the stored displayed-frame detection and never reruns detection on a later frame. Missing-hand snapshots remain valid diagnostic examples with no classification vector. Decision values are not probabilities; selected confusion examples do not form a representative test set. Files are written into a temporary directory and only finalized when the complete capture succeeds.

```bash
python -m research.live_camera --output captures/confusions
```

For later analysis, compare intended/predicted letter pairs, inspect the preserved image landmarks versus the actual fingers, and compare joint angles and decision-score differences for errors and correctly classified controls. Photograph some correct examples too; collecting only errors cannot establish their frequency. Captures are not included in existing training folds or metrics automatically.

### Guided collection for an augmentation experiment

```bash
python -m research.live_camera --guided --role development
# At a separate time:
python -m research.live_camera --guided --role evaluation
```

The guided protocol has 27 tasks: C/D/O × neutral/up/down wrist tilt × three repetitions, with C/D/O interleaved within each tilt and repetition. Keep finger shape, camera and distance consistent; reform the hand between repetitions. Tilt labels are qualitative, not measured degrees. The fixed task label is shown, and the current prediction is hidden. Press Space/click to freeze, Enter to save, Esc to retry. Do not repeat an attempt merely because the model got it wrong; failed detections and wrong predictions are valid planned outcomes.

Each run creates `captures/guided/<role>_<timestamp>_<id>/session.json` and finalized capture directories. `capture.json` includes immutable task identity, intended letter, tilt, repetition and development/evaluation role. Protocol metadata marks these examples as outside training. The original Kaggle manifest/cache remain untouched. The window closes when all 27 tasks are finalized. The terminal prints the session path; resume using:

```bash
python -m research.live_camera --guided --resume captures/guided/YOUR_SESSION_DIRECTORY
```

Resume reconstructs progress from finalized captures and ignores interrupted temporary writes; it rejects changed model hashes or duplicate task IDs. Evaluation-session role is retained when resuming. The existing model is the reference used for collection; future candidate models can be compared on the same saved inference inputs.

Choose augmentation settings using the development session and existing training data only; reserve the evaluation session until the candidate is fixed. Apply augmentation only within training partitions, never before splitting in a way that could put transformed duplicates into validation. The augmentation runner is now available (section 9); the guided camera mode itself remains a collection tool and does not retrain models. These two sessions share a signer, so they cannot establish subject-independent generalization.

For the optional extended feature-inspection app, launch Streamlit through the same Python environment so MediaPipe is available:

```bash
python -m streamlit run app.py
```

Stop the server with Ctrl-C in its terminal. The native camera window now provides nine model buttons, also available through keys 1-9:

- **Reference**: original Shin 820-input reference model (`models/asl_shin.joblib`).
- **World joints**: Shin plus 15 joint-angle measurements (`models/asl_shin_angles.joblib`).
- **All-letter tilt [3]**: the all-letter tilt-augmented joint-angle model (`models/asl_shin_angles_tilt15.joblib`).

- **C-only tilt [4]**: selective C-only tilt (`models/asl_shin_angles_C_tilt15.joblib`).
- **G/H rotation [5]**: selective G/H direction rotations (`models/asl_shin_angles_GH_rotation.joblib`).
- **Image + depth [6]**: the 849-input 3D-plus-projected-angle model (`models/asl_shin_angles_2d.joblib`).
- **Fresh original [7]** and **Mirror trained [8]**: the matched original/reflected training comparison.
- **G/H y180 only [9]**: standalone G/H turns on the original-data model, without mirrors, camera calibration or routing rules (`models/asl_shin_angles_GH_y180.joblib`).

Models are loaded lazily and cached in the same window; switching does not restart the camera. A snapshot uses the currently displayed model's bundle/hash and matching 820-, 835- or 849-value vector. A model cannot be switched within a guided session. The extended Streamlit app can also predict letters using the default joint-angle SVM. If its generated model artifact is absent (e.g. after cloning), train it once:

```bash
python -m research.live_model
```

For the first three buttons on a fresh checkout, train each missing artifact (existing files are protected against overwrite):

```bash
python -m research.live_model --variant shin
python -m research.live_model --variant shin_angles
python -m research.studies.tilt_augmentation --stage train
```

The current local workspace contains the six historical trained artifacts and the two mirror-study artifacts (the later sections reproduce the C-only and G/H models). The reference uses the same project-profile SVM and original inputs as its frozen CV comparison. Its separate live schema uses 820 features and no angle scaler; six variants use 835; Image + depth uses 849. The live reference was also compared on the existing 27 development captures: 17/27 correct, C/D/O macro-F1 .609524.

The default joint-angle training command uses all 60,251 accepted development images, the same 835 input columns and SVM/preprocessing configuration as the isolated five-fold angle study, and saves `models/asl_shin_angles.joblib` plus a provenance JSON. It refuses to overwrite an existing artifact unless `--force` is given. This deployment fit is not another evaluation run; the 97.748% accuracy belongs to the earlier five-fold comparison, not to an independent test of this final fit.

In the app leave **Recognize letters using trained model** enabled and switch on **Webcam on**. No target letter is required. A typed letter only controls comparison against workbook references and does not enter the model. Without a target, features are shown against the model's predicted letter; that is a descriptive comparison, not independent evidence that the prediction is correct. No-hand/multiple-hand outcomes clear the displayed prediction. Alternatives are decision-function rankings, not probabilities. The preview is mirrored for convenience, while the frame used by the detector is not mirrored.

Select another compatible locally trained model with `python -m streamlit run app.py -- --model /absolute/path/to/model.joblib`. Only trusted local joblib artifacts should be loaded. The app does not record a new webcam dataset or constitute a controlled rotation experiment.

## 2. Verify the preserved experiment

```bash
python -m unittest discover -s tests -v
python -m research.verify
python -m research.verify --features
```

The first command runs arithmetic, geometric, workbook/slash, input-schema, live-frame and weighting tests. The last command rebuilds all 60,251 descriptor rows from stored landmarks and requires exact float32 equality to the historical central cache. It does not rerun the landmark detector. The verification also reconciles all eight isolated model variants and all 26 letters with saved OOF predictions, checks frozen outer partitions, input manifest SHA256 and targeted weighting metrics. It writes `results/verification_current.json`.

Expected inventory: 78,000 images; 60,251 `ok`, 17,598 `no_hand`, 151 `multiple_hands`. Expected cached dimensions: X=60,251 x 916 plus angles=60,251 x 15. The manifest SHA256 is `ff04b8e95e280109416799fa69d955861113a3908aeda583fe36ea2ef8ca3d35`.

Main pooled SVM results: Shin 97.63655375% / macro-F1 .973862351; Shin+15 angles 97.74775522% / .975243912. Angles fix 143 images and lose 76, net +67. These scores apply only to accepted detections. Read the report for comparability limits.

## 3. From downloaded images to accepted landmarks (start from scratch)

This section is the complete data preparation route for a new collaborator. **Importing images and detecting landmarks are separate steps.** The importer does not detect hands. The analysis step runs MediaPipe on every imported image and determines which images are eligible for classifier training.

### 3.1 Clone and install

If you do not have the repository yet:

```bash
git clone https://github.com/nlamah/ktai-hand-gesture.git
cd ktai-hand-gesture
conda create -n ktai-hand-gesture python=3.12
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

Run subsequent commands from this repository root, with this environment active. On macOS install `llvm-openmp` from conda-forge if LightGBM reports a missing `libomp` library (section 1). Dataset files and derived caches are ignored by Git: cloning alone does not supply the images or landmarks.

### 3.2 Download the actual dataset from Kaggle

1. Open [ASL Alphabet by grassknoted on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. Sign in if Kaggle requests it, download the dataset archive, and extract the entire archive. Keep both the training and supplied test folders. Do not download a different ASL dataset.
3. Place the extracted source outside the repository, for example `../asl-alphabet-download`. This is the download source, not the imported experiment project.
4. Check the extracted structure. The source directory passed to the importer must contain `asl_alphabet_train` and `asl_alphabet_test`. Kaggle sometimes repeats these folder names; the importer accepts either structure:

```text
../asl-alphabet-download/
  asl_alphabet_train/asl_alphabet_train/A/A1.jpg
  asl_alphabet_train/asl_alphabet_train/B/B1.jpg
  ...
  asl_alphabet_test/asl_alphabet_test/A_test.jpg
```

The variant `asl_alphabet_train/A/A1.jpg` and `asl_alphabet_test/A_test.jpg` is also supported. Point `--source` to their common parent, not to the `A` folder. Only A–Z are imported: 3,000 training images per letter, 78,000 total. The `space`, `del` and `nothing` classes are excluded. Supplied test images with the same pixels as training images are excluded and recorded in the manifest; they are not an independent test set. J and Z here are static images, not motion recognition.

### 3.3 Import the images (no landmark detection yet)

Use a new project directory for the experiment. The following example consistently uses `../asl-default`; do not overwrite the historical `data/asl_kaggle` project.

```bash
python -m research.dataset import --source ../asl-alphabet-download --project ../asl-default
```

Replace the source path if you extracted elsewhere. Import creates `manifest.json`, `originals/` (original bytes) and `images/` (RGB PNG detector inputs). They represent the same images, not separate training datasets. The manifest records labels, hashes, original filenames and relative paths. Initially image status is `pending`. Import can take time and requires storage for both representations in addition to the downloaded source.

If import was interrupted, resume with the **same source and output paths**, without moving or modifying the downloaded files:

```bash
python -m research.dataset import --source ../asl-alphabet-download --project ../asl-default --resume
```

Do not resume a completed project to start a different experiment; choose a new project path instead.

### 3.4 Detect landmarks in every image

```bash
python -m research.dataset analyze --project ../asl-default
```

This is the step responsible for the substantial image rejection. It opens each imported RGB PNG and runs **MediaPipe Hand Landmarker in IMAGE mode**, allowing up to two detected hands (`num_hands=2`). Detection and presence confidence thresholds default to **0.5 each**. The checked model is `models/hand_landmarker.task`; if absent, it is downloaded automatically from Google's MediaPipe model storage and checked against the SHA256 in `research/data.py`. Internet access is required for that first download.

For one detected hand, the manifest stores 21 image landmarks, 21 world landmarks, handedness, MediaPipe version and the requested detector thresholds. The geometric feature calculation is also validated. Each image receives one status:

| Status | Meaning | Used to train the letter classifier? |
|---|---|---|
| `pending` | Not processed yet | No |
| `ok` | Exactly one hand detected and geometric features valid | Yes, if included and assigned the development role |
| `no_hand` | Detector returned no hand | No |
| `multiple_hands` | Detector returned more than one hand | No |
| `invalid_geometry` | One hand returned, but geometric features failed validation | No |

Rejected images **are not deleted**. A detected landmark configuration is not necessarily anatomically correct; `ok` is an automatic acceptance criterion, not a human quality label. Detection confidence is different from letter-classification confidence.

Analysis checkpoints the manifest every 1,000 images and saves completed work on normal Python interruption. It may remain quiet between checkpoints; the command prints the processed count when finished. Stop with Ctrl-C and rerun the same command to process remaining `pending` images. After an abrupt process termination, work since the last saved checkpoint may be repeated. Only one writer can update a project at a time.

### 3.5 Count acceptances and inspect rejected images

```bash
python -m research.detection_report --project ../asl-default --out runs/detection_default
```

The command prints status counts and percentages for included development images and writes:

- `runs/detection_default/by_letter.csv`: counts for each status and acceptance percentage for every letter.
- `runs/detection_default/rejected_images.csv`: IDs, letters, statuses, original filenames, and absolute image paths. Open these paths to inspect rejected images. Pending images are listed in the counts but are not called rejected.

Historical reference counts (not guaranteed for a new installation):

| Status | Images | Percentage of 78,000 |
|---|---:|---:|
| `ok` | 60,251 | 77.245% |
| `no_hand` | 17,598 | 22.562% |
| `multiple_hands` | 151 | 0.194% |
| Total rejected | 17,749 | 22.755% |

Thus the published project classifier metrics apply to **accepted detections**, not all 78,000 input images. Report detection coverage separately from accuracy/macro-F1. The historical manifest did not record threshold metadata: although the current implementation defaults to 0.5/0.5, an identical historical detector configuration cannot be independently proven. MediaPipe versions and platforms can change detections; a fresh run need not produce the exact historical counts or landmarks.

### 3.6 Experiment with rejection thresholds

The relevant implementation is `Detector` and `analyze_project` in `research/data.py`; CLI options are in `research/dataset.py`. To test lower thresholds without changing the reference project, import the same source into a separate project and analyze it:

```bash
python -m research.dataset import --source ../asl-alphabet-download --project ../asl-threshold030
python -m research.dataset analyze --project ../asl-threshold030 --detection-confidence 0.3 --presence-confidence 0.3
python -m research.detection_report --project ../asl-threshold030 --out runs/detection_threshold030
```

Compare both CSV reports. To identify which individual images changed status, join `rejected_images.csv` inventories or the two manifest sample lists by `id`. IDs are based on image pixels and stay the same across these imports. Keep the model, software versions, source images and acceptance rule fixed when comparing thresholds. Vary one threshold at a time if you want to distinguish detection from landmark-presence effects; 0.3/0.3 above is only an example, not a recommended optimum.

**Changing flags on an already completed project does not automatically redetect it.** By default only pending images are analyzed. To deliberately replace detection results for every image in an experimental project:

```bash
python -m research.dataset analyze --project ../asl-threshold030 --detection-confidence 0.3 --presence-confidence 0.3 --reprocess
```

`--reprocess` overwrites existing landmark/status results in that project; export your reports first or use a separate project per setting. For an interrupted first detection pass, resume with the same thresholds and without `--reprocess`. If an all-image `--reprocess` pass is interrupted, rerun with `--reprocess` to ensure every image gets the new settings; this repeats already completed work. Without that flag, old non-pending records would remain and settings could be mixed across samples. Do not run threshold experiments against the preserved historical project.

A lower rejection rate alone is not proof of improvement: inspect newly accepted landmarks for wrong hands or implausible joints, count rejection per letter, and assess downstream classification on fixed evaluation images. Selecting a new accepted subset also changes the classification task; compare coverage and performance on the shared accepted subset, and report the added subset separately. Do not tune thresholds using final test labels. These experiments need no webcam: they operate on stored images.

### 3.7 Convert accepted landmarks to model inputs

After landmark detection finishes, build a new feature cache from the same project:

```bash
python -m research.prepare --project ../asl-default --out runs/cache_default
```

This reads landmarks already stored in the manifest: **it does not detect landmarks again**. It retains included development samples with `status=ok` and writes `inputs.npz`, `inputs_metadata.json` and `summary.json`. They contain 820 Shin descriptors, 70 category indicators, 26 compatibility scores and 15 separate angles, plus labels, IDs and five grouped folds (seed 42). The six groups are numbered filename blocks of 500, not verified people. Do not claim person-independent evaluation from these folds.

The output directory must not already exist. For a threshold experiment, use its project and a separate cache, for example `--project ../asl-threshold030 --out runs/cache_threshold030`. All subsequent training commands must explicitly use the chosen cache:

```bash
python -m research.studies.compare_models --stage train --models svm lightgbm --run runs/cache_default --out runs/from_scratch_models
python -m research.studies.compare_angles --models svm lightgbm --run runs/cache_default --baseline-run runs/from_scratch_models --out runs/from_scratch_angles
```

These are substantial training jobs. Section 4 describes their models. Omitting `--run` selects the historical default cache and therefore does **not** train on your new extraction. Historical result verification (`research.verify`) is tied to the frozen 60,251-image evidence; it is not a generic validator for a new detector experiment.

To rebuild from the preserved historical landmarks instead, without redetection:

```bash
python -m research.prepare --project data/asl_kaggle --out runs/cache_from_frozen_landmarks
```

The active workbook now uses English labels with the same allowed values; historical slash relaxations remain applied in memory from JSON. The workbook and proposal JSON now use English text, so their byte hashes differ from historical Dutch inputs even though their allowed feature values are preserved. New caches record their actual inputs; use one cache consistently across a new comparison.

## 4. Retrain SVM and LightGBM on shared folds

These are real training runs on 60,251 detected images, not unit tests. They can take substantial time and write large model checkpoints. Use a fresh output directory; do not overwrite frozen evidence. The central historical cache supplies exact original features/folds by default.

```bash
python -m research.studies.compare_models --stage train --models svm lightgbm --out runs/model_comparison
python -m research.studies.compare_angles --models svm lightgbm --baseline-run runs/model_comparison --out runs/angles
python -m research.studies.report_angles --out runs/angles
```

The first command fits baseline820 and full931 for both classifiers. The second reuses those baseline OOF predictions only after checking input hashes and fits the isolated835/846/861 configurations. The unchanged baseline predictions make the feature comparison paired. Add `--run /tmp/ktai-cache-rebuilt` to **both** commands if using rebuilt inputs; input hashes must agree. The results in `results/` are the historical frozen evidence, not outputs of these new commands.

SVM: RBF C10, gamma=scale, balanced. LightGBM: 100 estimators,31 leaves,learning rate .1,min child20,seed42,deterministic,column-wise,4 threads. Angles are median-imputed and standardized within training folds; all other blocks pass through unchanged. No absent angles occurred historically.

## 5. Grouped permutation importance and ablation

Use the output models trained in step4. Validation permutations use five repeats and a 100-per-letter-per-fold cap, matching the measured study. The optional cap0 is a new, more expensive analysis, not the historical result.

```bash
python -m research.studies.compare_models --stage permutation --models svm lightgbm --out runs/model_comparison --repeats 5 --max-per-class 100
python -m research.studies.compare_models --stage ablation --models svm lightgbm --out runs/model_comparison --groups angles categories excel_scores
```

Permutation shuffles whole one-hot blocks jointly, but overlapping derived groups can still create implausible combinations. It measures fitted-model reliance, not unique added information. Ablation refits after removing a block. The two quantities are not interchangeable.

## 6. Repeat weighting searches

```bash
python -m research.studies.tune_groups --out runs/group_weights
python -m research.studies.tune_index --baseline runs/angles --out runs/index_weights
```

Both use5outer/3inner grouped folds; inner training and validation cap150 images/letter and deterministic seeds. Group weights .25/.5/1/2; index MCP/PIP/DIP .5/1/2. Searches make one coordinate pass maximizing mean inner macro-F1 then accuracy. Final outer fits use all outer-training data. Weight2 acts on standardized values and contributes4x to that squared-distance component, while gamma=scale also changes. These bounded searches are not exhaustive model tuning.

The targeted study reuses the unweighted angle OOF baseline, checks sample IDs and retains inner-group evidence. Use `--run` consistently if using another cache. No independent final test was acquired.

## 7. Figures, report and evidence

```bash
python -m research.paper_assets
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.paper_core_assets
```

The first command regenerates the pipeline diagram, actual dataset O-error examples, all-letter tables, model-effect plot and complete confusion matrices from frozen results. It requires the retained dataset for qualitative examples and recomputes paired statistics and the descriptive block-resampling sensitivity range. The second command regenerates the earlier detailed feature-study assets (the former Appendix K control table), checking that coordinate-control baselines match the historical scores and that corrected/lost counts reconcile with accuracy. It also updates the main pipeline and letter-effects figures. These assets distinguish the earlier world-joint net +67 result from the same-coordinate net +4 result; no model is retrained. Figure source is code, not generated illustrative data.

With a standard TeX installation:

```bash
cd paper
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
pdflatex appendix.tex
pdflatex appendix.tex
```

Alternatively, install official Tectonic and run from the repository root:

```bash
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

The official ICML2021 `.sty`/`.bst` are included; download origin/checksums are in `paper/template_provenance.json`. T1 font encoding preserves Times under both pdfLaTeX and Tectonic/XeTeX. The current main report is a concise Dutch working version with TODOs for the two pending contributions. Keep the main text within five pages; references may continue on page six. The technical appendix remains a separate document. Inspect the rendered PDF after modifications.

`results/angles/` contains the main8comparisons and full26-letter metrics/predictions; `results/model_comparison/` the full feature/permutation/ablation study; `results/index_weights/` targeted tuning; `results/weights/` broad nested tuning; `results/pilot_*` the earlier3fold81-feature comparison. Do not compare pilot fold means directly to pooled5fold metrics. Compact historical reports contain old path strings as provenance; current commands in this file replace those paths.

## 8. Cleanup and recovery

`CLEANUP.json` records the sibling archive path, old Git revision, moved directories and preserved manifest/workbook hashes. Original tools, checkpoints, extra datasets and previous documents are in that archive. Nothing is permanently deleted. The `.git`, `.agents` and local environment/credentials are left intact. Restore archived items to their recorded original relative paths if needed, after backing up the current version; do not overwrite current work blindly.

The primary code and report are AI-generated; code inspection, tests and consistency audits are attempts at verification, not independent human or linguistic validation. There is no controlled rotation or signer-independent evaluation yet. Shin's paper reports87.60% on ASL Alphabet and no matchingF1; our97.6-97.7% filtered-grouped results are not a reproduction of its score.


## 9. Fixed tilt augmentation experiment

This experiment was run with one predefined recipe: for every original Kaggle training row, add -15° and +15° rotations of its 210 direction vectors about the camera-x axis. Only columns 190:820 are transformed; normalized distances 0:190 and extra joint angles 820:835 remain unchanged. There are 180,753 fitting rows from 60,251 distinct images. Each of the three variants has sample weight 1/3 to retain total fitting weight per original. SVM/preprocessing parameters remain the fixed project profile.

```bash
python -m research.studies.tilt_augmentation --stage train
python -m research.studies.tilt_augmentation --stage cv
python -m research.studies.tilt_augmentation --stage evaluate
python -m research.studies.report_tilt
```

`--stage train` refuses to overwrite `models/asl_shin_angles_tilt15.joblib`. Use a new `--candidate` and `--out` for a separate experiment. The source hashes and preset are locked in `results/tilt_augmentation_01/protocol.json`; no angle or axis search was performed. CV uses the original five grouped partitions and augments only fitting rows after splitting, leaving validation examples untouched. Fold checkpoints support resume under the same frozen settings. Captures are never training inputs. Evaluation sessions must match the baseline model hash and expected role; incomplete evaluation sessions receive no scores.

The historical run yielded:

| Scope | Baseline accuracy / macro-F1 | Augmented accuracy / macro-F1 |
|---|---|---|
| Five-fold Kaggle,26 letters |97.74776% / .975244 |97.70294% / .974686 |
| Development webcam,27 C/D/O images |62.96% / .610035 |66.67% / .642439 |

C remains 3/9 correct, D 8/9, and O improves 6/9 to 7/9. The already inspected friend's O example changes D to O; this selected example is descriptive, not an independent test. The evaluation session existed but contained 0/27 records when this experiment was run, so it has no held-out webcam result. The fixed feature-space rotation is not a physical camera/perspective/occlusion simulation. No statistical significance or general improvement is claimed.

Try the trained candidate in the same native camera tool, after closing the existing camera window:

```bash
python -m research.live_camera --model models/asl_shin_angles_tilt15.joblib
```

Baseline comparison:

```bash
python -m research.live_camera --model models/asl_shin_angles.joblib
```

Both accept the same snapshot controls and save their actual model hashes. The baseline remains the default. All outputs, matched OOF predictions, per-letter scores and numerical audits are retained in `results/tilt_augmentation_01/`.


## 10. Selective augmentation of C only

```bash
python -m research.studies.tilt_augmentation --stage train --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.tilt_augmentation --stage cv --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.tilt_augmentation --stage evaluate --letters C --out results/tilt_C_only_01 --candidate models/asl_shin_angles_C_tilt15.joblib
python -m research.studies.report_C_tilt
```

The same ±15-degree camera-x recipe is applied only to C. There are 1,912 selected originals, 58,339 unselected originals and 64,075 total fitting rows. Selected sources have three variants weighted 1/3 each; all other examples retain weight 1. To avoid accidentally changing class balance or angle normalization through selective duplication, scaling/imputation and balanced class weights are fitted on original fitting rows, and gamma=scale is resolved on that original preprocessed matrix. Full-data resolved gamma and class weights match the baseline exactly. These reference-fitting controls differ from the earlier all-letter run's automatic gamma over its augmented matrix and are recorded in the new experiment protocol.

All five original outer folds were repeated with augmentation after splitting. The same 60,251 validation IDs are scored exactly once each. Accuracy remains 97.74776% (58,894 correct); macro-F1 changes .97524391 to .97524465 due to three wrong-to-wrong prediction changes. No formerly incorrect example is recovered and no correct example is lost. Development predictions are identical to baseline: 17/27, macro-F1 .610035, C 3/9, D 8/9, O 6/9. The already inspected friend's O remains D. This particular C-only recipe therefore gives no observed practical improvement; no alternative angles/axes were searched in this run.

Live comparison:

```bash
python -m research.live_camera --model models/asl_shin_angles_C_tilt15.joblib
```

The original and all-letter models are retained separately. Full per-letter metrics, matched predictions, input hashes and controls are in `results/tilt_C_only_01/`.


## 11. Narrative report and technical supplement

Historical revision (superseded by the Dutch group draft below): the English ICML report followed the research story: informal C/D/O and U/V/R observations, hand-shape features, measured angles, persistent C/D confusion and the two tilt-augmentation trials. The main text is four pages, followed by references. Technical equations, detailed settings, original all-letter tables/confusions and augmentation controls are in the separate appendix. AI-generation and verification limitations remain explicit.

Rebuild the narrative figures/tables from the local stored evidence:

```bash
python -m research.paper_story_assets
```

This needs the existing development captures for its hand-cropped C examples; the already generated figures are included for ordinary LaTeX compilation. Existing detailed figures/tables can still be regenerated with `python -m research.paper_assets`. Compile both `paper/report.tex` and `paper/appendix.tex` using the earlier LaTeX/Tectonic instructions. `paper/technical_details.tex` contains the material moved out of the main text; do not overwrite it by copying the new narrative methods section.

## G/H hand-orientation experiment

A separate experimental model adds camera-y direction rotations of -90, +90 and 180 degrees only for G and H. Distances and joint angles remain unchanged. This approximates orientation variation in feature space, not physical palm/back rendering or detector errors. Existing models are preserved.

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_GH_rotation.joblib
```

The initial caption is **G/H rotation**. Select it again at any time with button/key **5**; **All-letter tilt [3]** is the separate earlier all-letter ±15° model.

Training and validation (use fresh candidate/output paths to retrain without overwriting):

```bash
python -m research.studies.tilt_augmentation --stage train --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.tilt_augmentation --stage cv --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.tilt_augmentation --stage evaluate --letters G H --axis y --degrees 0 -90 90 180 --out results/rotation_GH_01 --candidate models/asl_shin_angles_GH_rotation.joblib
python -m research.studies.report_GH_rotation
```

See `results/rotation_GH_01/report.md` for all-letter metrics and replay of previously selected errors. Those known webcam errors are diagnostic examples, not an independent G/H test.

## Live overlays and capture rounds

Hand landmarks and connections are visible by default. **L** toggles the overlay; raw camera and detector-input files remain unchanged. The overlay follows the mirrored, letterboxed preview and is included in the saved preview.

Every launch starts a uniquely identified round. **New round / N** starts another during live viewing, without restarting the camera. Switching models stays in the same round. Use `--round-name "GH-back-of-hand"` for a descriptive label. The round label and number are visible below the model buttons; each new `capture.json` contains `capture_round` (ID, name, number, start timestamp), `session_id`, overlay state and the active model path/hash. Capture folders remain under `captures/confusions/`; group them by round ID for analysis. Older captures retain their original session IDs and are not modified. Nothing is saved until a photo is explicitly confirmed. Guided protocols keep their original planned session identity and lock the model/new-round controls.

## Projected image angles alongside 3D angles

One fixed, unaugmented candidate adds the same 14 joint triplets as internal 2D pixel angles to the existing 835 inputs, for 849 total. Projected MCP angles remain proxies and projected angles are not anatomical ground truth. Pixel coordinates use the exact detector image width/height, stored as `image_size`; raw normalized x/y would distort nonsquare images. The same vectorized feature function handles training and live inference. Geometry is reconstructed from the frozen manifest; the landmark detector is not rerun. Median imputation and standardization of all 29 angle inputs are fitted on each training fold only. SVM parameters remain C=10, RBF, gamma=scale, balanced classes.

```bash
python -m research.studies.compare_image_angles --stage all
python -m research.studies.report_image_angles
python -m research.live_camera --model models/asl_shin_angles_2d.joblib --round-name "2D-3D-test"
```

The all stage resumes the fixed protocol/model and finished folds. `--stage train` refuses to overwrite an existing candidate; `--stage cv` reuses fold checkpoints; `--stage evaluate` replays saved captures. The small new feature cache is `results/angles_2d_01/pixel_angles.npz`; the original dataset, cached inputs and earlier models remain unchanged. The native tool exposes the candidate through **Image + depth [6]**. Snapshot exports contain all 849 feature names/values, 15 original `joint_angles_degrees` and 14 `projected_angles_degrees`, detector image dimensions, active model hash and round metadata.

Five-fold pooled accuracy changes 97.74776% to 97.69298%; macro-F1 .97524391 to .97465145. There are 70 corrected and 103 newly incorrect predictions. Four folds decline slightly; one improves. Of the latest 11 deliberately selected G/H-model errors, World joints recognizes 2 and the candidate 3 (T is newly correct; G remains A). The older C/D/O development round changes 17/27 to 16/27 correct. These captures were not used to fit or tune the model and are reused development diagnostics, not independent test scores. Full fold and all-letter metrics are in `results/angles_2d_01/report.md`.

Five folds preserve the earlier comparison and provide large validation partitions. More folds cannot repair overlap between people/sessions: the current groups are filename blocks, not verified signer identities. Report the pooled out-of-fold scores and per-fold variation, without treating fold standard deviation as a confidence interval. A new signer/session held out from development is needed to assess live robustness.

## G/H-only diagnostic classifiers

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.compare_gh_classifier
python -m research.studies.gh_middle_diagnostic
```

The first command trains three fixed binary RBF SVMs using only 15 3D measurements, 14 projected 2D angles, or all 29. It restricts the original five group folds to the same 4,801 G/H images (2,430 G; 2,371 H), with training-only imputation/scaling. Captures do not enter fitting. Bundles are saved under `models/gh_diagnostic/` and results under `results/gh_classifier_01/`. These diagnostic bundles always choose G or H; they are not compatible replacements for the 26-letter live models.

Binary CV accuracy is 96.563% (3D), 95.793% (2D), and 96.584% (both). The latest saved G is still H for all three. G is nevertheless well recognized in the original 26-class validation: 2,416/2,430 correct (99.424% recall); World joints mistakes four G images for H, nine for J and one for F. These are accepted-image, filename-block validation results, not new-signer webcam scores.

The second command is an explicitly post-hoc diagnostic using only the 3D/2D middle-PIP measurements. It recognizes the three stored G/H error examples correctly, but CV accuracy is lower (94.085%, macro-F1 .94084). Results are in `results/gh_middle_01/`; do not treat those reused examples as an independent test or install this binary rule as a general classifier.

For visual inspection, `results/gh_classifier_01/G_training_examples.png` shows 24 accepted G examples, and `G_training_files.txt` maps all 2,430 original G filenames to their existing image paths. The source images remain in the single dataset at `data/asl_kaggle/images/`.

## Live G/H-only diagnosis

```bash
conda activate ktai-hand-gesture
python -m research.live_gh --variant middle
```

This separate native window always chooses G or H; other letters are outside its scope. Buttons/keys 1-4 select all 3D angles, all 2D angles, both, or middle-PIP-only. L toggles the landmark overlay; Q/Esc closes. Close the ordinary camera window before starting this one. The selected development examples motivated the middle-only model; recognizing them does not establish independent live accuracy. The ordinary nine-model tool remains `python -m research.live_camera`.

## Guided G/H viewpoint collection

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture
```

Three participants, G/H, four views (palm, oblique palm, oblique back, back), five repetitions: **120 planned photos, 40 per person**. This supersedes the earlier 60/90-photo three-view proposal. Predictions are hidden; participant transitions require Enter. Use `--participants 2` for 80 photos. Photos are separate from training. P01 is development; P02/P03 are reserved evaluation. Full instructions, the four-view illustration and resume/status commands are in [docs/GH_CAPTURE.md](docs/GH_CAPTURE.md).

## Core-question report revision

The ICML-style report is a Dutch group draft for readers without ML background: RQ1 concerns preprocessing/detection recovery, RQ2 training-set size and RQ3 explicit features. RQ1/RQ2 contributions remain TODOs. The main text explains the pipeline and evaluation with examples and uses exact correct counts in `plain_results_nl.tex`. The Dutch appendix now contains only six referenced sections: A data/settings, B feature results, C featuretool, D mirroring, E standalone G/H rotation and F handedness normalization. Its selected tables come from the same frozen results via `research.studies.compact_appendix_tables`; hashes are in `paper/appendix_table_sources.json`. Earlier detailed sources remain archived in the repository and are not included in the compiled appendix. Augmentation studies remain separate supporting evidence. `paper/INTEGRATION_TODO.md` lists the handoff. Regenerate the existing detailed feature tables/plots from frozen results with:

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.paper_core_assets
tectonic -X compile paper/report.tex
tectonic -X compile paper/appendix.tex
```

`paper/core_results.tex`, `paper/core_letter_rows.tex` and `paper/figures/core_*.pdf` are generated from the saved primary results. The current rubric review is `paper/RUBRIC_REVIEW.md`; its point estimates are provisional, not an official grade. Compiling figures does not retrain any model.

## Named alphabet test with an example for every letter

```bash
conda activate ktai-hand-gesture
python -m research.alphabet_capture
```

The participant form records name/code, development/new-person status, ASL fingerspelling experience and hand. Five A-Z rounds give 130 photos per person. A fixed dataset example appears beside the camera; **Space saves directly**, **H** marks extra guidance, and Q/Esc pauses. Predictions are hidden. The same stored images can later be scored by both frozen models, separately for new and development participants. See [the complete instructions](docs/ALPHABET_CAPTURE.md), including resume commands and guidance rules. Dataset examples are not independently validated ASL instructions; this is an example-guided static reproduction task.

## Same-coordinate direction-versus-joint-angle experiment

This experiment separates the type of angle from the coordinate source. All new features use the exact normalized image x/y/z landmarks used by the Shin reference; they do not use world landmarks or aspect-correct pixels. No images or landmarks are redetected. It uses the historical 60,251 accepted images and five frozen folds.

```bash
conda activate ktai-hand-gesture
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.compare_direction_joint
```

For both SVM and LightGBM, compare:

| Variant | Features | Input count |
|---|---|---:|
| `shin` | 190 distances + 630 axis-relative directions | 820 |
| `distance_joint` | Same 190 distances + 15 explicit joint measurements | 205 |
| `shin_joint` | All 820 Shin values + the same 15 joint measurements | 835 |

The 15 measurements retain the earlier definitions: 14 interior joint angles plus signed thumb MCP deviation. Finger MCP angles use wrist-based geometric proxies. In this normalized coordinate system they are descriptors, not calibrated anatomical angles. `shin_world` is the previously evaluated Shin-plus-world-angle arm, reused as context rather than a same-coordinate comparison.

This is a control within the interpretable-feature investigation. Shin already encodes the adjacent segment directions from which joint bending can be derived. The study tests whether making that relation explicit helps a fixed learner; it does not claim new sensing information. Categorical flexion is also derived from angle thresholds. The world-joint SVM gains 67 net correct images; same-coordinate addition gains only four, and replacement loses 1,883. The original full package does not improve SVM. The paper retains these negative results rather than treating joint measurements as a generally improved recognizer.

Source: `research/joint_angles.py`; experiment runner: `research/studies/compare_direction_joint.py`. Baseline and world-angle OOF predictions are reused only after checking source hashes, image IDs, labels and folds. Each added angle block is median-imputed and standardized using its training partition only. Fixed classifier parameters match the project profile; there is no fresh hyperparameter tuning. In particular, RBF `gamma=scale` adapts to each representation. Conclusions therefore concern these fixed recipes, not optimally tuned upper bounds for each representation.

Results, per-letter precision/recall/F1, confusion matrices, paired corrected/lost counts, figures and an English interpretation are saved under `results/direction_joint_01/`. Models and resumable prediction checkpoints are under `runs/direction_joint_01/` (ignored by Git). Running the same command resumes completed folds; different inputs require fresh `--out` and `--work` directories. Both algorithm selections must remain the same when resuming. This experiment does not replace any live model. Test angle geometry with `python -m unittest tests.test_joint_angles -v`.

After both classifiers finish, independently verify every reported metric, all per-letter scores and confusion counts, training-only angle scaling for all 20 new models, and a replay sample from every held-out fold:

```bash
python -m research.studies.verify_direction_joint
```

This writes `results/direction_joint_01/verification.json`, including runtime versions and code/result hashes. It requires the local fold models in `runs/direction_joint_01/`.

## Standalone G/H y180 experiment

The isolated experiment uses the historical 60,251 accepted original images and five filename-block folds. A freshly fitted original-only SVM is compared with one additionally trained on G/H-only 180-degree camera-y turns. Originals and turns each have weight 0.5 for G/H; other sources have weight one. The candidate shares original-fit preprocessing, resolved gamma and class priors. It has 65,052 fitting rows and no mirrors, camera calibration or routing gate. Validation uses unchanged originals. Both arms remain 26-class classifiers.

```bash
python -m research.studies.gh_rotation_standalone --stage freeze
python -m research.studies.gh_rotation_standalone --stage cv
python -m research.studies.gh_rotation_standalone --stage train
python -m research.studies.report_gh_rotation_standalone
python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib
```

Compare **World joints [2]** with **G/H y180 only [9]**. Evidence is in `results/rotation_GH_y180_01/`, including every prediction, both arms' fold metrics, all 26 letter scores, paired changes, confusions, selected camera replay and verification. The fresh reference exactly reproduces all historical held-out predictions. The candidate corrects 4 errors and loses 5 (net -1); macro-precision changes 97.53902% to 97.53874% and macro-F1 0.97524391 to 0.97523305. This gives no overall benefit. The selected back-view H is recovered but both known back-view G examples remain wrong. No selected capture enters fitting or constitutes an independent test. Appendix E reports this separately from mirror augmentation and the earlier combined candidates.

## Earlier combined G/H adaptation

The earlier combined follow-up is described in `docs/GH_VERTICAL_ADAPTATION.md`, with results in `results/gh_vertical_01/`. Load it explicitly with `--model models/asl_gh_vertical_adapter.joblib`; button 9 now selects the standalone original-data experiment. The combined candidate adds G/H-only turns to the mirror-trained reference and reuses three saved back-of-hand images as calibration training. Those images cannot provide independent validation. The earlier index-axis roll remains a separate development artifact in `results/gh_view_adaptation_01/`; it is not the vertical turn.

```bash
python -m research.studies.gh_vertical_adaptation --stage prepare
python -m research.studies.gh_vertical_adaptation --stage train
python -m research.studies.gh_vertical_adaptation --stage cv
python -m research.studies.check_gh_vertical_only
python -m research.studies.report_gh_view_adaptation --study gh_vertical_01
```

See `docs/GH_ROTATION.md` for the distinction between the existing feature-space G/H candidate and a prospective RGB-image rotation experiment. The latter requires new landmark detection, fixed source-image groups and a separate training-only augmentation path. Neither the same-coordinate study nor the paper revision runs that new experiment or changes the default live model.

## Horizontal RGB mirror experiment

The full protocol is `docs/MIRROR_EXPERIMENT.md`; generated results are `results/mirror_01/report.md`, and Appendix D gives the paper version. All 78,000 A-Z originals and their mirrors are redetected, with recorded model hash, IMAGE mode, max two hands and 0.5/0.5 thresholds. Source pixels and frozen filename-block folds are checked. Mirrored derivatives never cross the source-group boundary.

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.mirror_dataset --workers 4
python -m research.studies.mirror_augmentation --stage cv
python -m research.studies.mirror_augmentation --stage train
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.report_mirror
python -m research.studies.verify_mirror
```

The test mixture has one view per source, with half mirrored within each letter/fold using seed 20261106. Its exact choices are in `test_mix.csv` and are shared by the two arms. Each original fitting source retains total weight one; only an accepted mirror of that same source is added. Training-only scaling, SVM gamma and original class priors are held equal. Both fixed SVM/LightGBM recipes are evaluated without tuning. Failures remain no predictions in all-attempt metrics. `metrics.csv` additionally contains accepted-only and paired-both-accepted scopes, all five folds, macro-precision/recall/F1 and accuracy. Confusion matrices retain the no-prediction column. Detector-labelled handedness metrics do not establish true left/right performance.

Chunk checkpoints and fold models are under ignored `runs/mirror_01/`. The extraction and CV stages resume matching work. Full-data training refuses to overwrite its new artifacts. It creates `models/asl_shin_angles_mirror_baseline.joblib` and `models/asl_shin_angles_mirrored.joblib`, exposed by native buttons/keys **7** and **8**. These match the unchanged live 835-input encoder; the prior default remains World joints. Start with `python -m research.live_camera --model models/asl_shin_angles_mirror_baseline.joblib` for a matched comparison.

The historical manifest labels 57,424/60,251 accepted images Right (95.31%) and 2,827 Left (4.69%), with 17,749 sources not accepted. Those estimates describe the depicted hand, not the person's dominant hand. Fresh original detections are used for the current baseline so that unknown historical thresholds are not silently mixed with the new mirrors. The study is a synthetic reflection test; separately collected left-handed signers are needed for prospective validation.

## Bijlagebeelden en opgeslagen featurecontrole

De hoofdtekst en de volledige opgeschoonde bijlage A–F zijn Nederlands. Alleen onderdelen waarnaar de hoofdtekst verwijst worden opgenomen. De bijlage veronderstelt kennis van precisie en F1 en geeft de technische uitwerking. De actuele pagina-aantallen staan in `paper/team_draft_verification.json`. Vijf pagina's geldt voor de hoofdtekst; referenties mogen op pagina zes staan.

```bash
python -m research.studies.compact_appendix_tables
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.appendix_visuals
python -m streamlit run app.py -- --capture captures/confusions/20261031T152116143793Z_11756e66
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

De figuurgenerator vereist de vier gebruikte lokale cameraregistraties, de dataset, de bestaande spiegelcache en de referentie-/G/H-y180-modellen. Er wordt niet opnieuw gedetecteerd of getraind. De eerdere rapportweergaven blijven als bronmateriaal beschikbaar; bijlage C toont nu de door de gebruiker aangeleverde screenshot in `paper/figures/feature_tool_screenshot.png`, met twee als Nederlandse vectortekst heropgemaakte detailkaartjes en een vectoroverzicht van de kenmerken. Het bronbestand is ongewijzigd; bij vervanging moeten bijschrift en overgenomen kaartinhoud worden gecontroleerd. PNG's, featurewaarden, selectiebeleid en hashes staan in `results/appendix_visuals_01/`; PDF-figuren staan in `paper/figures/`. De synthetische spiegelvoorbeelden gebruiken opgeslagen held-out voorspellingen. Zie `docs/APPENDIX_VISUALS.md`.

## Canonical detector-Right normalization (standalone comparison)

This study reuses the completed `runs/mirror_01/detections/` cache and the original-only fold estimators; it does not redetect RGB images or add mirror rows to canonical training. Keep the existing mirror evidence intact. The candidate shares each original-only fold's preprocessing, resolved SVM gamma and class priors. Each independently detected test view is normalized using its own detector handedness.

```bash
conda activate ktai-hand-gesture
python -m research.studies.handedness_normalization --stage geometry
python -m research.studies.handedness_normalization --stage cv --model svm
python -m research.studies.handedness_normalization --stage cv --model lightgbm
python -m research.studies.handedness_normalization --stage train
python -m research.studies.handedness_normalization --stage export
python -m research.studies.verify_handedness
python -m research.studies.report_handedness
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

`results/handedness_01/protocol.json` freezes code, detection-cache and split hashes before fitting. Fold estimators/checkpoints live under `runs/handedness_01/`; all predictions, metrics, per-letter scores, paired changes and verification live under `results/handedness_01/`. Generated `paper/handedness_*_nl.tex` extracts contain only verified measurements. Main text and appendix F distinguish canonicalization from RGB-mirror augmentation and G/H rotation. The candidate is selectable in the native camera as **Right normalized [0]**, or via `--model models/asl_shin_angles_right_hand.joblib` in both camera and featuretool. Legacy bundles omit the normalization flag and retain their original input path.

The fixed mix is half original RGB and half reflected RGB per letter/fold; it is not a verified equal mix of real human left/right hands. Rejected detections remain false negatives in all-attempt scores. Accepted-only and both-accepted-source results are reported separately. No independent signer evaluation is implied by repeated filename-block validation.

### Primary reporting denominator for mirroring and normalization

The paper and appendices D/F now compare classification **only on accepted detections** (`scope=accepted_only`), consistent with the original feature study. The primary normalization baseline is the SVM 835-input model on 60,251 original accepted images: macro-precision 97.53902%, macro-F1 0.97524391. Original, reflected and mixed test presentations are reported separately; the mixed baseline is not the original-image baseline. Within a presentation all arms use exactly the same accepted test IDs.

`research.studies.report_handedness` and `research.studies.compact_appendix_tables` select these existing verified rows. All-attempt results remain secondary in the study report and raw CSVs. Since rejected detections have no predicted letter, excluding them changes recall/F1 but not precision. No models, predictions or train/test assignments were changed for this reporting revision.

### Feature illustrations and the supplied screenshot

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.feature_illustrations
tectonic -o paper paper/appendix.tex
```

This generates the explanatory vector diagram `paper/figures/feature_concepts_nl.pdf` and its provenance in `results/feature_illustrations_01/`. The diagram uses schematic drawing coordinates, not measured hands. The unchanged user-supplied screenshot (7 November 2026) is versioned as `paper/figures/feature_tool_screenshot.png`; LaTeX displays the full image and two clipped detail views without changing that PNG. Appendix C contains 19 workbook checks (13 flexion, 2 spread, 1 crossing, 1 thumb shape, 2 contacts). The thumb-over-palm idea is explicitly distinguished from its implemented signed-MCP-angle proxy. This is an illustration-only update with no new training, predictions or feature-rule changes.

## Eight-variant feature study and held-out permutations

This is the current primary RQ3 study, replacing the earlier 931-input full-package comparison in the paper. Historical experiment files remain intact. `research/feature_study.py` defines D (190 distances), S (630 axis-direction cosines), E (70 one-hot categorical values for 19 own checks), and J (15 world-joint measurements). The eight representations are DS, D, DSE, DE, DSJ, DJ, DSEJ and DEJ. Columns 890:916 (26 candidate-letter workbook-match scores) are explicitly excluded. D is always present.

```bash
conda activate ktai-hand-gesture
python -m research.studies.feature_factorial --stage parity
python -m research.studies.feature_factorial --stage train --model svm
python -m research.studies.feature_factorial --stage train --model lightgbm
python -m research.studies.fast_factorial_permutation
python -m research.studies.feature_factorial --stage permutation --model svm
python -m research.studies.feature_factorial --stage permutation --model lightgbm
python -m research.studies.feature_factorial --stage export
python -m research.studies.verify_feature_factorial
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.report_feature_factorial
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

`--stage train` and `--stage permutation` can be run separately; `--variants` and `--folds` select disjoint work items for parallel execution. `--stage run` trains its requested models first and then permutes their held-out inputs. All outputs resume from checkpoints under `runs/feature_factorial_01/`; numerical evidence and the protocol live in `results/feature_factorial_01/`. Use new paths/code version for a changed protocol; the frozen source/input hashes intentionally reject silent changes.

Classification uses all 60,251 accepted original sources, with the existing five filename-block folds. All 80 fold estimators use the same fixed model recipes; only J is imputed and standardized using its training fold. The DS and DSJ predictions must reproduce the historical 820/835-input predictions exactly. The current translated workbook has different bytes from the original but the category layout is checked by recomputing 390 raw cases and checking every one-hot group across the complete cache. It supplies feature order/options, not a true letter label.

Permutation uses a separately frozen subset: 100 held-out images per letter/fold (13,000 total), shared by every variant and learner. Five seeded row permutations move every column of one group together; labels and the fitted model stay unchanged. Present D/S/E/J blocks are tested for all eight variants. The 19 individual categorical checks are tested in DSE/DE without J; five E families and five J finger families are tested in DSEJ/DEJ. This gives 3,900 fold/group/repeat interventions. CSVs retain macro-precision and macro-F1 drops; paper figures show F1 drops in percentage points. Spread over folds/repeats is descriptive, not an independent confidence interval.

`predictions.csv`, `metrics.csv`, `per_letter.csv`, `paired_changes.csv` and `confusions.csv` preserve the full accepted-image evaluation. `factor_effects.csv` gives the four contexts for adding each optional block. `permutation_selection.csv`, `permutation.csv` and `permutation_summary.csv` preserve selection and all hussel results. The verifier independently reconciles scores, checks train-only scalers, hashes and shared permutation orders, replays saved fold models and selected interventions, and checks historical DS/DSJ parity. `paper_extracts.json` records the generated TeX/figure hashes.

The same-coordinate joint-angle controls remain separate supporting evidence. E/J use world landmarks while D/S use image landmarks; neither factorial nor permutation results establish an independent sensor contribution or causal feature importance. All folds were repeatedly inspected during development and are not verified participants. The existing live and augmentation models retain their recorded protocols.

The optional optimized SVM route (`fast_factorial_permutation`) fills the same whole-group checkpoints using float64 RBF distance decomposition and the fitted LIBSVM coefficients. It verifies every selected baseline prediction against ordinary-SVC OOF predictions and replays perturbed cases; near-zero pairwise margins use ordinary prediction. `execution_backend.json` freezes this execution addendum and `kernel_verification.json` records parity. No models, sampling, metric definitions or permutations change. Omitting the optimized step gives the ordinary prediction route with identical experimental settings, at higher computation cost.

## G/H capture reset and larger Dutch guidance (7 November 2026)

The two rejected sessions and 22 attempts were withdrawn from active capture folders and the paper. They and their derived results/figure remain recoverable under ignored `runs/discarded_gh_20261107/`; the manifest and original hashes are in `results/gh_capture_reset_20261107/withdrawal.json`. The active session locations contain withdrawal markers, and both resuming and new writes are blocked. Their camera-example exporter also refuses to regenerate withdrawn evidence.

Start a fresh personal round with `python -m research.gh_view_capture --participants 1`. The 1160×752 native panel keeps the exact 640×480 preview and original model inputs, with large Dutch instructions and a schematic palm-normal arrow. New plans record `explicit-gh-pose-v3` and `large-nl-gh-guidance-v1`. One, two and three participants give 40, 80 and 120 planned photos. See `docs/GH_CAPTURE.md`.

## G/H-afdekking: verkennende volumemodellen en puntverstoring

Deze aanvullende proef gebruikt alleen de bestaande trainingslandmarks. `research/synthetic_hand.py` bouwt een grof palm-/vingervolume met dieptebuffer. Het is geen anatomische passing. Een vaste selectie van 64 oorspronkelijke G/H-foto's wordt vergeleken met hun renders bij 0 en 180 graden. MediaPipe accepteert 64/64 bronfoto's en 0/64 renders bij elk aanzicht. Deze route levert dus geen nieuwe detecties voor training op; de bekende bronpunten worden niet als detectoruitvoer gepresenteerd.

Een afzonderlijke proxy voegt alleen aan vermoedelijk afgedekte vingerpunten in de geroteerde beschrijvingen normaalverdeelde verstoring toe, met standaardafwijking 2% of 5% van de pols–middelvingerknokkelafstand per coördinaat. Beide sterktes gebruiken hetzelfde reproduceerbare bronpatroon. De modellen zijn opnieuw getraind op de vijf bestaande folds, met de oorspronkelijke train-only preprocessing, gamma, klassegewichten en hetzelfde totale gewicht per bron. Dit modelleert onzekere handpunten, geen empirisch gemeten afdekkingsfouten. Er worden geen ingetrokken webcamopnamen gebruikt en geen live-modellen vervangen.

```bash
python -m research.studies.gh_occlusion --stage prepare
python -m research.studies.gh_occlusion --stage screen
python -m research.studies.gh_occlusion --stage train --arm noise002
python -m research.studies.gh_occlusion --stage train --arm noise005
python -m research.studies.gh_occlusion --stage report
python -m research.studies.gh_occlusion_rotated_test
python -m research.studies.verify_gh_occlusion
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.report_gh_occlusion
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

`screen` gebruikt de bestaande lokale MediaPipe-asset en de grafische diensten van macOS. `prepare` controleert voor alle 4.801 G/H-bronnen dat nulverstoring de historische rotatie reproduceert. De bevroren protocollen en CSV's staan in `results/gh_occlusion_01/`; features en tien nieuwe foldmodellen in `runs/gh_occlusion_01/`. Gewijzigde broncode/inputhashes worden geweigerd: gebruik voor een gewijzigde methode een nieuwe run.

De eerste toets gebruikt alle 60.251 ongewijzigde, geaccepteerde foto's en meet behoud van datasetclassificatie. Dat is geen handrugtoets en bevestigt de ASL-uitvoering van de labels niet. De aanvullende kunstmatige toets vervangt uitsluitend de 4.801 G/H-toetsbeschrijvingen door hun y180-rotatie en houdt de overige letters gelijk. Ze meet het leren van de toegevoegde geometrie. Ze omvat geen nieuwe RGB-waarneming, nieuwe detectiefout of onafhankelijke handrug-/persoonstoets. Beide toetsen en beide ruissterktes worden gerapporteerd; er wordt geen beste variant achteraf als live-standaard gekozen.

### Vier nieuwe vrije G/H-voorbeelden

```bash
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.gh_live_diagnostics
tectonic -o paper paper/report.tex
tectonic -o paper paper/appendix.tex
```

De runner gebruikt vier expliciete bron-ID's uit `captures/confusions/` van 7 november 2026; die lokale bestanden zijn nodig bij reproductie. De 22 ingetrokken begeleide opnamen worden niet gebruikt. Hij controleert beeld-/modelhashes, de opgeslagen inputvector en voorspelling en voert dezelfde opgeslagen landmarks door de referentie en de rotatievariant. Er wordt niet opnieuw gedetecteerd of getraind. `results/gh_live_diagnostics_20261107/` bevat het bevroren protocol, alle vier voorspellingen, de hoekvergelijking met de bestaande trainingscache en verificatie. Bijlage E.4 toont de vier handuitsneden met vectorlandmarks. De bedoelde letters zijn gebruikersannotaties en de aanzichten zijn visueel beschreven; dit zijn geen ASL-expertlabels of representatieve precisie-/F1-resultaten.
