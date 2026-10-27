# Workflow — training data → finger inspection → two models → comparison

The separate components share one dataset. You do not need to record videos:
the webcam saves individual photos. The classifier learns letter labels from
the landmarks in your own photos. The supplied drawings are recording guides.

A possible research question:

> Does adding explicit finger angles, finger states and relative distances
> improve recognition of NGT letter poses compared with the same classifier
> receiving only normalized hand landmarks?

Start with eight letters to learn the workflow, then expand to 26.
Photographs measure hand poses, not the complete execution of moving letters.

## 0. Extract and install

Extract `ktai-hand-gesture.zip`. Open Terminal in **ktai-hand-gesture**, the
project root containing `apps/`, `scripts/` and `requirements.txt`. This guide
is stored in `docs/`; run every command below from the project root.
If your terminal is in the directory containing the extracted project:

```bash
cd ktai-hand-gesture
conda env create -f environment.yml
conda activate ktai-hand-gesture
```

If the environment already exists:

```bash
conda activate ktai-hand-gesture
python -m pip install -r requirements.txt
```

## 1. Initialize a dataset with alphabet references

```bash
python -m scripts.init_dataset --project data/ngt
python -m streamlit run apps/collect_app.py -- --project data/ngt
```

The separate capture app opens in your browser. It shows a crop for each letter
of the Dutch manual alphabet. The dataset initially contains no training photos;
initialization never adds drawings as training samples.

For older ASL recordings, check the poses before treating them as NGT data.
Refresh supplied references with:

```bash
python -m scripts.init_dataset --project <dataset-directory>
```

See `assets/SOURCES.md` for the image source and reference migration behavior.
To reopen a dataset export, extract it and replace `data/ngt`
in commands with the path to its `dataset` directory. Original photos and
landmarks remain usable; the new scripts recompute features from those
landmarks. Work on a copy if you want to preserve the original dataset.

## 2. Plan and record your photos

Choose **Guided capture → Start with 8 letters**. The initial selection is
`A B C D E F K L`. Edit the list or choose **All 26 letters**.
Use the same selected letters in every round.

| Round | Usage | Photos per selected letter | Purpose |
|---|---|---:|---|
| 1 | development / Development | 10 | Initial poses |
| 2 | development / Development | 10 | Independent second round |
| 3 | development / Development | 10 | Third round for 3-fold cross-validation |
| 4 | test / Separate final test | 10 | Reserve until all choices are fixed |

This produces 320 photos with eight letters or 1,040 with 26 letters when all
rounds are completed. This is a starting experiment, not a guarantee of adequate
data or accurate recognition. Check usable hand detections too. More nearly
identical photos from one burst do not replace new sessions.

Example settings:

- **Participant:** `participant1`.
- **Condition:** `baseline`.
- **Photos per gesture:** `10`.
- **Countdown for next gesture (s):** `5`.
- **Time between photos (s):** `3`.
- **Usage:** `Development` for the first three rounds.

Click **Open webcam and start new session**. A separate local camera window
shows the reference on the left and webcam on the right. It counts down, saves
a photo, and moves to the next letter after the selected photo count.

Keyboard controls: **SPACE** start/pause, **+ / −** longer/shorter intervals,
**R** restart countdown, **Q** close. Reset your hand pose between photos.
Take a break and reposition yourself between sessions. Give every participant
a distinct name. Each camera start creates a unique session name.

Keep conditions reasonably similar for the initial comparison. Use **Separate final test**
for round four and exclude those images from threshold, feature and model tuning.

Mirroring changes the display only; stored analysis images remain unmirrored.
Source images, labels, sessions and landmarks are retained. Live viewing does
not save video. Guided capture waits for a usable hand detection before saving,
so these photos are already selected for detectability. To measure detector
failures too, collect independent test photos using a regular camera app and
upload them unfiltered with role `test`. Document this recording method.

### Record directly from Terminal

This opens the same guided camera window. Repeat the development command for
each round; each invocation creates a new session name.

```bash
python -m apps.webcam record --project data/ngt --labels A B C D E F K L --photos 10 --interval 3 --change-delay 5 --participant participant1 --condition baseline --role development
```

For the reserved final test:

```bash
python -m apps.webcam record --project data/ngt --labels A B C D E F K L --photos 10 --interval 3 --change-delay 5 --participant participant1 --condition baseline --role test
```

## 3. Check the dataset

Open **Check dataset** in the capture app. For each letter, check:

- Total photos and development photos with usable detections.
- Three genuinely distinct development sessions.
- Separate test photos for every selected letter.
- Correct label, participant, session and condition metadata.

Correct incorrectly saved metadata. Do not invent sessions to bypass an error:
photos from one actual recording round belong together. This prevents nearly
identical photos from appearing in both training and validation.

Guided capture analyzes photos immediately. Analyze uploaded photos with the
app button or:

```bash
python -m scripts.analyze_dataset --project data/ngt
```

Inspect detection failures. Correct or exclude a demonstrably incorrect label
or failed recording in advance. Do not remove difficult test photos because
the classifier predicts them incorrectly. The final test counts missing hand
detection as a failure of the complete recognition pipeline.

Stop the capture app with **Ctrl+C** in Terminal before continuing.

## 4. Inspect finger features separately

```bash
python -m streamlit run apps/feature_check_app.py -- --project data/ngt
```

This second app does not train a letter classifier. It uses development photos
only to investigate the feature code in `gesturelab/features/finger_features.py`.
**Code and definitions** shows the actual Python implementation.
`FINGER_FEATURES.md` explains the formulas, default thresholds and limitations.

In **Visual inspection**:

1. Assess each finger yourself: extended, bent, intermediate or unscorable.
   Do not guess the state of an occluded finger.
2. Save your assessment.
3. Reveal landmarks and computed measurements. Compare angles and straightness
   with the visible pose. A straight joint has an interior angle of 180°.
4. Seek varied examples: open and closed fingers, different thumb positions,
   ambiguous cases, multiple letters, sessions and camera positions.
5. Investigate errors: incorrect landmarks, ambiguous human assessment or
   unsuitable thresholds? Change only what you can justify.
6. Adjust thresholds if needed and click **Save thresholds**.

Start with approximately 40–60 varied photos. Include positive and negative
examples for each finger state. Do not derive human assessments from the code's
answer or solely from the letter label.

**Precision and recall** evaluates two binary tasks per finger: extended? bent?

- **Precision = TP / (TP + FP):** fraction of positive predictions that are correct.
- **Recall = TP / (TP + FN):** fraction of actual positives that are found.
- Intermediate is negative for both tasks.
- Unscorable assessments are skipped and counted.
- Missing detection is a false negative when the human assessment is positive.
- A blank score means a zero denominator; collect suitable examples.

Define and justify your own acceptance criteria in advance. For example, high
precision for index-finger extension may matter if false extensions confuse
letters. Always report sample counts, TP, FP and FN. High precision can coexist
with low recall.

This is a development diagnostic because these assessments also guide code
changes. Do not present it as an independent final test. The table checks ten
binary states, not all 81 derived features. Inspect continuous angles and
distances visually and through calculation tests.

Stop the app and export the measurements:

```bash
python -m scripts.evaluate_features --project data/ngt --out results/feature_check
python -m unittest discover -s tests -p test_finger_module.py -v
```

Outputs: `finger_precision_recall.csv`, `finger_predictions.csv` and
`feature_audit.json`. Calculation tests check known angles, boundaries and
feature columns; they do not prove MediaPipe estimates every real hand correctly.
Record findings and changes for your report.

## 5. Freeze one shared split

Close both apps. Fix the feature code, thresholds and selected letters.
Create one experiment directory:

```bash
python -m scripts.prepare_split --project data/ngt --out experiments/ngt_01 --folds 3 --grouping session --model svm --labels A B C D E F K L
```

Omit `--labels A B C D E F K L` to use every recorded development label.
The experiment freezes:

- Identical photo IDs and grouped validation folds for both models.
- Separate test photos from other sessions.
- Classifier, fixed random seed, feature code and thresholds.
- Dataset checksums to detect changes that invalidate the comparison.

In **3-fold cross-validation**, different development sessions are validated in
turn while the others are used for training. Classifier scaling is fitted within
each training fold. Reserved test data is not used.

With only two development sessions per letter, use `--folds 2`. A separate test
session is still required. Two sessions total, including one test session, are
not enough. For a software trial without a final test, explicitly use
`--allow-no-test`.

With `--grouping participant`, evaluate generalization to other people. You need
at least as many development participants per letter as folds, plus separate
test participants. All sessions of one person then form one group.

## 6. Train the two models separately

For the feature code locations and instructions for adding input variants, see
[TRAINING_INPUTS.md](TRAINING_INPUTS.md). The commands below retain the original
126-versus-207-input comparison.

Without additional finger features:

```bash
python -m scripts.train_without_features --experiment experiments/ngt_01
```

With additional finger features:

```bash
python -m scripts.train_with_features --experiment experiments/ngt_01
```

| Variant | Inputs | Values |
|---|---|---:|
| Without finger features | 21 normalized image landmarks × 3 + 21 normalized world landmarks × 3 | 126 |
| With finger features | Same 126 coordinates + 81 derived geometric features | 207 |

Both variants receive the same source information. Extra calculations include
angles, extension/bending states, fingertip distances, thumb position, spread
and local depth. MediaPipe's 3D landmarks are estimates, not independent depth
camera measurements.

Each script runs cross-validation and then trains a final model on all
development photos. It does **not predict test photos yet**. Models are saved at:

```text
experiments/ngt_01/without_features/model.joblib
experiments/ngt_01/with_features/model.joblib
```

Each variant also saves `training_features.csv`, `cv_folds.csv`,
`cv_predictions.csv` and `training.json`. Inspect these to verify features and
scores. `split.json` contains training, validation and final-test photo IDs.

## 7. Compare development results first

```bash
python -m scripts.compare_results --experiment experiments/ngt_01
```

Open `experiments/ngt_01/comparison_cv/report.html` in a browser. On macOS:

```bash
open experiments/ngt_01/comparison_cv/report.html
```

The report compares accuracy, macro-F1, recall by letter, confusion matrices,
coverage and classifier timing. Scores and prediction CSVs allow you to trace
errors back to photos. Synthetic projects are explicitly marked as software
tests; their scores are not real recognition results. Error bars and a small
score difference do not establish statistical significance.

Investigate errors before changing anything: labels, landmark quality, ambiguous
poses, class balance, sessions, thresholds and model settings. Select methods
using development CV and record your choice before viewing the final test.

To compare another classifier, create a new experiment for both variants:

```bash
python -m scripts.prepare_split --project data/ngt --out experiments/ngt_rf --folds 3 --model random_forest --labels A B C D E F K L
python -m scripts.train_without_features --experiment experiments/ngt_rf
python -m scripts.train_with_features --experiment experiments/ngt_rf
python -m scripts.compare_results --experiment experiments/ngt_rf
```

After data or feature-code changes, create a new experiment such as `ngt_02`.
Do not edit frozen checksums to force an old experiment to continue.

## 8. Evaluate both fixed models on the final test

Run only after all choices are fixed:

```bash
python -m scripts.compare_results --experiment experiments/ngt_01 --final-test
```

Open `experiments/ngt_01/final_test/report.html`. On macOS:

```bash
open experiments/ngt_01/final_test/report.html
```

Both models are evaluated on exactly the same reserved photos. Missing hand
detection produces unknown and counts as an error. Coverage is the fraction
of photos for which a known letter label could be predicted.

Repeating the command reuses the saved result. Trained variants and final tests
are not overwritten for the same model pair. If you tune after seeing this
score, treat the test data as development data and collect new independent
test photos. Preserve old experiments for comparison.

## 9. Try live recognition and preserve your work

```bash
python -m apps.webcam predict --project data/ngt --model experiments/ngt_01/with_features/model.joblib
```

Replace `with_features` with `without_features` to try the other model. Use
`--camera 1` for another connected camera. Close with **Q**. Live viewing does
not save training photos or provide reliable accuracy without independent
correct labels. The classifier normally chooses among known letters, so an
unfamiliar gesture may receive an incorrect known label.

Export the complete dataset:

```bash
python -m scripts.export_dataset --project data/ngt --out ngt_dataset.zip
```

The export includes unchanged source images, analysis images, labels, sessions,
landmarks, alphabet references, manual finger assessments and thresholds.
`features.csv` is recomputed using those thresholds. The manifest also retains
originally cached features.

For submission, also preserve the program files, **experiments** and
**results/feature_check**. Experiments are outside the dataset and are not
included automatically in `ngt_dataset.zip`. Include small example data and
commands that let a reviewer run the experiment.

After moving the same unchanged dataset, supply its new location:

```bash
python -m scripts.compare_results --experiment experiments/ngt_01 --project extracted/dataset
```

Only the location changes. Contents must match the frozen dataset. Do not add
new images to an existing experiment's data. Translated source code also changes
code hashes: create a new experiment when using the updated implementation.

### Sharing a dataset ZIP with classmates

A dataset ZIP lets another student continue recording into your existing dataset.
It contains images, labels, sessions, landmarks, reference images and feature
assessments. Share the project code separately and use the same code version;
follow the [installation instructions](../README.md#installation) on each computer.
The dataset ZIP does not include the code or the separate experiment directories.

1. **Export the current dataset** from the project root:

   ```bash
   python -m scripts.export_dataset --project data/ngt --out exports/ngt_dataset_v01.zip
   ```

2. **Send that ZIP to the next student.** They should extract it into a new
   directory, keeping earlier versions separate. For example, after placing the
   received ZIP in `exports/`:

   ```bash
   python -m zipfile -e exports/ngt_dataset_v01.zip data/shared_v01
   ```

   The dataset root is now `data/shared_v01/dataset`: use the directory containing
   `manifest.json`, `images/` and `originals/`, not the directory above it.

3. **Open the received dataset** in the capture app:

   ```bash
   python -m streamlit run apps/collect_app.py -- --project data/shared_v01/dataset
   ```

   Check that **Project directory** points to the extracted `dataset` directory.
   There is no need to initialize a new dataset or copy individual image files.

4. **Add the next student's photos** using **Guided capture**. Give each student
   a unique **Participant** code, such as `student02`, and use the same agreed
   letter labels and condition names. Each camera start creates a new session.
   New photos are appended; existing photos remain in the dataset. Use
   **Development** for development rounds and **Separate final test** only for
   deliberately reserved test rounds. Keep existing test assignments unchanged.
   Guided captures are analyzed immediately; uploaded photos need analysis.

5. **Check and export the updated dataset.** Use **Check dataset** to verify the
   new participant, sessions and photo counts, then stop the capture app and run:

   ```bash
   python -m scripts.export_dataset --project data/shared_v01/dataset --out exports/ngt_dataset_v02.zip
   ```

   Send `ngt_dataset_v02.zip` to the next student, who repeats the process in a
   new extraction directory. Agree on who currently has the latest version.

**Use one successive dataset version for now.** If several students independently
extend the same original ZIP, their copies diverge. The project currently has no
automatic dataset merge command. Copying image folders or overwriting
`manifest.json` does not merge the labels, sessions and assessments correctly.
Sequential handoff of the latest ZIP avoids that problem.

After collecting everyone's contributions, keep the combined dataset in one
chosen directory and create a **new experiment** using that path. Train both
variants on the same new split; do not reuse a frozen experiment from before
the additional photos were collected. Use development CV to make choices and
leave the reserved final-test photos out of tuning.

## 10. Write your report

A possible structure for a five-page assignment:

1. **Question and data:** letters, participants, sessions, conditions, photo and
   detection counts, and the alphabet source.
2. **Implementation:** MediaPipe landmarks, normalization, both input sets and
   classifier. Reference the independent feature module.
3. **Feature inspection:** human assessment method, thresholds, precision/recall
   with counts, and specific errors.
4. **Experiment and results:** grouping, separate final test, accuracy/macro-F1,
   confusion matrices, feature comparison.
5. **Discussion:** limitations, error causes and justified next steps.

Report only measured scores. If extra features do not help, analyze information
already present in landmarks and potentially unstable features. A small
difference across few sessions does not prove a general advantage. If only one
participant was recorded, state that no new participants were tested.

## 11. Tests, example data and troubleshooting

Run without recording your own webcam images:

```bash
python -m unittest discover -s tests -v
python tests/smoke_modular.py --out examples/my_software_test
```

Use a new output directory for the smoke test. It creates synthetic skeleton
images and runs splitting, both training variants, CV, final testing and reports.
Synthetic scores verify software behavior, not real NGT recognition.
`scripts/demo.py` generates the synthetic examples used by these tests.

### Optional: what is `projects/demo_synthetic` for?

**Skip this directory when working with your own NGT photos.** Continue using
`data/ngt` and your existing experiments. The demo is an optional software
example, not an extra step in the recording or training workflow.

The directory name uses an underscore: `demo_synthetic`. It contains 160
artificial skeleton images for five generic poses (`fist`, `open_hand`, `peace`,
`pointing`, `thumbs_up`), not alphabet recordings. There are 120 development
samples and 40 test samples. Their landmarks are generated mathematically;
they do not test MediaPipe detection on real hands.

| Item | Why it exists | Do you need to open it? |
|---|---|---|
| `manifest.json` | Connects images, labels, sessions, roles and known landmarks | Only to inspect the dataset format |
| `originals/` | Preserves each generated source image | Usually no |
| `images/` | Stores the corresponding RGB analysis image | Usually no |
| `sources/`, `reports/` | Standard dataset directories; initially empty | No |
| `rules.json`, `.project.lock` | Internal files created by dataset initialization | No |

The 160 source files and 160 analysis files explain most of the file count.
You do not need to inspect them one by one.

To inspect geometric measurements without collecting photos, open the feature
inspection app on this dataset:

```bash
python -m streamlit run apps/feature_check_app.py -- --project projects/demo_synthetic
```

Choose **Visual inspection**, select a photo and expand **Show landmarks,
computed states and measurements**. Compare the displayed skeleton with its
angles and finger states. This is useful for understanding the calculations;
use real development photos for your research assessments.

To verify the complete training/comparison pipeline, use the smoke-test command
above instead. It creates its own temporary example dataset inside the selected
output directory; it does not need `projects/demo_synthetic` first.

You can delete the generated `projects/demo_synthetic` directory if you do not
use it. Keep `scripts/demo.py`: the unit tests and smoke test import its generators and
do not depend on this saved demo directory. To recreate it later:

```bash
python -m scripts.demo --out projects/demo_synthetic
```

The output directory must be new or contain no samples. Do not point this
command at `data/ngt`, and do not mix synthetic images into your real dataset.

### Troubleshooting

| Problem | Check |
|---|---|
| Only one session | Count actual development sessions per label with successful detection. Two photos are not two sessions; test sessions do not count for CV. |
| Two sessions but an error at 3 folds | Use 2 folds if both are development, or collect a third development round. Keep a separate test session too. |
| Unanalyzed photos | Run `scripts/analyze_dataset.py`. |
| No hand found | Show one complete hand, improve light, and check pose and distance. |
| Camera does not open | Close other camera apps, try `--camera 1`, and grant Terminal/Python camera access. |
| No camera window on a remote server | OpenCV runs on the computer running Python; run locally on the webcam computer. |
| Blank precision or recall | The denominator is zero. Add scorable positive/negative examples. |
| Dataset/code changed | Create a new experiment directory and retrain both variants. |
| Experiment directory exists | Choose a new name; earlier experiments are preserved. |
| Missing image after moving data | Extract the complete dataset export, including `images` and `originals`. |
| Linux lacks `libGLESv2.so.2` | Install `libgles2` in the local Linux environment. |

Automated checks do not open a physical webcam. Check camera permission,
drivers and recognition on your own real photos locally.

## Sources and scope

- MediaPipe Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- Grouped cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Precision definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- Dutch manual alphabet: https://helenkeller.nl/doofblindheid/communiceren/nederlands-handalfabet/
- Image and attribution: `assets/ngt_alphabet.png` and `assets/SOURCES.md`.

This application recognizes individual hand poses within selected labels. Full
sign-language translation additionally requires movement, multiple hands,
body posture, facial expression and context.
