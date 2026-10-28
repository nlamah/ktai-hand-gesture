# Practical webcam research protocol

Run these commands from the project root in the `ktai-hand-gesture` environment.
This guide separates exploration, improvement and a new final test. Existing
datasets, models and comparisons are preserved. No new dependencies, features
or parameter search procedures are added.

## 1. Select and register the baseline

The current reference model, trained with the current feature code, is:

```text
experiments/shin2021_finger_kaggle_01/svm/shin2021_combined/model.joblib
```

Registration reads the actual sklearn model and checks SVC, the Shin input
variant and 820 input columns. It records parameters, preprocessing, classes,
column order and available provenance hashes. It does not train a model.

```bash
python -m scripts.research register --model experiments/shin2021_finger_kaggle_01/svm/shin2021_combined/model.joblib --experiment experiments/shin2021_finger_kaggle_01/svm --alphabet ASL --name "SVM Shin baseline" --require-baseline --out exports/research_setup/baseline.json
```

This registration was already created during implementation. Skip this command
if the file exists; use a new output name for a new selection. Existing
registrations and output directories are not overwritten. `--alphabet ASL` is an
explicit declaration about the source, not a conclusion based on letter names.
Historical MediaPipe settings are not fully recorded in the old model files
and therefore remain unverified. Older models with different feature code are
not silently converted.

## 2. Explore one model live

```bash
python -m apps.webcam ab --model-a exports/research_setup/baseline.json --out exports/live_baseline --intended-label U --countdown 2
```

Without `--model-b`, both panels show the same model: this is exploration with
one model. The example letter `U` is used only for assessment. The model can
still predict all 26 original classes, including J and Z. Omit `--intended-label`
to show agreement or disagreement only. To use a different intended gesture,
restart with that label before recording.

Controls: **F** freezes/resumes the view, **S** or the **Save frame** button starts
capture with the chosen countdown, **L** toggles landmarks and **Q/Esc** stops.
The saved image and both predictions share the same frame ID. Predictions are
raw by default; `--smoothing 1` is the default. A larger value displays smoothing
separately. SVM scores are not shown as probability percentages. `--no-mirror`
disables display mirroring only; the analysis pixels remain identical. Selected
live images are development/demonstration material and are never automatically
added to training or an independent final test.

## 3. Compare two models side by side

An existing second model could be the earlier Shin/finger-feature combination:

```bash
python -m scripts.research register --model experiments/shin2021_finger_kaggle_01/svm/shin2021_with_finger_features/model.joblib --experiment experiments/shin2021_finger_kaggle_01/svm --alphabet ASL --name "Shin plus finger features" --out exports/research_setup/shin_fingers.json
python -m apps.webcam ab --model-a exports/research_setup/baseline.json --model-b exports/research_setup/shin_fingers.json --out exports/live_comparison --intended-label V --countdown 2
```

Both models receive the same frame and the output of a single MediaPipe call.
Each then uses its own feature definition, stored column order and preprocessing.
The app checks the alphabet, complete class list and feature version. Missing
historical detector requirements are reported. If detection fails, the previous
prediction disappears from both panels.

With an intended label, there are four categories: both correct, only A correct,
only B correct and both incorrect. Without a label, matching predictions indicate
agreement only. Many consecutive frames do not constitute independent trials.

## 4. Collect new development recordings

An unconfirmed example is available at
`exports/research_setup/example_draft_protocol.json`. It is not a recording or
test protocol approved by you. The command below creates your own version.

Choose six relevant letters yourself. A/U/V/M/N/R below are an **example**, not an
exclusion of other letters. Check the ASL references and determine whether the
intended label remains valid under the chosen setup. Use
`--confirm-label-validity` only after assessing this choice yourself.

```bash
python -m scripts.research protocol --out exports/research_setup/development_p01_s01.json --alphabet ASL --letters A U V M N R --participant p01 --session development_s01 --role development --repetitions 5 --countdown 3 --interval 3 --camera 0 --seed 42 --angle-method camera_view --confirm-label-validity
python -m apps.webcam record-protocol --protocol exports/research_setup/development_p01_s01.json --project data/webcam_development
```

This creates 150 planned attempts: six letters × five conditions × five separate
repetitions. Conditions are frontal, approximately 15° left/right and
approximately 30° left/right. Condition blocks are shuffled for each repetition
using the stored seed. Letter order remains fixed within a block. At each new
block, the app pauses so you can adjust the setup. Relax your hand and form the
gesture again before each separate attempt.

With `camera_view`, change the camera's viewing direction while keeping the
intended gesture the same. Define and record the perspective from which you
describe left and right beforehand. With `gesture_rotation`, rotate the gesture
relative to a fixed camera. These are different interventions: orientation can
change meaning. Angles are **nominal arrangements**, not measured camera angles;
no actual angle is inferred from landmarks. Supply custom conditions as a JSON
list with `--conditions path/to/conditions.json`, using entries such as
`{"name":"front","nominal_angle":0}`.

The left panel shows a region of the existing William Vicars/Lifeprint ASL chart;
the right panel shows the unmirrored webcam view. The source, file hash and crop
region are recorded in the protocol. The NGT images in `assets/alphabet/` are not
used here. Review the references yourself: a photo with movement arrows for J/Z
does not replace complete movement instructions. Other references can be supplied
with `--references` and `--reference-source`; missing images are not invented.

**Space** starts/pauses, **R** makes an identifiable retake of the previous attempt,
**L** shows/hides landmarks and **Q/Esc** stops. To resume after stopping:

```bash
python -m apps.webcam record-protocol --protocol exports/research_setup/development_p01_s01.json --project data/webcam_development --resume
```

Every planned attempt is saved, including images with no hand or multiple hands.
The app does not wait for a favorable detection moment. The original PNG, exact
analysis pixels, UUID, hashes, timestamp, intended label, participant/session,
condition, repetition, order, detection status, landmarks and detector settings
are preserved. Retakes replace nothing and count as additional attempts. Failure
to detect a hand does not prove poor lighting.

For example, add an explicit human note or invalidity declaration with:

```bash
python -m scripts.research annotate --project data/webcam_development --photo-id PHOTO_ID_FROM_MANIFEST --note "Gesture performed incorrectly; assessed manually" --invalid
```

Use the actual photo ID from `manifest.json`. The photo remains stored and the
reason is appended. This command applies only to new research recordings.

## 5. Analyze error patterns

```bash
python -m scripts.research analyze --model-a exports/research_setup/baseline.json --project data/webcam_development --out exports/webcam_development_analysis
```

Open `exports/webcam_development_analysis/report.html`. The output contains
precision/recall/F1/support per letter, confusion matrices with counts and row
normalization, condition and detection scores, and correct/incorrect examples
with landmarks. Click a confusion pair to filter the displayed examples. The
complete selection is in `predictions.csv`; the images are a limited selection.
Add `--model-b` to include a second model.

Enter notes in the HTML report and click **Download entered human notes**.
Save the file inside the project, for example as
`exports/webcam_development_analysis/review_notes.json`, then import it:

```bash
python -m scripts.research notes --analysis exports/webcam_development_analysis --notes exports/webcam_development_analysis/review_notes.json
```

These are human observations/hypotheses, not proven causes. Predictions never
replace labels. Reserved test images are excluded from this development analysis.

## 6. Train one targeted improvement

This command refits only the existing SVM/Shin pipeline on the original
development photos plus explicitly selected new development recordings. It
clones the actual saved estimator and settings; any learned preprocessing is
fitted again using only the new training set. It performs no parameter search
or CV comparison.

```bash
python -m scripts.research train --baseline exports/research_setup/baseline.json --additions data/webcam_development --all-development --dataset-out data/asl_viewpoint_development_01 --experiment-out experiments/asl_viewpoint_01 --hypothesis "More usable development recordings from different viewpoints reduce the observed confusion" --change "Original development data plus selected new viewpoint recordings; identical SVM and Shin features" --expected "Less confusion on preselected letters in new separate test recordings"
```

Adapt the hypothesis and expected outcome to your own error analysis. Use
`--ids ID1 ID2 ...` instead of `--all-development` to select specific development
attempts. Test roles are rejected, including attempts whose original protocol
role was test. Detection failures and excluded attempts are recorded in the
selection log and remain in the recording dataset. Exact pixel duplicates with
the same label are logged and not counted twice; conflicting labels are rejected.

The new dataset contains copies and requires additional disk space. The new
experiment includes `selection.json`, counts per letter/participant/condition,
training IDs, provenance hashes, configuration and `registered_model.json`.
Both data **amount and composition** change. This does not isolate the effect
of viewpoint alone.

Optionally run a separate control with additional frontal photos: make a new
selection, use new output directories and add `--condition front --match-count N`.
Replace N with the same number of usable added photos as in the viewpoint
improvement. The command rejects a different count and does not automatically
select favorable photos. This control is not run automatically.

## 7. Freeze models and the final-test protocol

For a separate classifier experiment using the same original Shin inputs and
folds, see [MLP_TRAINING.md](MLP_TRAINING.md). It compares a fixed small MLP with
the unweighted, unaugmented SVM baseline before any independent final test.

For an optional original-plus-mirror training experiment targeting left/right
hand differences, see [MIRROR_TRAINING.md](MIRROR_TRAINING.md). It augments only
training landmarks; the resulting model accepts ordinary webcam input.

An alternative targeted experiment is to weight the existing index/middle
fingertip distance when U/V/R are confused. See
[FINGERTIP_WEIGHTING.md](FINGERTIP_WEIGHTING.md) for that separate training route.
Choose its weight using development data and register the new model before
freezing the final comparison below.

Choose the final letters, participants, conditions and settings before recording
and evaluation. Use a new session and new photos; changing a flag does not turn
previously viewed live images into an unseen final test.

```bash
python -m scripts.research protocol --out exports/research_setup/test_p01_s02.json --alphabet ASL --letters A U V M N R --participant p01 --session final_test_s02 --role test --repetitions 5 --countdown 3 --interval 3 --camera 0 --seed 2026 --angle-method camera_view --confirm-label-validity
python -m scripts.research freeze --protocol exports/research_setup/test_p01_s02.json --model-a exports/research_setup/baseline.json --model-b experiments/asl_viewpoint_01/registered_model.json --out exports/research_setup/final_test_seal_01.json --hypothesis "The frozen viewpoint improvement makes fewer errors than the baseline on the same new recordings"
```

The seal records the protocol hash, timestamp and both model identities and
configurations. It is an inspectable log of ordering and choices, not proof of
human independence. One participant remains one participant; many photos or
angles do not replace multiple people.

## 8. Make new test recordings without predictions

```bash
python -m apps.webcam record-protocol --protocol exports/research_setup/test_p01_s02.json --seal exports/research_setup/final_test_seal_01.json --project data/webcam_final_test
```

This mode loads no classifier and shows no predicted letter, correctness or
model score. Landmarks can be displayed; all detection statuses still count as
offered attempts. Use the same arguments with `--resume` to continue a protocol
that has not yet been evaluated. After evaluation, new independent collection
requires a new protocol/session.

## 9. Evaluate the same saved test photos

```bash
python -m scripts.research evaluate --seal exports/research_setup/final_test_seal_01.json --project data/webcam_final_test --out exports/webcam_final_comparison_01
```

Both models receive the same saved detector output for each photo. There is no
model-specific selection of favorable detections. The software checks protocol
fields, recording order, timestamps, model hashes, pixels, detector settings and
known exact training IDs/pixel hashes. Missing provenance is not treated as a
passed overlap check. Restore that information or, only after considering the
limitation, explicitly use `--acknowledge-unchecked-overlap`; the limitation
remains in the report. An intentionally incomplete session requires
`--allow-incomplete`, and missing classes are reported.

Accuracy and macro-F1 use exactly the same recorded, included photos with usable
detection. A model-specific feature/prediction error remains in that denominator
and counts as wrong. The predeclared stress letters determine the macro-F1
classes; a missing class contributes zero. All original model answers remain
possible: a predicted Z in a stress set without Z counts as wrong. Rejected or
unknown answers have a separate confusion-matrix column. Detection coverage
uses all offered attempts, including explicitly invalidated attempts; included
and usable counts are reported separately.

Repeating evaluation with the same seal is technically allowed with a new output
directory; the access log records earlier evaluations. To try a different model
later on the same already viewed set, create a new seal with **`--viewed-retest`**
and a new output name. This requires a previously recorded evaluation and is
explicitly marked as a viewed-set retest. Changes based on these results need
new unseen data for independent confirmation. A log cannot establish what
someone viewed outside the application.

## 10. Export results and disagreement images

Evaluation and development analysis directly write HTML, JSON and CSV:
`report.html`, `result.json`, `predictions.csv`, `condition_scores.csv`,
`detection_coverage.csv`, per-letter scores, letter differences, confusion
matrices and selected examples with landmarks. `result.json` contains the
protocol, both model configurations, counts and limitations. Paired categories
show corrected errors (only B correct), new errors (only A correct) and shared
errors. There are no CV error bars or automatic significance claims.

For the separately selected live frames:

```bash
python -m scripts.research export-live --source exports/live_comparison --out exports/live_comparison_export
```

This creates category counts, a CSV and a `differences/` directory. These counts
describe selected development frames; they are not an independent test.

## Technical checks and your own review

```bash
python -m unittest discover -s tests -v
python tests/smoke_research.py --out exports/research_software_test_new
```

The second command exercises the workflow with small **synthetic software-test
data**, including detection failure, improvement, live export and a saved test.
Its scores are not an ASL recognition measurement. Always choose an output
directory that does not exist yet. See [VALIDATION.md](VALIDATION.md) for the
separate audit and one-variant replay.

Codex ran the reported software tests/audit. Codex did not test a physical
webcam, measure actual viewpoints, perform human gesture validation or conduct
a new independent webcam test. You must review references and label validity,
define the setup, collect real development recordings, choose the hypothesis
and only then freeze a new test protocol. These actions have not been recorded
as performed on your behalf. Historical detector settings, near-duplicates and
real participant identities remain unverified where the source lacks that
information.
