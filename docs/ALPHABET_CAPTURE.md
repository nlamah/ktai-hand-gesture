# Alphabet capture: Shin reference versus joint angles

## Start

Close other camera apps. From the repository root:

```bash
conda activate ktai-hand-gesture
python -m research.alphabet_capture
```

First enter the participant name/code, previous development involvement, ASL fingerspelling experience and hand. The camera starts only after **Start capture**. The researcher should select **My hand has already been used during development**, even if their own photos were never training data. Select **New participant, not previously used** only when that person's images/results have not previously influenced the approach.

Each participant gets a separate session. The default is five A–Z rounds: **130 photos per person**, or 390 for three people. Relax and form each sign again; these are not five video frames of a single held pose.

## Examples and interpretation

A fixed example of the requested letter appears beside the camera image. [Review all examples](../assets/alphabet_capture_examples.jpg) before collecting the test. These are accepted Kaggle training images with their original dataset labels, not independently validated official ASL instructions. Ideally ask someone familiar with ASL fingerspelling to review the examples before the final experiment. Do not silently change examples during testing. Configuration is in `config/alphabet_capture_examples.json`; each session freezes example files and checksums.

New sessions show right-hand examples. Each example records whether its source image is horizontally mirrored for that display. The camera preview is unmirrored for comparison with the example. The model input is unchanged and never contains the example image. Old sessions retain their recorded display settings; start a new session to use the right-hand examples. Examples are not predictions. J and Z use static dataset forms; movement is not evaluated.

This is an **example-guided reproduction task**. With novice participants, results reflect both imitation and recognition, rather than spontaneous expert ASL production. Experience can improve sign execution but does not establish the linguistic validity of every photo.

## Controls

1. Enter participant details and click Start capture.
2. Check name, hand and role; press Enter to begin.
3. Form the requested letter with the same hand throughout. Keep camera, lighting, distance and setup consistent. Include the whole hand and wrist, preferably without the face.
4. **Briefly press Space** or click the capture button. The photo is saved immediately; no Enter confirmation is needed. A short cooldown limits duplicate saves.
5. Relax and follow the next letter. The tool advances through all five rounds.
6. Press **H before the photo** if additional assistance or sign correction was provided. The flag applies to that photo and then resets.
7. **Q/Esc** in the live window pauses and releases the camera.

Predictions remain hidden. Keep failed detections; do not repeat an attempt because the classifier was wrong. The tool closes after the final photo. Obtain consent for local photo storage and use participant codes instead of names if preferred.

## Permitted guidance

- **Before collection:** give everyone the same explanation and practice opportunity without model outputs. Correct sign execution during practice.
- **During collection:** standardized framing reminders are allowed, such as keeping the whole hand visible or maintaining the agreed distance.
- **“Turn the palm forward”:** use only when that orientation is part of the shared predefined task, not because a letter is recognized poorly.
- **“Bend the little finger more”:** this changes the sign itself. Prefer practice-time correction. If necessary during the test, apply the same criterion to everyone and mark assistance with H. Results then concern guided execution.
- **Avoid:** moving or rotating until a prediction becomes correct, retaining only successful photos, or tuning models on this final test.

Not training on someone's photos does not automatically make that person independent: their hands and observed errors may have influenced development choices. Experience and assistance are therefore recorded. Expert review can strengthen label validity. A novice group is also valid for an intended novice use case; state that scope rather than mixing experience levels into an unqualified general claim.

## Pause and resume

The terminal prints the actual session directory under `captures/alphabet_test/`. Resume it rather than creating another session for the same participant:

```bash
python -m research.alphabet_capture --resume captures/alphabet_test/alphabet_DATE_ID
```

Check progress without opening the camera:

```bash
python -m research.alphabet_capture --resume captures/alphabet_test/alphabet_DATE_ID --status
```

Start without the entry form:

```bash
python -m research.alphabet_capture --name "Researcher" --kind development --experience familiar
python -m research.alphabet_capture --name "Participant 2" --kind new --experience beginner
```

These are example profiles, not existing registrations. `--repeats 1` makes a shorter 26-photo session. Use the same repetition count and examples for everyone in one evaluation.

## Evaluate completed sessions

Pass the actual completed session directories:

```bash
python -m research.evaluate_alphabet captures/alphabet_test/SESSION_1 captures/alphabet_test/SESSION_2 captures/alphabet_test/SESSION_3
```

This reuses the saved images/landmarks and the two frozen model files. Nobody needs to take separate photos for the second model. No training occurs.

Outputs in `results/alphabet_evaluation/` include a report, predictions, per-letter precision/recall/F1, confusion matrices and summary. Scores are reported per participant, across participants and separately for new participants. Known development participants are not treated as independent new users. The main table counts failed detections as incorrect attempts; `summary.json` also reports accepted-only scores and detection coverage. Assistance flags remain available, without silently excluding unfavorable attempts.

This is a small new-participant check, not a representative population study. Once its outcomes guide model or feature selection, new material is required for another independent final evaluation.
