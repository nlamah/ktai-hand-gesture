# Next G/H rotation experiment

The feature study tests whether interpretable descriptions add value to a geometric reference. The next G/H investigation asks whether rotation augmentation helps across intended capture conditions. Its results must remain separate from the completed feature comparison.

## What already exists

Live button **G/H rotation [5]** uses `models/asl_shin_angles_GH_rotation.joblib`. Its training recipe adds camera-y direction rotations of -90, +90 and 180 degrees to G/H examples, retaining the originals. Only the 630 Shin direction inputs change; distances and world-joint inputs remain fixed. This is an approximation in normalized feature coordinates. It neither rotates RGB photos nor reruns MediaPipe, and does not create realistic occlusions or back-of-hand views.

The frozen five-fold comparison corrects nine baseline errors but loses 15 correct predictions: net **-6**, accuracy 97.73780% versus 97.74776% for the world-joint reference. This -6 concerns augmentation relative to the world-joint model; it is separate from the +67 world-joint and +4 same-coordinate feature comparisons. Selected webcam examples do not establish population accuracy. Evidence is in `results/rotation_GH_01/`.

Try this existing candidate:

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_GH_rotation.joblib
```

The existing runner can create a separate feature-space candidate. For example, if a small in-plane rotation is chosen in advance, these paths keep the earlier model and results intact:

```bash
python -m research.studies.tilt_augmentation --stage train --letters G H --axis z --degrees 0 -15 15 --candidate models/asl_GH_z15.joblib --out results/GH_z15_01
python -m research.studies.tilt_augmentation --stage cv --letters G H --axis z --degrees 0 -15 15 --candidate models/asl_GH_z15.joblib --out results/GH_z15_01
python -m research.live_camera --model models/asl_GH_z15.joblib
```

These are illustrative commands, not an executed or selected new experiment. Synthetic direction rotations in normalized image coordinates remain an approximation even around z. The runner learns selective-augmentation scaling, class weights and resolved gamma on original fitting rows. Each source retains total weight one; only the fitting partition receives rotated copies. A custom candidate is selected with `--model`; it does not add or replace a live button.

## If the intention is to rotate the training photos

This needs a separate RGB-augmentation path; the existing feature-space runner does not implement it. Predefine the rotation angles, interpolation, canvas/border policy and acceptance rule. Include all original G/H sources being tested, including failed detections when evaluating detection coverage. Keep the original file, label and source group for every derivative.

1. Split original source images into training and held-out groups first. All derivatives inherit their source's group; none may cross into validation.
2. Rotate G/H training photos only and rerun the same MediaPipe detector. Record accepted and rejected detections for each angle. Do not label a detector acceptance as proof of accurate landmarks.
3. Compare a model using original training data with one additionally using the accepted rotated examples. Keep fitting settings fixed and account for how many copies each source contributes. Record detection configuration, augmentation parameters and source/model hashes.
4. Score both models on the same untouched images. Report G/H precision, recall, F1, G-to-H/H-to-G errors, and all-letter accuracy and macro-F1. Measure any change in detection coverage separately from conditional classification performance.
5. Use a development collection to choose a candidate and reserve new participants/sessions for its final check. Save every planned attempt, including failed detections. Existing selected errors remain diagnostic examples.

A 2D photo rotation changes in-plane orientation; it cannot reveal the back of the hand or simulate a new palm/back viewpoint. To investigate that condition, collect actual viewpoint changes with `python -m research.gh_view_capture`; instructions are in `docs/GH_CAPTURE.md`. If light is also varied, record it as a separate condition so its effects are distinguishable from rotation.

No new G/H training, data collection or live-model replacement is performed by the paper revision.
