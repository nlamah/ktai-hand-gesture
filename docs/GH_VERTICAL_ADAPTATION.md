# G/H: turn the landmark geometry around the vertical axis

This document describes the earlier **combined** mirror/rotation/calibration candidate. The later [standalone rotation experiment](GH_ROTATION_STANDALONE.md) contains no mirrors, calibration or routing rule and is now live button **G/H y180 only [9]**. Its results are reported separately in Appendix M.

For the described palm-facing training examples, a 180-degree turn around the **vertical camera y axis** changes both the pointing side on screen and the palm/back direction. Around the wrist, the landmark transformation is:

```text
x' = wrist_x - (x - wrist_x)
y' = y
z' = wrist_z - (z - wrist_z)
```

Horizontal reflection changes x only. This turn changes x and z; its determinant is +1, so it is a proper rotation rather than a handedness reflection. For the pairwise direction inputs, translation cancels and the exact rule is `(dx,dy,dz) -> (-dx,dy,-dz)`. Pair distances and intrinsic world-joint angles are invariant. The source's anatomical hand identity and letter label are retained. Synthetic depth does not provide a new MediaPipe handedness measurement.

The implementation is [research/gh_vertical.py](../research/gh_vertical.py). Tests compare the transformed inputs with features freshly encoded from wrist-centred rotated landmarks, including the change in palm-normal depth direction. No RGB photo is described as a generated back view.

## Training with the existing data

The starting point is the mirror-trained A–Z SVM. Existing original and RGB-reflected G/H fitting examples additionally receive a y180 copy. Other letters receive no y180 copies. All variants of one Kaggle source together retain total weight one. Reflections continue to address left/right hands; vertical turns address the separate palm/back orientation mismatch.

The three previously saved back-facing G/H examples (two G, one H) are also included as **calibration training**, with source weight 20 each across their views. This is explicitly recorded in `results/gh_vertical_01/calibration_sources.json`. Raw capture records are unchanged. They are no longer held-out evidence for this candidate. The palm-facing H is a previously inspected development control, not calibration.

The new 26-class learner uses the matched baseline's scaling, gamma and class weights. Its G/H proposals are used only for poses with a sufficiently visible, mainly horizontal projected index. Other proposals retain the baseline prediction. The gate is fixed; it is not proof of correct landmarks or protection against every false G/H prediction.

Both rotation and calibration change in this comparison, so any improvement cannot be attributed to rotation alone. The earlier index-axis roll is a separate development candidate; it preserves finger direction and is **not** the requested vertical-axis operation. Its artifacts remain under `results/gh_view_adaptation_01/` and `models/asl_gh_back_adapter.joblib` for inspection. It can be loaded with `--model`.

## Evidence and limits

[Results and checks](../results/gh_vertical_01/report.md) report all-letter precision/F1, G/H precision/recall/F1, new false G/H predictions and selected capture replays. The source-data check uses the same five filename-block folds and balanced original/reflected test mix as the mirror experiment. Rotated copies are fitting data only; source images remain in their assigned fold. The three extra camera calibration images are fitting data in every fold.

This checks source-data regressions, not unseen back-view accuracy. Landmarks contain estimated depth in normalized image coordinates. A geometric turn cannot reconstruct occluded fingers or reproduce the detector's errors when the real hand is turned. Good recognition of the three reused camera images is training fit. The model remains an optional live development candidate; independent back-view improvement has not been established.

Executed results: on the balanced 78,000-source presentation, macro precision changes from **0.974508 to 0.974647**, and all-attempt macro-F1 from **0.845599 to 0.845682**. There are 11 corrected and 5 newly incorrect predictions: net **+6**. Accepted-only macro-F1 changes from **0.974206 to 0.974324**. G's all-attempt F1 decreases from 0.898011 to 0.897029; H's increases from 0.872957 to 0.874096. These small mixed changes do not demonstrate a general improvement.

The rotation-only ablation recognizes the known back-facing H, but still calls both known back-facing G examples H. Rotation plus calibration recognizes all three; this is training fit. The separate palm-facing H control remains incorrectly predicted G. The saved examples therefore do not show that a vertical turn alone solves the live mismatch.

## Run and reproduce

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_gh_vertical_adapter.joblib
```

The explicit `--model` command selects this combined candidate. Compare it with **Mirror trained [8]** on the same stream. Button **9** selects the later standalone candidate, not this combined model.

```bash
python -m research.studies.gh_vertical_adaptation --stage prepare
python -m research.studies.gh_vertical_adaptation --stage train
python -m research.studies.gh_vertical_adaptation --stage cv
python -m research.studies.check_gh_vertical_only
python -m research.studies.report_gh_view_adaptation --study gh_vertical_01
python -m unittest discover -s tests
```

Preparation needs the original local captures; training needs the existing mirror-study cache and model. The included calibration cache was copied unchanged from the index-roll experiment: no new photos or detector run were needed for the vertical variant. The model works without the photos. Hashes, calibration roles, source code, model and verification are recorded in the result directory.

The `check_gh_vertical_only` ablation also fits a model with the y180 turns **without** any camera calibration. It replays the previously selected G/H examples to show whether rotation alone explains their change. Those examples remain development controls; this arm has no independent back-view test or source-data cross-validation.
