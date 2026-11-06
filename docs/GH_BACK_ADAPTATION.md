# G/H back-of-hand adaptation

This development candidate addresses the mismatch between the palm-facing G/H examples in the dataset and the back-of-hand webcam poses. It uses the existing data; it does not require a new recording round to train the included model.

## What changes

The starting point is the 26-class mirror-trained SVM, using the same 835 inputs. For G/H fitting rows only, add one synthetic 180-degree roll around the estimated index MCP-to-tip direction. This proper 3D rotation preserves the pointing direction instead of rotating a horizontal H into a vertical U-like pose. Distances and intrinsic joint measurements stay fixed. Original, mirrored and rolled versions of one source together retain weight one.

Three previously saved, visually inspected back-of-hand examples (two G and one H) are also added as **calibration training**. Their RGB pixels and horizontal reflections are redetected with the same MediaPipe settings. Each calibration source has total weight 20 across its available views and rolls. This is a fixed personal-calibration weight, not a search over test scores. Historical capture JSON files stay unchanged; the new protocol explicitly records their reuse for training at the user's request. The palm-facing H remains a previously inspected development control.

The added 26-class SVM reuses the baseline's scaling, gamma and class weights. A routing gate allows its G/H proposals only when the index has a sufficiently visible, mainly horizontal projection in normalized image coordinates. Other proposals keep the original baseline prediction. Thus this is not an always-G/H binary diagnostic. The gate is not a calibrated anatomical angle or a guarantee of correct hand detection.

## What this does not imply

A geometric roll is not a new RGB view: it cannot reveal hidden fingers, generate the real back of a hand, or simulate detector errors. The three existing camera images supply actual target-view measurements, but success on them is **training fit**, not independent webcam accuracy. The source dataset labels are retained; the model is not an ASL correctness assessor. G/H may also have side-facing variants, so a strict front/back binary rule is not a linguistic standard.

ASL reference: [Bill Vicars / ASL University, G and H feedback](https://www.lifeprint.com/asl101/topics/feedback.htm) describes palm-back or palm-side rather than palm-front for citation forms. A [fingerspelling study](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.953019/full) likewise describes inward-facing palms for G/H. Screen left/right depends on the hand and preview mirroring.

## Reproduce

```bash
conda activate ktai-hand-gesture
MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.gh_view_adaptation --stage prepare
python -m research.studies.gh_view_adaptation --stage train
python -m research.studies.gh_view_adaptation --stage cv
python -m research.studies.report_gh_view_adaptation
python -m unittest discover -s tests
```

Preparation requires the three original local capture folders and checks their hashes. Training also uses the mirror experiment's cached detector outputs and fitted baseline; it leaves the source dataset and earlier models intact. The new artifact is `models/asl_gh_back_adapter.joblib`. Metadata list the calibration sources, fixed weight, roll, gate and provenance. Reproduction on another machine needs those local calibration files; the saved model can be used without the photos.

CV retains the source folds and balanced presentation mix of the mirror study. Its baseline predictions are checked by source ID, label and fold before reuse. The calibration images are additional fitting data in every fold. Normal-dataset changes, all-letter effects and replay of selected camera examples are reported separately under `results/gh_view_adaptation_01/`. This checks regressions on source data; it is not an independent test on a new back-of-hand cohort.

## Use live

```bash
python -m research.live_camera --model models/asl_gh_back_adapter.joblib
```

This index-axis candidate is available through `--model`. The separate [combined vertical-axis candidate](GH_VERTICAL_ADAPTATION.md) also uses mirror training and calibration. Compare either of these with **Mirror trained [8]**. Button **G/H y180 only [9]** now selects the later [standalone rotation experiment](GH_ROTATION_STANDALONE.md), whose reference is **World joints [2]**. Keep the whole hand and wrist visible. The mirrored screen preview does not change detector input. Save failures with Space and the actual intended letter; the tool records the model hash, exact features and selected scores. Scores are internal SVM decision values, not calibrated probabilities.

The model is ready for practical comparison. A new recording with the hand back/side toward the camera is needed before claiming that the viewpoint problem has been solved for unseen attempts or other people.
