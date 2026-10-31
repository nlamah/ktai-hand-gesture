# ASL live feature check

This local Streamlit app opens the browser webcam and compares its detected hand
features with the letter entered by the user. It shows a mirrored live preview
and updates the comparison about twice per second:

- Green means the feature matches one of the selected letter's workbook values.
- Red means the feature differs or could not be recognized. With no detected
  hand, the app shows that all scored features are unavailable.
- Gray means the workbook marks the feature as not scored (`kan`).

The app checks feature values; it does not independently predict which letter
the hand represents. Its expected values are only as reliable as the workbook.
The experimental threshold values are in
`config/asl_landmark_thresholds_tuned.json`. To try another threshold file or
workbook, pass `--thresholds` and `--reference` when starting the app.

Start it from the project root in the `ktai-hand-gesture` environment:

```bash
conda activate ktai-hand-gesture
python -m streamlit run apps/asl_live_feature_check_app.py \
  --server.address 127.0.0.1 --server.port 8503
```

By default, the updated workbook is read from
`assets/asl_feature_references.xlsx`. To use another workbook, upload it in the
app or set its path:

```bash
python -m streamlit run apps/asl_live_feature_check_app.py \
  --server.address 127.0.0.1 --server.port 8503 -- \
  --reference "/path/to/ASL feature references.xlsx" \
  --thresholds config/asl_landmark_thresholds_tuned.json
```

Type one letter A–Z, enable **Webcam aan**, and allow camera access in the
browser. The browser sends reduced-resolution JPEG frames to the local
Streamlit process for MediaPipe analysis; audio is not requested and frames are
not saved. Turn the webcam toggle off to stop the camera.

The tuned thresholds were selected from train data and remain experimental.
Some selected values are at the edge of the searched ranges; the workbook
entries (including the thumb MCP shape approximations) should be reviewed
visually before treating a red/green result as a definitive ASL judgment.

## Three flexion states and slash alternatives

Workbook review and live comparison distinguish `gebogen` (bent),
`licht gebogen` (partly bent), and `gestrekt` (extended). For index, middle,
ring and little fingers, the updated trained thresholds are at most 125 degrees
for bent, above 125 and below 150 for partly bent, and at least 150 for extended.
The thumb IP retains three states. Thumb MCP is now trained as B-shaped versus
not B-shaped using `workbook_thumb.b_min_deviation_deg`; neutral, C and L
references are mapped to not B-shaped.

Slash alternatives accept exactly the listed categories. `licht gebogen/gestrekt`
accepts those two categories; `gebogen/gestrekt` does not implicitly accept
partly bent. A single named category stays strict. `kan` remains unscored.
The original spreadsheet is not edited.

Only finger flexion and workbook crossing were retuned in
`experiments/asl_three_state_crossing_02/`. Selection maximizes per-feature,
per-category macro-F1 on training source groups, with accuracy as tie breaker,
so the partly-bent category and rare crossing category receive weight.
The existing group holdout is a development diagnostic, not a new independent
final evaluation. No test-split images were used.

```bash
python -m gesturelab.features.asl_threshold_tuning \
  --fingers-and-crossing \
  --initial-thresholds experiments/asl_feature_spread_tuning_v2_02/categorical_thresholds.json \
  --output-dir experiments/asl_three_state_crossing_03
```

## Crossing rule

The previous rule only checked whether the lateral order of fingertips was
reversed relative to the MCPs. It missed crossings around DIP level when tips
returned to their original order, and could falsely flag folded fingers.

The workbook/live rule compares the PIP–DIP–tip polylines at equal heights in
the local palm frame, searching for a reversal of lateral order. It limits
depth separation and excludes strongly folded PIP/DIP joints. Degenerate or
non-overlapping profiles produce unknown. This geometry does not prove physical
contact, and still depends on accurate landmarks during occlusion.

On detected development holdout images, false crossing predictions decreased
from 29 to 1. Neither method correctly recognized either of the two true
crossings in that holdout; one became unknown with the new rule. On training
images the new rule detects 30 of 39 true crossings (one unknown), with five
false positives, compared with 31/39 and 138 false positives previously.
These are diagnostics, not evidence of reliable webcam R recognition.
The general categorical training extractor keeps its original crossing rule.

## Updated spread rule

The live and workbook review comparison uses the updated workbook supplied on
2026-10-30 (the original Downloads file is preserved). U expects no index-middle
spread; V and K expect spread. All other workbook corrections are imported too.

For index-middle and middle-ring, spread now requires a lateral gap at PIP,
DIP and fingertip level, consistent finger ordering, roughly the same palm-relative
depth, and fingers pointing in the same general direction. Fanning out is allowed.
Strongly folded PIP/DIP joints are excluded; a bent MCP is allowed, as needed for K.
Degenerate geometry gives unknown. Landmark centers approximate the gap;
they cannot prove skin contact.

Only `workbook_spread` was recalibrated against the updated workbook; thumb,
flexion and other thresholds were frozen. Selected bounds: minimum lateral gap
0.25 palm lengths, maximum depth difference 0.6 palm lengths, minimum PIP/DIP
angle 120 degrees. Threshold search uses training source groups only.

Exploratory diagnostics on the previously used group holdout, for index-middle
spread only: U 11/12 correct (previously 12/12), V 6/8 (previously 2/8),
K 12/12 (previously 0/12). This holdout was inspected while revising the rule,
so these are development diagnostics, not an independent final accuracy claim.
Webcam performance still needs visual confirmation. The original tip-distance
rule in the general categorical training extractor is unchanged.

Reproduce with:

```bash
python -m gesturelab.features.asl_threshold_tuning \
  --spread-only \
  --initial-thresholds experiments/asl_feature_threshold_tuning_merged_fingers_01/categorical_thresholds.json \
  --output-dir experiments/asl_feature_spread_tuning_v2_03
```

Results: `experiments/asl_feature_spread_tuning_v2_02/`.

## Spreadsheet v3: binary thumb and training-data agreement

The updated project workbook is copied from `Untitled spreadsheet (2).xlsx`.
Only thumb MCP is reduced to B-shaped/not B-shaped; thumb IP, finger flexion,
spread, crossing and contact retain their respective categories. The binary
thumb threshold is calibrated against the signed MCP deviation, maximizing
category-balanced macro-F1 on fitting source groups. Unknown handedness or
unusable geometry remains unknown, rather than being assigned not B-shaped.

Run all workbook-relevant threshold groups (the general training extractor is
not retrained):

```bash
python -m gesturelab.features.asl_threshold_tuning \
  --workbook-all --initial-thresholds experiments/asl_three_state_crossing_02/categorical_thresholds.json \
  --output-dir experiments/asl_workbook_v3_binary_thumb_02
python -m scripts.report_asl_workbook \
  --thresholds experiments/asl_workbook_v3_binary_thumb_02/categorical_thresholds.json \
  --output-dir experiments/asl_workbook_v3_binary_thumb_02
```

The report separates full-image compliance, per-feature compliance, detection
failures and unknown geometry. `kan` is omitted. The letter table ranks failures
including unknowns; the detailed CSV also separates actual mismatches from
unknown values and shows the most common incorrect category. Fit-group and
reused development-holdout results are reported separately from the descriptive
score across all training images. No final test data is used.
