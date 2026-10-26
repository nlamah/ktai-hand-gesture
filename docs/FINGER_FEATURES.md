# Derived finger features — additional features calculated from landmarks

This guide covers the 81 additional geometric finger features calculated
from landmarks, rather than the landmark coordinates themselves.

The single implementation of these finger features is
`gesturelab/features/finger_features.py`. Both training scripts and the inspection app
use that code. `gesturelab/features/inspection.py` also normalizes landmarks.

For where these features enter training and how to define additional input
variants, see [TRAINING_INPUTS.md](TRAINING_INPUTS.md).

## Define what you mean first

A letter label does not automatically establish the true finger states.
Assess each visible finger yourself as **extended**, **bent**, **intermediate**
or **unscorable**, preferably before revealing the computed result. An occluded
thumb may be unscorable. This avoids merely confirming the code's own answer.

Default software definitions:

- Extended: both joint angles ≥155° and straightness ≥0.80.
- Thumb extension: both angles ≥145° and straightness ≥0.80.
- Bent: at least one angle <125° or straightness <0.65.
- Otherwise: intermediate, which is not automatically classified as bent.

These configurable working hypotheses are not medical or linguistic standards.
The thumb has different joints from other fingers. The extra base angle is a
geometric approximation, not an anatomical measurement. Save changed thresholds
in the inspection app; `finger_thresholds.json` stores them. New experiments
copy them into their own directories.

## Calculations from landmarks

For three points A–B–C, the interior angle at B is:

`angle = arccos(dot(unit(A-B), unit(C-B)))`, converted to degrees.

A straight line has an angle of 180°; a right angle is 90°. Straightness is:

`distance(base, tip) / (distance(base, joint1) + distance(joint1, joint2) + distance(joint2, tip))`

A straight finger approaches 1. Distance features are divided by the
wrist-to-middle-knuckle distance. Local fingertip coordinates use a palm-based
coordinate system. Calculated geometry is invariant under ideal uniform scaling,
translation and 3D rotation of the same points. Changing perspective, occlusion,
mirroring and inaccurate landmark estimates are outside that guarantee.

| Group | Count | Examples / purpose |
|---|---:|---|
| Per finger | 5 × 10 = 50 | Two angles, base proxy angle, straightness, extended, bent, tip-to-palm distance and three local tip coordinates |
| All fingertip pairs | 10 | Thumb/index contact and close or spread fingers |
| Thumb tip to MCP/PIP/DIP of four fingers | 12 | Thumb position in closed hands |
| Adjacent fingers | 3 × 3 = 9 | Spread angle, geometric crossing indicator and relative depth difference |
| **Total** | **81** | |

The crossing indicator is a geometric approximation; perspective can mislead it.
The precision/recall table checks extended/bent indicators. It does not prove
that the remaining 71 numeric features are correct or useful. Inspect angles,
fingertip distances and difficult photos too.

## Precision and recall per finger

For the question **is the index finger extended?**:

- TP: both you and the code say extended.
- FP: you say bent or intermediate; the code says extended.
- FN: you say extended; the code says bent/intermediate or misses the hand.

`precision = TP / (TP + FP)`

`recall = TP / (TP + FN)`

**Calculation example, not a project result:** TP=8, FP=2 and FN=1 yield
precision 8/10=0.80 and recall 8/9≈0.89. Precision alone is insufficient: making
few positive predictions can miss many genuinely extended fingers.

The app evaluates both extension and bending for each finger. Intermediate is
negative for both tasks. Unscorable assessments are excluded and counted
separately. Missing hand detection produces no positive prediction and counts
as FN for a positive human assessment. Output also includes detection coverage
and complete-state correctness. Zero denominators yield blank/null scores.

Assess negative and difficult cases too. Twenty nearly identical photos of one
pose say little about other poses. You may define acceptance criteria in advance,
such as precision ≥0.90 and recall ≥0.80 for a specific finger. These are chosen
research goals, not standards or guaranteed performance. Always report TP/FP/FN
and sample counts too.

## Locate errors

1. Is the human assessment clear? Otherwise choose unscorable.
2. Are landmarks positioned correctly? Incorrect points can make correct angle
   formulas produce incorrect states.
3. Do angles and straightness match simple geometric examples? Run the tests.
4. Does the threshold work across varied development photos?
5. Do additional features help the letter classifier on held-out images?

High precision for extension differs from high precision for letter A. Feature
inspection evaluates a geometric property; model comparison evaluates complete
letter labels.

## Try the calculations in Python

```python
from gesturelab.features.finger_features import angle, classify_finger

print(angle([-1, 0, 0], [0, 0, 0], [1, 0, 0]))  # 180
print(classify_finger(170, 175, 0.95, "index"))  # (True, False)
print(classify_finger(110, 175, 0.75, "index"))  # (False, True)
print(classify_finger(140, 145, 0.75, "index"))  # (False, False)
```

```bash
python -m unittest discover -s tests -p test_finger_module.py -v
```

These tests check calculations, boundaries and normalization. Human assessments
remain necessary to evaluate quality on real hands.
