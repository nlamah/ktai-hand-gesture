# G/H capture with explicit hand-position instructions

This is an optional collection protocol. See [RQ3/README.md](../RQ3/README.md) for completed experiments and the [live guide](LIVE_TESTPLAN.md) for free model comparisons.

The two guided sessions from 7 November were withdrawn because execution was incorrect. Their captures and derived examples are excluded from the current study. The exclusion record is `results/gh_capture_reset_20261107/withdrawal.json`; withdrawn sessions cannot be resumed.

## Start

Close other camera windows and run from the repository root:

```bash
conda activate ktai-hand-gesture
python -m research.gh_view_capture --participants 1
```

This plans 40 attempts: two letters, four views and five repetitions. Use `--participants 2` or `--participants 3` for more people; the default is three. Each new session receives a new directory. You can pause and resume.

## Hand position

The large panel beside the camera shows:

1. **Hand:** use the specified physical right or left hand. A reflected preview does not change which hand you use.
2. **Letter:** for G, extend the index finger and thumb approximately parallel and bend the other fingers. For H, extend the index and middle fingers together, bend the ring/little fingers and fold the thumb.
3. **View:** form the letter first, then turn the whole hand while preserving finger shape. Screen pointing direction may change naturally.

| Instruction | Camera view |
|---|---|
| PALM TOWARDS CAMERA | Palm visible; knuckles face away. Experimental palm control. |
| OBLIQUE: PALM VISIBLE | Turn roughly 45 degrees from the palm control; part of the palm stays visible. |
| OBLIQUE: BACK VISIBLE | Turn roughly 45 degrees from the hand-back view; part of the back stays visible. |
| BACK TOWARDS CAMERA | Knuckles and hand back visible; palm faces away. |

The diagram is a **top view**: orange marks the palm side, blue the back, and the camera is below. **The arrow indicates palm direction, not finger direction.** It is a schematic, not an anatomical model. Angles are qualitative instructions, not calibrated measurements.

[ASL University](https://www.lifeprint.com/asl101/topics/feedback.htm) describes inward/sideways palms for usual G/H. Palm-facing attempts are an experimental control. Review execution beforehand, preferably with someone familiar with ASL.

## Controls

- **Enter at the start:** confirm participant and hand.
- **Space or capture button:** freeze the frame.
- **Enter after capture:** save the attempt with its fixed intended letter.
- **Esc during confirmation:** retry a procedure error, such as forming the wrong requested letter.
- **Q/Esc in the live view:** pause and release the camera.

Predictions remain hidden. Keep classification errors and failed detections. Relax and reform the sign between attempts. Keep hand, camera, distance and lighting consistent, with the whole hand and wrist visible.

## Resume

Use the actual session directory printed by the terminal:

```bash
python -m research.gh_view_capture --resume captures/gh_viewpoint/GH_views_DATE_ID
python -m research.gh_view_capture --resume captures/gh_viewpoint/GH_views_DATE_ID --status
```

New sessions record instruction version `explicit-gh-pose-v3` and the English display version `large-en-gh-guidance-v1`. Keep display/instruction versions identifiable when comparing sessions.

## Storage and interpretation

Each saved attempt includes the original frame, exact detector input, preview without a prediction and `capture.json`: intended letter, participant, hand, view, status, landmarks and model information. Both comparison models later receive the same saved inputs; collection does not train a model.

P01 is development. Additional participants are reserved evaluation only while model and analysis choices remain fixed. Captures stay local under `captures/gh_viewpoint/` and are not automatically training data. There is no current public evaluator for a newly collected G/H session; the existing `live_examples` workflow concerns four fixed free captures.

## Occlusion experiment already completed

Landmark rotation alone cannot hide a finger or produce new detector errors. A coarse source-derived hand volume was rendered and redetected: MediaPipe accepted 0/128 renders versus 64/64 source photographs. A separate 2%/5% hidden-point noise proxy trained ten fold models but added no gain over pure rotation on the original or clean artificially rotated tests. These are not validated hand-back observations. Results and commands are in [the RQ3 guide](../RQ3/README.md); deployment models were not replaced.
