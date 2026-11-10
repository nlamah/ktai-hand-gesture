# Live comparisons and capture collection

See [RQ3/README.md](../RQ3/README.md) for the question, completed experiments, findings and reproduction commands. This guide covers collecting new examples.

## Compare models freely

```bash
conda activate ktai-hand-gesture
python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib
```

This starts with the G/H rotation model. Keys: **1** Shin DS, **2** world joints DSJ, **9** G/H rotation, **7/8** matched mirror reference/mirror training, **0** right normalization. Without `--model`, the default is DSJ.

**Space → intended letter → Enter** saves the frame, exact detector input, landmarks, model hash and prediction under `captures/confusions/`. **L** toggles landmarks, **N** starts a round and **Q/Esc** exits. Keep successful examples too. Free captures support error analysis; they do not estimate precision/F1 or error frequency and are not automatically used for training.

The four current paper examples are in `results/gh_live_diagnostics_20261107/`. They are separate from the 22 withdrawn guided G/H captures. Withdrawn attempts are not research evidence; their exclusion is recorded in `results/gh_capture_reset_20261107/withdrawal.json`.

## Optional predefined evaluation

Fix tasks in advance and retain correct, incorrect and failed-detection attempts. Hide predictions during collection and evaluate both models on the same saved images. Fix hand, lighting, distance, guidance, repetitions and participant roles. Anyone whose errors influenced development is a development participant.

| Goal | Start | Protocol |
|---|---|---|
| G/H viewpoints: 40 attempts per person | `python -m research.gh_view_capture --participants 1` | [G/H capture](GH_CAPTURE.md) |
| Shin DS versus DSJ, five alphabets: 130 attempts per person | `python -m research.alphabet_capture` | [Alphabet capture and evaluation](ALPHABET_CAPTURE.md) |

The alphabet evaluator compares DS and DSJ, not all eight feature variants. The G/H tool records a protocol but has no current public evaluator for a new session. `RQ3 live_examples` replays only the four existing free examples. A new G/H evaluation therefore requires analysis of the new sessions.

Check G/H execution beforehand using [ASL University / Vicars](https://www.lifeprint.com/asl101/topics/feedback.htm): the palm faces inward or sideways. Palm-facing images are an experimental control. Screen left/right alone cannot establish correct execution.

## Outstanding group contributions

- **RQ1:** targeted/general augmentation methods, code and results.
- **RQ2:** executable code for the supplied detection-recovery text and visual review of recovered landmarks.
- **Shared paper:** abstract, missing RQ2 figure/code appendices/group bibliography, and author checks of sources and conclusions.

See the [integration checklist](../paper/INTEGRATION_TODO.md). New capture sessions are optional further work, not completed final validation.
