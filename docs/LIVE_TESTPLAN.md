# Live model comparison and optional data collection

The [RQ3 guide](../RQ3/README.md) covers the controlled feature study. This guide covers inspecting models and collecting new examples; these activities are not completed final validation.

## Inspect predictions

From the repository root:

```bash
conda activate ktai-hand-gesture
python -m research.live_camera
```

Keys: **1** shared Shin DS, **2** world-joint DSJ (default), **A** additional existing model variants, **L** landmarks, **N** new round, **Q/Esc** close.

**Space -> intended letter -> Enter** saves the displayed frame, exact detector input, landmarks, model hash and prediction under `captures/confusions/`. Keep successful examples too. Free captures support error analysis; they do not estimate precision/F1 or error frequency and are not automatically used for training. The preview is reflected for convenience; model inputs remain unchanged.

For feature inspection in the browser:

```bash
python -m streamlit run app.py
```

Select DS or DSJ, enable Camera on, optionally enter the intended letter and save a capture. The intended letter selects feature expectations and never controls the predicted letter. Only one tool can use the camera at a time.

## Optional predefined evaluation

Fix tasks and models before collection. Retain correct, incorrect and failed-detection attempts and hide predictions during recording. Keep lighting, camera, distance, guidance and participant roles consistent. Anyone whose results influenced development is a development participant.

```bash
python -m research.alphabet_capture
```

The [alphabet guide](ALPHABET_CAPTURE.md) describes the five A-Z rounds (130 attempts per participant), pause/resume and paired evaluation of DS and DSJ on the same saved images. It does not evaluate all eight feature combinations. Report detection coverage and classification separately, and distinguish new participants from those involved in development. Obtain participant agreement before sharing captured images.
