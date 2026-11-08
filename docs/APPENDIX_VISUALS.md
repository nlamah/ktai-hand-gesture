# RQ3 figures and source files

See [RQ3/README.md](../RQ3/README.md) for experiments and results. This guide covers the paper figures.

## Paper versions

The standalone technical supplement includes feature-concept diagrams, the original C screenshot and sharp detail cards, selected A/H mirror examples, G/H training skeleton rotations, the occlusion approximation and four selected live examples. The integrated group appendix contains the compact protocol, screenshot, importance figure and four live examples.

The 22 withdrawn guided G/H captures are excluded. The four free live examples are a separate set and do not estimate accuracy.

## Feature tool

```bash
conda activate ktai-hand-gesture
python -m streamlit run app.py
```

Inspect an existing local capture without opening the camera:

```bash
python -m streamlit run app.py -- --capture /path/to/capture/capture.json
```

Use an actual local path; captures are excluded from Git. The intended letter selects workbook expectations only. A green card does not independently validate anatomy or ASL execution.

The original screenshot is `paper/figures/feature_tool_screenshot.png`. The two detail cards are typeset vector text showing the same expected/observed categories; they are not new measurements. Source rectangles, text and hashes are in `results/feature_illustrations_01/detail_cards.json`.

## Regenerate figures

The included figures are sufficient to compile the paper. These commands overwrite figure outputs and require their source inputs:

| Output | Command | Required inputs |
|---|---|---|
| Feature-concept diagrams | `python -m research.studies.feature_illustrations` | Included screenshot, workbook and measurement rules |
| G/H training skeletons | `python -m research.studies.gh_training_rotation_figure` | Frozen manifest/cache and raw Kaggle photographs |
| Four selected live examples | `python -m RQ3 live_examples` | Four specific private captures, cache and both frozen models |
| Permutation figures | `python -m RQ3 features --action report` | Published feature results |

The three group figures are supplied under `paper/group_integration/figures/`.

## Provenance

- `results/feature_illustrations_01/`: conceptual drawings, screenshot and detail cards. Drawings are not measured hands.
- `results/appendix_visuals_01/`: selection and feature records for the A1208/H1198 mirror illustrations. Selected recoveries do not estimate a success rate.
- `results/gh_training_rotation_01/`: G1128/H1198, original/rotated points and parity with training augmentation. Rotation produces no new hand-back photograph or detector output.
- `results/gh_live_diagnostics_20261107/`: four current free examples, model replay and hashes.

Not spread does not imply contact. Thumb across/beside the palm is approximated by signed thumb-MCP deviation. The geometric rules are not independently validated anatomy. Regenerating photographic figures requires the specific raw images or private captures.

## Current group document

The shortened group body fits five pages, with references continuing onto page 6. Its three RQ3 figures retain the verified 76 permutation values, unchanged original feature-tool screenshot and the same four annotated live examples. No new accuracy claim is inferred from selected photos.

The N47 original/dilated/overlay illustration is supplied through `code-snippets/n47-example.tex`, preserving the existing caption and resolving the body reference. The recovery-rate figure and both original coauthor code snippets are retained. Unused template images and the duplicate recovery figure are removed from the current Overleaf project. All active figure paths resolve.
