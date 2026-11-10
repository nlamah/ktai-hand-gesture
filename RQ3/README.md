# RQ3: hand-shape features and joint measurements

This guide describes Ton's contribution. Installation, the input bundle and the shared baseline are covered in the main [README](../README.md). Numbering follows the group paper: RQ1 targeted data collection, RQ2 detection recovery, RQ3 feature comparison.

## Question and main comparison

**Can explicit hand-shape categories and joint measurements supplement or replace Shin's landmark directions?**

| Block | Description | Inputs |
|---|---|---:|
| D | Normalized distances between image landmarks | 190 |
| S | Shin directions between image landmarks; not internal joint angles | 630 |
| E | Own categories: flexion, spreading, crossing, thumb shape and contact | 70 |
| J | Joint measurements from estimated world landmarks | 15 |

D is always present. S, E and J are independently included/omitted: **DS, D, DSE, DE, DSJ, DJ, DSEJ and DEJ**. The full representation has 905 inputs. Workbook-match scores are excluded.

Eight representations, two learners and five fixed folds produce 80 models. Only 60,251 accepted original single-hand detections are included. All variants share sources and test folds. The SVM DS baseline has 97.38815% macro-precision and 0.97386235 macro-F1.

A same-coordinate control derives J from Shin's image points to distinguish representation changes from coordinate-source changes. After training, features are shuffled across 13,000 fixed held-out images: 3,900 group interventions measure fitted-model dependence. Permutation does not refit the model; the eight training variants measure adding/removing information.

## Run

From the repository root, after installation and extracting the frozen input bundle:

```bash
python -m baseline check
python -m RQ3 list
python -m RQ3 features --dry-run
python -m RQ3 features
python -m RQ3 coordinate_control
```

`features` checks inputs, trains both learners, permutes, exports, verifies and reports. Matching checkpoints are resumable. To verify existing results:

```bash
python -m RQ3 features --action verify
python -m RQ3 coordinate_control --action verify
```

Replay requires the inputs and fold models. [workflows.py](workflows.py) defines commands and output directories. Use these public entry points because some experiments require a specific supplied source implementation.

## Findings

- Adding E to DS loses 22 correct SVM predictions.
- Adding world J gains 67; the same-image-coordinate control gains only four.
- Adding both E and J loses three.
- The small world-joint gain does not recur with LightGBM. Without S, the tested categories/joint measurements do not fully replace Shin directions.

These findings concern fixed profiles and this dataset selection. World and image coordinates differ. Folds are filename blocks, not verified people, and were inspected during development. Scores describe exploratory classification after accepted detection, not new-user or webcam accuracy.

Evidence: [feature comparison](../results/feature_factorial_01/) and [coordinate control](../results/direction_joint_01/). Main files are `metrics.csv`, `paired_changes.csv`, `permutation*.csv` and `protocol.json`.

## Brief practical explorations

Alongside the main feature study, I briefly tried a few ideas for practical camera issues:

- **Left/right hands:** mirror training and normalization to detector-Right. These help on an artificial mirror mixture, but neither improves classification on original images. They do not establish better recognition for new left-handed users.
- **G/H views:** rotate the existing training landmarks to try to bridge the palm/back view difference. The model learns that artificial geometry, but two selected live G examples remain H. A coarse occlusion/noise approximation adds no demonstrated improvement.

These are small exploratory follow-ups, not extra main research questions or independent final validation. The systematic eight-way feature comparison remains the core contribution. Detailed protocols/results are retained in the appendix so the attempts can be checked.

### Reproduction details for the exploratory checks

The checks retain their own **835-input DSJ reference**, distinct from DS in the main comparison.

| Attempt | Command and prerequisite | Results |
|---|---|---|
| RGB mirror training | `python -m RQ3 mirrors` | [mirror_01](../results/mirror_01/) |
| Detector-Right normalization | `python -m RQ3 normalization`; run `mirrors` first | [handedness_01](../results/handedness_01/) |
| G/H landmark rotation | `python -m RQ3 gh_rotation` | [rotation_GH_y180_01](../results/rotation_GH_y180_01/) |
| G/H occlusion approximation | `python -m RQ3 occlusion`; run `gh_rotation` first | [gh_occlusion_01](../results/gh_occlusion_01/) |
| Four selected live examples | `python -m RQ3 live_examples` | [gh_live_diagnostics_20261107](../results/gh_live_diagnostics_20261107/) |

Each supports `--dry-run` and `--action verify`. Rendering needs raw photographs; live replay needs the four private captures. Neither is included in the frozen input bundle. Synthetic derivatives stay in their source's fold.

On originals, mirror training loses 26 correct predictions and normalization loses 136. G/H rotation corrects four errors and introduces five. On artificially rotated held-out G/H geometry, macro-F1 rises from 0.8693 to 0.9757; that creates no new photograph or detector output. The selected live cases do not estimate precision/F1 or error frequency. The G/H dataset/model audit is `verification_core.json`.

## Live tool, paper and references

```bash
python -m research.live_camera
```

Keys: **1** DS, **2** DSJ, **9** G/H rotation, **7/8** matched mirror reference/mirror training, **0** right normalization. The default is DSJ. **Space → intended letter → Enter** saves an example. See the [live guide](../docs/LIVE_TESTPLAN.md) for optional capture protocols.

The English contribution is in the [paste-ready LaTeX](../paper/group_integration/group_report_paste_ready.tex) and [technical appendix](../paper/group_integration/ton_appendix_en.tex). The [bibliography](../paper/group_integration/ton_references.bib) contains six sources: Shin, MediaPipe documentation, Kaggle ASL Alphabet, scikit-learn permutation documentation, LightGBM documentation and Vicars for G/H. See the [figure guide](../docs/APPENDIX_VISUALS.md) for provenance.
