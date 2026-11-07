# Standalone G/H rotation

This experiment isolates **adding 180-degree turns around the vertical camera y axis to G/H fitting examples**. It uses the historical original-data 835-input SVM as reference. It contains no mirror augmentation, webcam calibration or additional prediction-routing rule. Both arms remain ordinary 26-letter classifiers.

The transformation is `(dx,dy,dz) -> (-dx,dy,-dz)` for the 210 image-landmark direction vectors. It is equivalent to a wrist-centred landmark rotation. It changes both horizontal direction and depth, unlike a reflection. Pair distances and intrinsic joint measurements stay unchanged. The transformation is geometric; it cannot render hidden surfaces or simulate MediaPipe errors on a real hand back.

Both arms use the same 60,251 accepted original images and frozen five filename-block folds. Training-only G/H derivatives inherit their source's fold. G/H originals and turns each have weight 0.5; other originals have weight one. The full candidate uses 65,052 fitting rows from the same 60,251 sources. Original-fit imputation/scaling, numeric RBF gamma and balanced class priors are shared. Validation uses unchanged originals only. A freshly fitted reference exactly reproduces all historical held-out predictions.

| Measure | Original only | G/H y180 |
|---|---:|---:|
| Accuracy | 97.74776% | 97.74610% |
| Macro-precision | 97.53902% | 97.53874% |
| Macro-F1 | 0.97524391 | 0.97523305 |
| Correct | 58,894 | 58,893 |

Four errors are corrected and five correct predictions lost: net **-1**. G and H recall stay unchanged; their precision and F1 decline slightly. This gives no overall benefit on original views. One known back-view H is recovered, but both known back-view G examples remain wrong. All previously selected camera cases are diagnostic controls, never fitting data in this experiment and never an independent test.

The [full report](../results/rotation_GH_y180_01/report.md) contains all 26 letter scores, fold changes, camera diagnostics and limitations. The paper reports the result separately in Appendix E. Computational verification passes for all metrics, frozen inputs, model replay, shared preprocessing, source weights and source-group separation. Filename blocks are not verified signers; there is no independent back-view evaluation. Conditional original-view scores do not demonstrate live robustness.

## Reproduce and try live

```bash
conda activate ktai-hand-gesture
python -m research.studies.gh_rotation_standalone --stage freeze
python -m research.studies.gh_rotation_standalone --stage cv
python -m research.studies.gh_rotation_standalone --stage train
python -m research.studies.report_gh_rotation_standalone
python -m research.live_camera --model models/asl_shin_angles_GH_y180.joblib
```

Compare **World joints [2]** with **G/H y180 only [9]**. The historical world-joint reference remains the default. Existing candidates and their artifacts are retained: multi-angle G/H rotation [5], mirror training [8], and the earlier combined mirror/rotation/calibration model available through `--model models/asl_gh_vertical_adapter.joblib`. Their outcomes answer different questions and are not pooled with this experiment.
