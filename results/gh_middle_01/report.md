# Exploratory middle-finger-only G/H diagnostic

This additional ablation was proposed after inspecting the failed three-representation experiment. It is post-hoc, not an independent confirmation. Only the middle-PIP angle from the 3D estimate and 2D pixel projection is used; no webcam examples enter fitting. The two-class model always outputs G or H and cannot recognize other letters.

Same frozen five-fold G/H accuracy: 94.085%; macro-F1 0.94084.

| Capture | Intended | Prediction | 3D PIP | 2D PIP |
|---|---|---|---:|---:|
| 20261031T152108010733Z_9233886f | G | G | 102.1 | 50.0 |
| 20261031T152116143793Z_11756e66 | H | H | 155.8 | 178.7 |
| 20261031T181839806179Z_9ef071cc | G | G | 95.3 | 26.3 |

Reproduce with `python -m research.studies.gh_middle_diagnostic`. A fresh, balanced webcam round is needed before considering a second-stage recognizer.
