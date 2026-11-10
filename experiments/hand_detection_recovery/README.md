# Hand Detection Recovery — RQ2

The maintained [RQ2 guide](../../RQ2/README.md) covers the question, Conda setup, raw-image dataset layout, settings, outputs, visual review, verification and repeatability.

From the repository root in the existing Conda environment:

```bash
conda activate ktai-hand-gesture
python -m RQ2 list
python -m RQ2 recovery --name rq2_check_110 --limit 110 --dry-run
python -m RQ2 recovery --name rq2_check_110 --limit 110
python -m RQ2 recovery --name rq2_check_110 --action verify
python -m RQ2 recovery --name rq2_check_110 --action report
```

Reproduction runs baseline, preprocessing, retry, analysis, verification and figures. The numerical implementation and direct script commands remain available in this directory and `research/recovery_experiment.py`.

**Use a new name to preserve previous results.** Reproduction deletes an existing output directory with the same name unless `--resume` is supplied. Resume with the same limit and settings:

```bash
python -m RQ2 recovery --name rq2_check_110 --limit 110 --resume
```

Outputs are written under `experiments/hand_detection_recovery/runs/<name>/`. Copy the visual review template before entering judgments; reporting regenerates that template. Raw images are required for detection and image rendering; a landmark cache alone is insufficient. The current project uses `requirements.txt`, not `environment.yml`.
