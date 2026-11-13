"""Replay RQ1 predictions on the exact stored 60-image webcam test dataset.

Run from repository root:
    python -m scripts.youri_reproduce

Needs the three fitted model.joblib files and data/youri_test/manifest.json.
Training and test images should be provided separately for full verification.
The script does NOT train new models or mutate the original evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib

from gesturelab.datasets.data import read_manifest
from gesturelab.features.training_inputs import model_features
from gesturelab.training.experiments import predict_bundle
from gesturelab.validation.metrics import score_rows

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/youri_test"
OUTPUT = ROOT / "exports/youri_reproduced"
LETTERS = ("M", "N", "R", "U")
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODELS = {
    "baseline": ROOT / "experiments/shin2021_kaggle_01/svm/shin2021_combined/model.joblib",
    "targeted_59": ROOT / "experiments/youri_targeted_01/shin2021_combined/model.joblib",
    "n_only_15": ROOT / "experiments/youri_n_only_01/shin2021_combined/model.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test():
    if not (TEST / "manifest.json").exists():
        raise FileNotFoundError(f"Missing test manifest: {TEST / 'manifest.json'}")
    samples = [
        row for row in read_manifest(TEST)["samples"]
        if row.get("role") == "test"
        and row.get("status") == "ok"
        and row.get("included", True)
    ]
    samples.sort(key=lambda row: row["id"])
    counts = Counter(row["label"] for row in samples)
    expected = Counter({letter: 15 for letter in LETTERS})
    if counts != expected:
        raise ValueError(f"Expected 15 usable test samples per M/N/R/U; found {counts}")
    if len({row["id"] for row in samples}) != 60:
        raise ValueError("Duplicate or missing test sample IDs")
    return samples


def predict(model_file: Path, samples: list[dict]):
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing trained model: {model_file}")
    bundle = joblib.load(model_file)
    results = []
    for sample in samples:
        vector = model_features(bundle, sample)
        prediction = str(predict_bundle(bundle, [vector])[0])
        results.append({
            "id": sample["id"],
            "label": sample["label"],
            "prediction": prediction,
        })
    return results


def mutual_confusion(rows, a, b):
    a_to_b = sum(row["label"] == a and row["prediction"] == b for row in rows)
    b_to_a = sum(row["label"] == b and row["prediction"] == a for row in rows)
    denominator = sum(row["label"] in (a, b) for row in rows)
    return {
        f"{a}_to_{b}": a_to_b,
        f"{b}_to_{a}": b_to_a,
        "rate": (a_to_b + b_to_a) / denominator,
    }


def main():
    samples = load_test()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    summary = {"test_samples": 60, "models": {}}

    for name, model_file in MODELS.items():
        rows = predict(model_file, samples)
        all_rows[name] = {row["id"]: row for row in rows}
        score = score_rows(rows, list(LETTERS), prediction_labels=ALPHABET)
        summary["models"][name] = {
            "model_sha256": sha256(model_file),
            "accuracy": score["accuracy"],
            "macro_f1": score["macro_f1"],
            "correct": sum(r["label"] == r["prediction"] for r in rows),
            "n": score["n"],
            "M_N": mutual_confusion(rows, "M", "N"),
            "R_U": mutual_confusion(rows, "R", "U"),
        }
        print(f"{name}: accuracy={score['accuracy']:.4f}, macro_F1={score['macro_f1']:.4f}")

    for candidate in ("targeted_59", "n_only_15"):
        categories = Counter()
        for sample in samples:
            sample_id = sample["id"]
            true_label = sample["label"]
            a_correct = all_rows["baseline"][sample_id]["prediction"] == true_label
            b_correct = all_rows[candidate][sample_id]["prediction"] == true_label
            if a_correct and b_correct:
                category = "both_correct"
            elif a_correct:
                category = "only_baseline_correct"
            elif b_correct:
                category = "only_candidate_correct"
            else:
                category = "both_incorrect"
            categories[category] += 1
        summary["models"][candidate]["paired_vs_baseline"] = dict(categories)

    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "actual", *MODELS])
        for sample in samples:
            writer.writerow([
                sample["id"], sample["label"],
                *(all_rows[name][sample["id"]]["prediction"] for name in MODELS),
            ])

    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved replay to {OUTPUT}")
    print("Expected accuracies: baseline=.6500, targeted_59=.7833, n_only_15=.7333")
    print("N-only results are exploratory (previously viewed test set).")


if __name__ == "__main__":
    main()
