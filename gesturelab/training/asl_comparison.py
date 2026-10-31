"""Compare saved ASL validation predictions without fitting or touching test data."""
import json
from pathlib import Path

import pandas as pd

from gesturelab.datasets.data import atomic_json


def _read_run(name, path):
    path = Path(path).resolve()
    report = json.loads((path / "training.json").read_text(encoding="utf-8"))
    split = json.loads((path / "split.json").read_text(encoding="utf-8"))
    if report.get("test_evaluated") is not False or split.get("test_evaluated") is not False:
        raise ValueError(f"{name}: this run does not declare test data untouched.")
    if report.get("dataset_fingerprint") != split.get("dataset_fingerprint"):
        raise ValueError(f"{name}: report and split fingerprints differ.")
    try:
        validation_ids = set(split["validation_ids"])
        development_ids = set(split["development_ids"])
        test_ids = set(split["test_ids"])
        labels = set(split["labels"])
        if not validation_ids <= development_ids or validation_ids & test_ids:
            raise ValueError(f"{name}: validation IDs overlap final-test or leave development.")
        if (report["training_ids"] != split["training_ids"]
                or report["validation_ids"] != split["validation_ids"]):
            raise ValueError(f"{name}: report IDs do not match the saved split.")
        predictions = pd.read_csv(
            path / "validation_predictions.csv", dtype={"id": str, "label": str})
        if not {"id", "label", "prediction"} <= set(predictions.columns):
            raise ValueError(f"{name}: validation predictions are missing required columns.")
        if predictions.label.isna().any() or predictions.prediction.isna().any():
            raise ValueError(f"{name}: validation predictions contain empty labels.")
        if not set(predictions.label) <= labels:
            raise ValueError(f"{name}: validation predictions contain unknown labels.")
        if report["validation"]["n"] != len(validation_ids):
            raise ValueError(f"{name}: validation metric count does not match frozen IDs.")
    except KeyError as error:
        raise ValueError(f"{name}: incomplete ASL run metadata ({error}).") from error
    expected_ids = validation_ids
    if predictions.id.duplicated().any() or set(predictions.id) != expected_ids:
        raise ValueError(f"{name}: validation predictions do not match frozen IDs.")
    return path, report, split, predictions


def compare_asl_runs(runs, output):
    """Require identical held-out IDs, labels, groups and dataset across saved runs."""
    if not isinstance(runs, dict) or len(runs) < 2:
        raise ValueError("Provide at least two named model runs.")
    if any(not isinstance(name, str) or not name.strip() for name in runs):
        raise ValueError("Every model run must have a non-empty name.")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Comparison output already exists. Choose a new directory.")
    loaded = {name: _read_run(name, path) for name, path in runs.items()}
    reference_split = next(iter(loaded.values()))[2]
    signature = (
        reference_split["dataset_fingerprint"],
        reference_split["training_ids"],
        reference_split["validation_ids"],
        reference_split["labels"],
        reference_split["grouping"],
        reference_split["role_map_sha256"],
    )
    base_predictions = next(iter(loaded.values()))[3]
    base_labels = dict(zip(base_predictions.id.astype(str), base_predictions.label.astype(str)))
    rows = []
    aligned = pd.DataFrame({"id": sorted(base_labels)})
    aligned["label"] = aligned.id.map(base_labels)
    for name, (path, report, split, predictions) in loaded.items():
        current_signature = (
            split["dataset_fingerprint"], split["training_ids"],
            split["validation_ids"], split["labels"], split["grouping"],
            split["role_map_sha256"],
        )
        if current_signature != signature:
            raise ValueError(f"{name}: model runs do not share identical frozen data roles.")
        if dict(zip(predictions.id.astype(str), predictions.label.astype(str))) != base_labels:
            raise ValueError(f"{name}: validation ground-truth labels differ.")
        score = report["validation"]
        rows.append({
            "run": name,
            "model": report["route"],
            "input_variant": report["input_variant"],
            "n_features": report["n_features"],
            "validation_accuracy": score["accuracy"],
            "validation_macro_f1": score["macro_f1"],
            "validation_coverage": score["coverage"],
            "validation_n": score["n"],
            "test_evaluated": False,
            "run_path": str(path),
        })
        indexed = predictions.set_index(predictions.id.astype(str))
        aligned[name + "_prediction"] = aligned.id.map(indexed.prediction.astype(str))
        aligned[name + "_correct"] = aligned[name + "_prediction"] == aligned.label

    output.mkdir(parents=True)
    summary = pd.DataFrame(rows).sort_values(
        "validation_macro_f1", ascending=False, kind="stable")
    summary.to_csv(output / "summary.csv", index=False)
    aligned.to_csv(output / "paired_validation_predictions.csv", index=False)
    (output / "report.html").write_text(
        "<!doctype html><html lang=\"nl\"><meta charset=\"utf-8\">"
        "<title>ASL modelroute vergelijking</title><h1>ASL modelroute vergelijking</h1>"
        "<p>Alleen dezelfde frozen development-validatiebeelden; geen final-test-evaluatie.</p>"
        + summary.to_html(index=False, escape=True)
        + "</html>\n",
        encoding="utf-8",
    )
    atomic_json(output / "comparison.json", {
        "dataset_fingerprint": signature[0],
        "training_ids": signature[1],
        "validation_ids": signature[2],
        "labels": signature[3],
        "grouping": signature[4],
        "role_map_sha256": signature[5],
        "runs": {name: str(value[0]) for name, value in loaded.items()},
        "test_evaluated": False,
        "note": "Development validation comparison only; repeated model selection may overfit this split.",
    })
    return output
