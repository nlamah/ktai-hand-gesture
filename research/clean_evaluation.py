"""Grouped evaluation, review copies and final SVM for a clean detection run."""
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2

import joblib
import numpy as np
import sklearn

from research.live_model import ANGLE_COLUMNS, BUNDLE_VERSION, SHIN_VERSION, classifier, load_bundle
from research.kaggle import LETTERS
from research.no_hand_gallery import build


def evaluate_and_fit(out):
    from research.clean_training import write_csv

    with (out / "all_images.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    build(out / "no_hand.csv")
    accepted = [r for r in rows if r["status"] == "ok"]
    with np.load(out / "inputs.npz") as cache:
        X = np.column_stack((cache["X"], cache["angles"]))
    metadata = json.loads((out / "inputs_metadata.json").read_text())
    if [s["id"] for s in metadata["samples"]] != [r["id"] for r in accepted]:
        raise ValueError("Feature metadata does not match the photo inventory.")
    y = np.array([r["letter"] for r in accepted])
    groups = np.array([r["group"] for r in accepted])
    predictions = {}
    for group in np.unique(groups):
        held = groups == group
        if sorted(set(y[~held])) != list(LETTERS):
            raise ValueError(f"Insufficient training samples for A-Z when holding out {group}.")
        print(f"Evaluating {group} ({held.sum()} photos)", flush=True)
        svm = classifier("svm", "shin_angles", (~held).sum())
        svm.fit(X[~held], y[~held])
        for index, predicted in zip(np.flatnonzero(held), svm.predict(X[held])):
            predictions[accepted[index]["id"]] = predicted
    counts = Counter()
    for r in rows:
        predicted = predictions.get(r["id"], "")
        r["voorspeld"] = predicted
        if predicted:
            result = "goed" if predicted == r["letter"] else "verkeerde_letter"
        else:
            if r["status"] == "ok":
                raise ValueError("Missing held-out prediction.")
            result = "geen_letter"
            folder = out / "geen_letter" / r["letter"]
            folder.mkdir(parents=True, exist_ok=True)
            copy2(r["original_path"], folder / r["filename"])
        r["uitkomst"] = result
        counts[r["letter"], result] += 1
        counts[r["letter"], r["status"]] += 1
    write_csv(out / "per_foto.csv", rows)
    summary = []
    for letter in LETTERS:
        goede = counts[letter, "goed"]
        verkeerde = counts[letter, "verkeerde_letter"]
        geen = counts[letter, "geen_letter"]
        total = goede + verkeerde + geen
        summary.append(dict(letter=letter, totaal=total, goed=goede, fout=verkeerde+geen,
                            verkeerde_letter=verkeerde, geen_letter=geen,
                            geen_hand=counts[letter, "no_hand"], meerdere_handen=counts[letter, "multiple_hands"],
                            ongeldige_geometrie=counts[letter, "invalid_geometry"],
                            correct_pct=round(100*goede/total, 2)))
    write_csv(out / "per_letter.csv", summary)
    print(f"Final training on {len(y)} accepted original images", flush=True)
    model = classifier("svm", "shin_angles", len(y)).fit(X, y)
    info = dict(version=BUNDLE_VERSION, input_variant="shin_angles", shin_version=SHIN_VERSION,
                angle_columns=list(ANGLE_COLUMNS), sklearn_version=sklearn.__version__, training_images=len(y),
                source_hashes=metadata["hashes"],
                input_cache_sha256=sha256((out / "inputs.npz").read_bytes()).hexdigest(),
                evaluation="Leave one numbered 500-image block per letter out at a time.")
    temporary = out / "asl_clean.tmp"
    joblib.dump({**info, "estimator": model}, temporary, compress=3)
    load_bundle(temporary)
    temporary.replace(out / "asl_clean.joblib")
    (out / "asl_clean.json").write_text(json.dumps(info, indent=2))
