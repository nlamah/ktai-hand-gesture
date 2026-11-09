"""Fresh MediaPipe detection on original A-Z train photos, followed by a new SVM fit."""
import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from research.data import Detector, MODEL_SHA256, decode_image
from research.kaggle import LETTERS, _split_folder, _training_entries
from research.no_hand_gallery import build


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fieldnames=None):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detect_letter(task):
    from research.live_model import detection_inputs

    letter, entries, out = task
    detector = Detector()
    rows, features = [], []
    try:
        for entry in entries:
            path = entry["source"]
            image = decode_image(path.read_bytes())
            key = sha256(str(image.size).encode() + image.tobytes()).hexdigest()[:24]
            detection = detector.analyze(image)
            status = detection["status"]
            if status == "ok":
                features.append(detection_inputs(detection)[0])
            rows.append(dict(id=key, letter=letter, filename=path.name, original_path=str(path),
                             group=entry["source_group"], hands=detection["n_hands"], status=status))
    finally:
        detector.close()
    write_csv(out / f"{letter}.csv", rows)
    np.save(out / f"{letter}.npy", np.asarray(features, dtype=np.float32).reshape(-1, 835))
    print(f"{letter}: {len(rows)} photos, {len(features)} accepted", flush=True)


def main():
    import mediapipe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/clean_training")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    source = _split_folder(ROOT / "data/asl-alphabet-download", "asl_alphabet_train")
    entries = list(_training_entries(source, 500))
    if Counter(e["label"] for e in entries) != {l: 3000 for l in LETTERS}:
        raise ValueError("Expected 3,000 original photos for each A-Z letter.")
    out.mkdir(parents=True, exist_ok=False)
    parts = out / "letters"
    parts.mkdir()
    tasks = [(l, [e for e in entries if e["label"] == l], parts) for l in LETTERS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(detect_letter, tasks))
    rows, arrays = [], []
    for letter in LETTERS:
        with (parts / f"{letter}.csv").open(encoding="utf-8-sig", newline="") as handle:
            group = list(csv.DictReader(handle))
        matrix = np.load(parts / f"{letter}.npy")
        if len(matrix) != sum(r["status"] == "ok" for r in group):
            raise ValueError("Feature rows do not match accepted detections.")
        rows.extend(group)
        arrays.append(matrix)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training pixels found.")
    write_csv(out / "all_images.csv", rows)
    write_csv(out / "no_hand.csv", [r for r in rows if r["status"] == "no_hand"], list(rows[0]))
    build(out / "no_hand.csv")
    counts = Counter((r["letter"], r["status"]) for r in rows)
    summary = [dict(letter=l, **{s: counts[l, s] for s in
                    ("no_hand", "multiple_hands", "invalid_geometry", "ok")},
                    no_hand_percent=round(100 * counts[l, "no_hand"] / 3000, 2)) for l in LETTERS]
    write_csv(out / "by_letter.csv", sorted(summary, key=lambda r: -r["no_hand"]))
    accepted = [r for r in rows if r["status"] == "ok"]
    X = np.concatenate(arrays)
    np.savez_compressed(out / "inputs.npz", X=X[:, :820], angles=X[:, 820:])
    meta = dict(samples=[dict(id=r["id"], label=r["letter"], group=r["group"]) for r in accepted],
                hashes=dict(inventory=sha256((out / "all_images.csv").read_bytes()).hexdigest(),
                            detector=MODEL_SHA256), mediapipe=mediapipe.__version__,
                detection_confidence=0.5, presence_confidence=0.5)
    (out / "inputs_metadata.json").write_text(json.dumps(meta))
    subprocess.run([sys.executable, "-m", "research.live_model", "--fresh-fit", "--cache", str(out),
                    "--out", str(out / "asl_clean.joblib")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
