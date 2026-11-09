"""Match local Kaggle originals to the frozen A-Z development cohort."""
import csv
import hashlib
from collections import Counter
from pathlib import Path

from research.data import Detector, decode_image
from research.kaggle import LETTERS, _entries


ROOT = Path(__file__).resolve().parents[1]


def plot_rejections(by_letter, out):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 9), layout="constrained")
    ax = fig.subplots()
    bars = ax.barh([r["letter"] for r in by_letter],
                   [float(r["rejected_percent"]) for r in by_letter], color="#177873")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    ax.invert_yaxis()
    ax.margins(x=0.15)
    ax.set(title="Uitval bij handdetectie per ASL-letter",
           xlabel="Afgewezen trainingsfoto’s (%)")
    fig.savefig(out / "uitval_per_letter.png", dpi=150)


def main():
    reference = ROOT / "results/angles_2d_01/cv_predictions.csv"
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        accepted = {r["id"]: r["label"] for r in csv.DictReader(handle)}
    if len(accepted) != 60251:
        raise ValueError("Unexpected historical cohort size.")

    train, tests, seen = [], [], {}
    detector = Detector()
    try:
        for entry in _entries(ROOT / "data/asl-alphabet-download", 500):
            path = entry["source"]
            image = decode_image(path.read_bytes())
            key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()[:24]
            label = entry["label"]
            row = dict(id=key, letter=label, filename=path.name, original_path=str(path))
            if entry["split"] == "test":
                tests.append(dict(row, detection_now=detector.analyze(image)["status"],
                                  duplicate_train_filename=seen.get(key, "")))
                continue
            if key in seen or (key in accepted and accepted[key] != label):
                raise ValueError(f"Unexpected duplicate or label mismatch: {path}")
            seen[key] = path.name
            train.append(dict(row, historical_status="accepted" if key in accepted else "rejected"))
    finally:
        detector.close()
    if len(train) != 78000 or accepted.keys() - seen.keys():
        raise ValueError("Local images and historical cohort could not be reconciled.")
    rejected = [r for r in train if r["historical_status"] == "rejected"]
    counts = Counter(r["letter"] for r in rejected)
    totals = Counter(r["letter"] for r in train)
    by_letter = [dict(letter=l, total=totals[l], accepted=totals[l]-counts[l],
                      rejected=counts[l], rejected_percent=round(100*counts[l]/totals[l], 2))
                 for l in sorted(LETTERS, key=lambda l: -counts[l]/totals[l])]
    out = ROOT / "runs/historical_image_audit"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("all_train_images", train), ("rejected_images", rejected),
                       ("by_letter", by_letter), ("test_images", tests)]:
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_rejections(by_letter, out)
    print(f"{len(train)} train images, {len(rejected)} rejected. Reports: {out}")


if __name__ == "__main__":
    main()
