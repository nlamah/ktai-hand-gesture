"""Compare MediaPipe hand-detection thresholds on development images without modifying a dataset."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

from PIL import Image

from gesturelab.datasets.data import Detector, MODEL_SHA256, read_manifest

BASELINE = (.5, .5)


def parse_confidences(value):
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Confidence values must be comma-separated numbers.") from error
    if not values or any(not 0 <= item <= 1 for item in values):
        raise argparse.ArgumentTypeError("Provide one or more confidence values between 0 and 1.")
    return tuple(dict.fromkeys(values))


def select_development_samples(samples, per_label):
    if per_label < 0:
        raise ValueError("The per-label sample limit cannot be negative.")
    groups = {}
    for sample in samples:
        if sample.get("included") is True and sample.get("role") == "development":
            groups.setdefault(sample["label"], []).append(sample)
    selected = []
    for label in sorted(groups):
        group = groups[label]
        group.sort(key=lambda sample: (
            hashlib.sha256(sample["id"].encode("utf-8")).hexdigest(), sample["id"]))
        selected.extend(group[:per_label] if per_label else group)
    return selected


def threshold_pairs(detection_confidences, presence_confidences):
    pairs = list(itertools.product(detection_confidences, presence_confidences))
    if BASELINE not in pairs:
        pairs.insert(0, BASELINE)
    else:
        pairs.remove(BASELINE)
        pairs.insert(0, BASELINE)
    return pairs


def tune_project(project, out, detection_confidences, presence_confidences, per_label=40):
    project = Path(project).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    manifest = read_manifest(project)
    samples = select_development_samples(manifest["samples"], per_label)
    if not samples:
        raise ValueError("No included development images were found in this project.")
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    pairs = threshold_pairs(detection_confidences, presence_confidences)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    print(f"Testing {len(samples)} development images across {len(pairs)} threshold pairs.", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.tmp-", dir=out.parent) as staging_dir:
        staging = Path(staging_dir)
        baseline_statuses = {}
        with (staging / "sample_results.csv").open("w", newline="", encoding="utf-8") as results_file:
            results = csv.DictWriter(results_file, fieldnames=(
                "sample_id", "label", "image_path", "baseline_status",
                "min_hand_detection_confidence", "min_hand_presence_confidence",
                "status", "n_hands", "recovered_vs_baseline", "lost_vs_baseline", "error"))
            results.writeheader()
            for index, (detection_confidence, presence_confidence) in enumerate(pairs, start=1):
                print(
                    f"[{index}/{len(pairs)}] detection={detection_confidence:.2f} "
                    f"presence={presence_confidence:.2f}",
                    flush=True)
                counts = {}
                recovered = lost = 0
                detector = Detector(detection_confidence, presence_confidence)
                try:
                    for sample in samples:
                        with Image.open(project / sample["image_path"]) as image:
                            detection = detector.analyze(image)
                        status = detection["status"]
                        counts[status] = counts.get(status, 0) + 1
                        if (detection_confidence, presence_confidence) == BASELINE:
                            baseline_statuses[sample["id"]] = status
                        baseline = baseline_statuses[sample["id"]]
                        newly_usable = baseline != "ok" and status == "ok"
                        no_longer_usable = baseline == "ok" and status != "ok"
                        recovered += newly_usable
                        lost += no_longer_usable
                        results.writerow({
                            "sample_id": sample["id"],
                            "label": sample["label"],
                            "image_path": sample["image_path"],
                            "baseline_status": baseline,
                            "min_hand_detection_confidence": detection_confidence,
                            "min_hand_presence_confidence": presence_confidence,
                            "status": status,
                            "n_hands": detection.get("n_hands", ""),
                            "recovered_vs_baseline": newly_usable,
                            "lost_vs_baseline": no_longer_usable,
                            "error": detection.get("error", ""),
                        })
                finally:
                    detector.close()
                usable = counts.get("ok", 0)
                summary_rows.append({
                    "min_hand_detection_confidence": detection_confidence,
                    "min_hand_presence_confidence": presence_confidence,
                    "images": len(samples),
                    "usable": usable,
                    "coverage": usable / len(samples),
                    "recovered_vs_baseline": recovered,
                    "lost_vs_baseline": lost,
                    "status_counts": counts,
                })

        with (staging / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
            columns = ("min_hand_detection_confidence", "min_hand_presence_confidence",
                       "images", "usable", "coverage", "recovered_vs_baseline",
                       "lost_vs_baseline", "no_hand", "multiple_hands", "invalid_geometry")
            writer = csv.DictWriter(summary_file, fieldnames=columns)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({
                    **{key: row[key] for key in columns if key in row},
                    **{key: row["status_counts"].get(key, 0)
                       for key in ("no_hand", "multiple_hands", "invalid_geometry")},
                })

        run = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": str(project),
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "model_sha256": MODEL_SHA256,
            "role": "development",
            "per_label_limit": per_label or None,
            "labels": sorted({sample["label"] for sample in samples}),
            "sample_count": len(samples),
            "baseline": {
                "min_hand_detection_confidence": BASELINE[0],
                "min_hand_presence_confidence": BASELINE[1],
            },
            "candidates": summary_rows,
        }
        (staging / "run.json").write_text(
            json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        if out.exists():
            raise FileExistsError(f"Output directory already exists: {out}")
        staging.rename(out)
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="data/asl_kaggle")
    parser.add_argument("--out", required=True, help="New output directory; existing directories are refused.")
    parser.add_argument("--detection-confidences", type=parse_confidences, default=parse_confidences("0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--presence-confidences", type=parse_confidences, default=parse_confidences("0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--per-label", type=int, default=40,
                        help="Deterministically sample up to this many development images per label; 0 uses all.")
    args = parser.parse_args()
    summary = tune_project(args.project, args.out, args.detection_confidences,
                           args.presence_confidences, args.per_label)
    print(f"Threshold results: {Path(args.out).expanduser().resolve()}")
    for row in summary:
        print(
            f"detection={row['min_hand_detection_confidence']:.2f} "
            f"presence={row['min_hand_presence_confidence']:.2f} "
            f"coverage={row['coverage']:.1%} "
            f"recovered={row['recovered_vs_baseline']} "
            f"lost={row['lost_vs_baseline']}")


if __name__ == "__main__":
    main()
