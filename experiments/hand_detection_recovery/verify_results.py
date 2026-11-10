"""Recompute recovery counts and verify the paired exports without running detection."""
import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path


def read_csv(run, name):
    with (run / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(run):
    manifest = run / "evidence_manifest.json"
    if manifest.exists():
        for filename, expected in json.loads(manifest.read_text(encoding="utf-8-sig"))["sha256"].items():
            assert sha256((run / filename).read_bytes()).hexdigest() == expected, filename
    baseline = read_csv(run, "baseline.csv")
    missed = read_csv(run, "no_hand.csv")
    paired = read_csv(run, "per_image.csv")
    summary = read_csv(run, "summary.csv")
    processed = read_csv(run, "processed.csv")
    settings = json.loads((run / "settings.json").read_text())
    variants = ("erosie", "dilatatie", "canny_overlay")
    assert len(baseline) == settings["images"]
    assert len({r["id"] for r in baseline}) == len(baseline)
    expected_misses = {r["id"] for r in baseline if r["status"] == "no_hand"}
    assert len(missed) == len(paired) == len(expected_misses)
    assert {r["id"] for r in missed} == {r["id"] for r in paired} == expected_misses
    expected_pairs = {(key, v) for key in expected_misses for v in variants}
    assert len(processed) == len(expected_pairs)
    assert {(r["id"], r["variant"]) for r in processed} == expected_pairs
    detections = [json.loads(line) for line in (run / "retry_landmarks.jsonl").read_text().splitlines()]
    assert len(detections) == len(expected_pairs)
    lookup = {(r["id"], r["variant"]): r for r in detections}
    assert set(lookup) == expected_pairs
    for row in paired:
        successes = [v for v in variants if row[v] == "ok"]
        assert row["recovered_variants"] == ";".join(successes)
        for variant in variants:
            result = lookup[row["id"], variant]
            assert result["retry_status"] == row[variant]
            if row[variant] == "ok":
                assert result["detection"]["n_hands"] == 1
                assert len(result["detection"]["image_landmarks"]) == 21
    counts = {v: sum(r[v] == "ok" for r in paired) for v in variants}
    counts["any_variant_unique"] = sum(bool(r["recovered_variants"]) for r in paired)
    assert {r["variant"] for r in summary} == set(counts)
    for row in summary:
        assert int(row["original_misses"]) == len(missed)
        assert int(row["recovered"]) == counts[row["variant"]]
        percent = round(100 * counts[row["variant"]] / len(missed), 3) if missed else 0
        assert float(row["recovery_percent"]) == percent
    unrecovered = read_csv(run, "unrecovered.csv")
    assert {r["id"] for r in unrecovered} == {r["id"] for r in paired if not r["recovered_variants"]}
    return dict(verified=True, images=len(baseline), original_misses=len(missed),
                originally_accepted=sum(r["status"] == "ok" for r in baseline),
                recovered=counts, recovered_files={r["filename"]: r["recovered_variants"]
                                                 for r in paired if r["recovered_variants"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
