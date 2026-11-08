
import json
import subprocess
import sys
from pathlib import Path

DEVELOPMENT = Path("data/youri_development")

ORIGINAL_SELECTION = Path(
    "experiments/youri_targeted_01/selection.json"
)

DATASET_OUT = Path("data/youri_n_only_01")
EXPERIMENT_OUT = Path("experiments/youri_n_only_01")


def main():
    if DATASET_OUT.exists() or EXPERIMENT_OUT.exists():
        raise FileExistsError(
            "N-only experiment already exists. "
            "Existing results will not be overwritten."
        )

    with open(DEVELOPMENT / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(ORIGINAL_SELECTION, encoding="utf-8") as f:
        previous = json.load(f)

    # Use only N images that were already included
    # in the original targeted experiment.
    previous_ids = set(previous["added_ids"])

    n_ids = sorted(
        sample["id"]
        for sample in manifest["samples"]
        if sample["id"] in previous_ids
        and sample["label"] == "N"
        and sample["role"] == "development"
        and sample["status"] == "ok"
        and sample.get("included", True)
    )

    print(f"Selected N images: {len(n_ids)}")

    if len(n_ids) != 15:
        raise ValueError(
            f"Expected 15 N images, found {len(n_ids)}"
        )

    command = [
        sys.executable, "-m", "scripts.research", "train",
        "--baseline", "exports/youri_baseline.json",
        "--additions", str(DEVELOPMENT),
        "--ids", *n_ids,
        "--match-count", "15",
        "--dataset-out", str(DATASET_OUT),
        "--experiment-out", str(EXPERIMENT_OUT),
        "--hypothesis",
        "Additional N examples reduce N to M confusion",
        "--change",
        "Original Kaggle training data plus 15 N images",
        "--expected",
        "Lower N to M errors without increasing M to N errors",
    ]

    subprocess.run(command, check=True)

    print("N-only training completed.")


if __name__ == "__main__":
    main()
