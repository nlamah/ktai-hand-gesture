"""Fit one fixed ASL categorical SVM/MLP or evaluate verified reference rules."""
import argparse

from gesturelab.features.asl_models import INPUT_VARIANTS
from gesturelab.training.asl_categorical import MODEL_ROUTES, train_asl_categorical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--role-map", required=True)
    parser.add_argument("--model", choices=MODEL_ROUTES, required=True)
    parser.add_argument("--out", required=True, help="New output directory; existing results are preserved.")
    parser.add_argument("--thresholds", help="Selected categorical_thresholds.json; omit to use schema defaults.")
    parser.add_argument("--input", choices=INPUT_VARIANTS, default="categorical")
    parser.add_argument("--reference-matrix", help="Required for --model rules.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(train_asl_categorical(
            args.project, args.role_map, args.out, args.model,
            args.thresholds, args.reference_matrix, args.seed, args.input,
        ))
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
