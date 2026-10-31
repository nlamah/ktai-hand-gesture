"""Measure grouped one-hot permutation importance on saved validation rows."""
import argparse

from gesturelab.training.asl_feature_analysis import permutation_importance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed ASL categorical SVM or MLP run.")
    parser.add_argument("--out", required=True, help="New output directory.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(permutation_importance(args.run, args.out, args.repeats, args.seed))
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
