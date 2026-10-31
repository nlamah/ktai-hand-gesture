"""Freeze grouped calibration and validation roles for ASL feature annotation."""
import argparse

from gesturelab.features.asl_roles import create_asl_role_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grouping", choices=("session", "participant", "source"), default="source")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(create_asl_role_map(**vars(args)))
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
