"""Calibrate categorical feature thresholds using blind human development labels."""
import argparse

from gesturelab.features.asl_calibration import calibrate_asl_thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--role-map", required=True, help="Grouped roles from scripts.prepare_asl_roles.")
    parser.add_argument("--candidates", required=True, help="JSON file containing explicit parameter candidates.")
    parser.add_argument("--out", required=True, help="New output directory; existing results are never overwritten.")
    args = parser.parse_args()
    try:
        print(calibrate_asl_thresholds(args.project, args.role_map, args.candidates, args.out))
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
