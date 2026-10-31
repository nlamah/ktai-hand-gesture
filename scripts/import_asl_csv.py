"""Import a second, explicitly documented ASL dataset from a local CSV."""
import argparse

from gesturelab.datasets.asl_csv import CSV_COLUMNS, import_asl_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help=f"CSV with columns: {', '.join(CSV_COLUMNS)}")
    parser.add_argument("--images-root", required=True, help="Root directory containing the CSV image paths.")
    parser.add_argument("--source-name", required=True, help="Name of the actual dataset and its provenance.")
    parser.add_argument("--project", required=True, help="New, separate Gesture Lab output directory.")
    args = parser.parse_args()
    try:
        result = import_asl_csv(args.csv, args.images_root, args.project, args.source_name)
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Dataset ready: {result['added']} images added; {result['duplicates']} duplicates skipped.")
    print("Images remain pending. Analyze them separately before creating an experiment.")


if __name__ == "__main__":
    main()
