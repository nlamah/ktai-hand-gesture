"""Compare saved ASL model predictions on their shared validation split."""
import argparse

from gesturelab.training.asl_comparison import compare_asl_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="NAME=PATH",
        help="Completed ASL run; repeat at least twice with distinct names.",
    )
    parser.add_argument("--out", required=True, help="New comparison output directory.")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not name.strip() or not path.strip() or name in runs:
            parser.error(f"Invalid or duplicate --run value: {item!r}. Expected NAME=PATH.")
        runs[name] = path
    try:
        print(compare_asl_runs(runs, args.out))
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
