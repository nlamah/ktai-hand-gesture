"""Read-only evidence audit of saved features, splits, pipelines and reported scores."""
import argparse

from gesturelab.validation.runner import run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--project', help='Moved dataset override; same manifest contents are required.')
    parser.add_argument('--paper', help='Local full Shin 2021 PDF to archive as audit evidence.')
    parser.add_argument('--require-shared-splits', action='store_true')
    parser.add_argument('--tolerance', type=float, default=1e-9)
    args = vars(parser.parse_args())
    try:
        print(run_audit(**args) / 'AUDIT.md')
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
