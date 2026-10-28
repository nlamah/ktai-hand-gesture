"""Explicitly retrain one existing variant in a separate audit output directory."""
import argparse

from gesturelab.validation.reproduction import reproduce_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--project')
    args = vars(parser.parse_args())
    try:
        print(reproduce_variant(**args) / 'AUDIT.md')
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
