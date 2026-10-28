"""Compare two saved classifier runs with identical inputs and grouped OOF assignments."""
import argparse
from gesturelab.training.classifier_comparison import compare_classifiers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Existing baseline experiment directory containing split.json.')
    parser.add_argument('--candidate', required=True, help='New classifier experiment directory containing split.json.')
    parser.add_argument('--variant', default='shin2021_combined')
    parser.add_argument('--out', required=True)
    try:
        print(compare_classifiers(**vars(parser.parse_args())))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Comparison stopped: {error}\n')


if __name__ == '__main__':
    main()
