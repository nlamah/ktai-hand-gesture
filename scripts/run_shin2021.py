"""Compare project, Shin 2021 and combined inputs on the same grouped data."""
import argparse

from gesturelab.training.classifiers import PROFILES
from gesturelab.training.paper_study import run_paper_study
from gesturelab.features.training_inputs import TRAINING_INPUTS, PAPER_COMPARISON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--out', default='experiments/shin2021_01')
    parser.add_argument('--models', nargs='+', choices=('svm', 'lightgbm'), default=['svm', 'lightgbm'])
    parser.add_argument('--variants', nargs='+', choices=tuple(TRAINING_INPUTS), default=PAPER_COMPARISON)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--grouping', choices=('session', 'participant', 'source'), default='session')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classifier-profile', choices=PROFILES, default='project')
    parser.add_argument('--split-from', help='Reuse verified photo/fold assignments from an existing experiment.')
    parser.add_argument('--allow-no-test', action='store_true')
    parser.add_argument('--allow-incomplete-test', action='store_true')
    args = vars(parser.parse_args())
    try:
        print(run_paper_study(**args) / 'report.html')
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
