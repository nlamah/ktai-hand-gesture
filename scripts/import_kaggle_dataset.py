"""Import all A–Z photos from the extracted Kaggle ASL Alphabet dataset."""
import argparse

from gesturelab.datasets.kaggle import import_kaggle_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/external/archive-kaggle')
    parser.add_argument('--project', default='data/asl_kaggle')
    parser.add_argument('--block-size', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exclude-overlapping-test', action='store_true', help='Exclude test copies of training pixels; preserve training photos.')
    args = parser.parse_args()
    try:
        print(import_kaggle_dataset(args.source, args.project, args.block_size,
              args.checkpoint_every, args.resume, lambda n, total: print(f'Imported {n}/{total}', flush=True),
              exclude_overlapping_test=args.exclude_overlapping_test))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
