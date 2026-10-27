"""Convert the downloaded ASL COCO export to a Gesture Lab dataset."""
import argparse
from gesturelab.datasets.coco import import_coco_dataset


def _progress(done, total):
    if done % 100 == 0 or done == total:
        print(f'Imported {done}/{total} images', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/external/asl_letters_v1')
    parser.add_argument('--project', default='data/asl')
    args = parser.parse_args()
    try:
        result = import_coco_dataset(args.source, args.project, _progress)
    except (ValueError, FileExistsError, OSError) as error:
        parser.exit(1, str(error) + '\n')
    print(f"Dataset ready: {result['added']} images added; {result['duplicates']} exact duplicates skipped.")
    print('Next: analyze landmarks, then prepare a new experiment with --grouping source.')


if __name__ == '__main__':
    main()
