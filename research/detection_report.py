"""Summarize landmark acceptance and export an inventory of rejected images."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from research.data import read_manifest


def report(project, out):
    project, out = Path(project), Path(out)
    samples = [s for s in read_manifest(project)['samples']
               if s.get('included') and s.get('role') == 'development']
    if not samples:
        raise ValueError('No included development images found in this project.')
    out.mkdir(parents=True, exist_ok=True)
    statuses = sorted({s['status'] for s in samples} | {'ok', 'no_hand', 'multiple_hands', 'invalid_geometry', 'pending'})
    counts = Counter(s['status'] for s in samples)
    print(f'Total development images: {len(samples)}')
    for status in statuses:
        print(f'{status}: {counts[status]} ({100 * counts[status] / len(samples):.3f}%)')
    with (out / 'by_letter.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['letter', 'total', *statuses, 'accepted_percent'])
        for letter in sorted({s['label'] for s in samples}):
            group = [s for s in samples if s['label'] == letter]
            counts = Counter(s['status'] for s in group)
            writer.writerow([letter, len(group), *(counts[k] for k in statuses), 100 * counts['ok'] / len(group)])
    with (out / 'rejected_images.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'letter', 'status', 'filename', 'image_path', 'original_path'])
        for sample in samples:
            if sample['status'] not in ('ok', 'pending'):
                writer.writerow([sample['id'], sample['label'], sample['status'], sample['filename'],
                                 str((project / sample['image_path']).resolve()),
                                 str((project / sample['original_path']).resolve())])
    print(f'Reports saved to {out.resolve()} (pending images are not counted as rejected).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report(args.project, args.out)


if __name__ == '__main__':
    main()
