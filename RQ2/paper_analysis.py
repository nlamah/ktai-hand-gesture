"""Recompute RQ2 paper counts from verified exports without changing the run."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from experiments.hand_detection_recovery.verify_results import verify, read_csv

VARIANTS = ('erosie', 'dilatatie', 'canny_overlay')


def digest(path):
    with Path(path).open('rb') as handle:
        return sha256(handle.read()).hexdigest()


def output_path(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Choose a separate output directory outside the source run.')
    return output


def write_table(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(baseline, paired):
    """Separate image counts, overlapping rejection statuses and retry counts."""
    baseline_ids = {r['id'] for r in baseline}
    misses = {r['id'] for r in baseline if r['status'] == 'no_hand'}
    if len(baseline_ids) != len(baseline) or len({r['id'] for r in paired}) != len(paired):
        raise ValueError('Duplicate image IDs.')
    if {r['id'] for r in paired} != misses:
        raise ValueError('Paired rows must cover exactly the baseline no_hand images.')
    allowed = {'ok', 'no_hand', 'multiple_hands', 'invalid_geometry'}
    for row in paired:
        if any(row[v] not in allowed for v in VARIANTS):
            raise ValueError('Unknown retry status.')
        if row['recovered_variants'] != ';'.join(v for v in VARIANTS if row[v] == 'ok'):
            raise ValueError('Inconsistent recovered_variants.')
    recovered = [r for r in paired if any(r[v] == 'ok' for v in VARIANTS)]
    rejected = [r for r in paired if not any(r[v] == 'ok' for v in VARIANTS)]
    other = [r for r in rejected if any(r[v] != 'no_hand' for v in VARIANTS)]
    status_images = {s: sum(any(r[v] == s for v in VARIANTS) for r in other)
                     for s in ('multiple_hands', 'invalid_geometry')}
    combinations = Counter(tuple(r[v] for v in VARIANTS) for r in other)
    by_letter = []
    for letter in sorted({r['letter'] for r in baseline}):
        total = sum(r['letter'] == letter for r in baseline)
        group = [r for r in paired if r['letter'] == letter]
        unique = sum(any(r[v] == 'ok' for v in VARIANTS) for r in group)
        by_letter.append(dict(letter=letter, images=total, original_misses=len(group),
                              miss_percent=100 * len(group) / total,
                              unique_recovered=unique,
                              recovery_percent=100 * unique / len(group) if group else None,
                              **{v: sum(r[v] == 'ok' for r in group) for v in VARIANTS}))
    summary = dict(images=len(baseline), original_misses=len(paired), unique_recovered=len(recovered),
                   additional_without_dilation=sum(r['dilatatie'] != 'ok' for r in recovered),
                   unrecovered=len(rejected), all_variants_no_hand=len(rejected)-len(other),
                   other_rejected_images=len(other), other_rejected_images_by_status=status_images,
                   other_rejected_retries_by_status=dict(Counter(r[v] for r in other for v in VARIANTS)),
                   note='Image status groups may overlap; retry counts count three attempts per image.')
    return summary, by_letter, [dict(zip(VARIANTS, key), images=count)
                               for key, count in sorted(combinations.items())], other


def analyze(run, output):
    run = Path(run).resolve()
    output = output_path(run, output)
    if output.exists():
        raise FileExistsError('Choose a new output directory; previous analyses are never overwritten.')
    verify(run)
    baseline, paired = read_csv(run, 'baseline.csv'), read_csv(run, 'per_image.csv')
    summary, letters, combinations, other = summarize(baseline, paired)
    summary['input_sha256'] = {p.name: digest(p) for p in sorted(run.iterdir())
                               if p.suffix in ('.csv', '.json', '.jsonl')}
    summary['analysis_sha256'] = digest(__file__)
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_table(output / 'by_letter.csv', letters, list(letters[0]))
    write_table(output / 'rejection_combinations.csv', combinations, [*VARIANTS, 'images'])
    write_table(output / 'other_rejected_images.csv', other, list(paired[0]) if paired else
                ['id', 'filename', 'letter', 'original_path', 'status', *VARIANTS, 'recovered_variants'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))


if __name__ == '__main__':
    main()
