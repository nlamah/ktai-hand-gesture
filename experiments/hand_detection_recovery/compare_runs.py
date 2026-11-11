"""Compare two completed runs by image ID, variant and stored landmark coordinates."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from verify_results import read_csv, verify
from path_utils import stored_path


def indexed_csv(run, filename, keys):
    rows = read_csv(run, filename)
    indexed = {tuple(row[k] for k in keys): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate keys in {run / filename}')
    return indexed


def landmarks(run):
    with (run / 'retry_landmarks.jsonl').open(encoding='utf-8') as handle:
        return {(r['id'], r['variant']): r for r in map(json.loads, handle)}


def compare(reference, repeat):
    verify(reference)
    verify(repeat)
    differences = []
    report = dict(reference=stored_path(reference), repeat=stored_path(repeat))
    configs = [json.loads((run / 'settings.json').read_text()) for run in (reference, repeat)]
    report['settings_equal'] = configs[0] == configs[1]
    report['changed_settings'] = [key for key in configs[0].keys() | configs[1].keys()
                                  if configs[0].get(key) != configs[1].get(key)]
    for name, file, keys, fields in [
        ('baseline', 'baseline.csv', ['id'], ['filename', 'letter', 'source_sha256', 'status', 'n_hands']),
        ('processed', 'processed.csv', ['id', 'variant'], ['processed_sha256']),
        ('retry', 'per_image.csv', ['id'], ['status', 'erosie', 'dilatatie', 'canny_overlay', 'recovered_variants']),
        ('summary', 'summary.csv', ['variant'], ['original_misses', 'recovered', 'recovery_percent']),
    ]:
        left, right = [indexed_csv(run, file, keys) for run in (reference, repeat)]
        mismatches = 0
        for key in sorted(left.keys() | right.keys()):
            changed = [f for f in fields if left.get(key, {}).get(f) != right.get(key, {}).get(f)]
            if key not in left or key not in right:
                changed = ['missing_record']
            if changed:
                mismatches += 1
                differences.append(dict(stage=name, key='|'.join(key), fields=';'.join(changed)))
        report[name] = dict(reference_records=len(left), repeat_records=len(right), different_records=mismatches)
    left, right = landmarks(reference), landmarks(repeat)
    coordinate_changes = 0
    coordinate_records = 0
    metadata_changes = 0
    maximum = 0.0
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key, {}).get('detection', {}), right.get(key, {}).get('detection', {})
        has_coordinates = 'image_landmarks' in a or 'image_landmarks' in b
        coordinate_records += has_coordinates
        changed = []
        for field in ('image_landmarks', 'world_landmarks'):
            if a.get(field) != b.get(field):
                changed.append(field)
            av = [v for point in a.get(field, []) for v in point]
            bv = [v for point in b.get(field, []) for v in point]
            if len(av) == len(bv) and av:
                maximum = max(maximum, max(abs(x-y) for x, y in zip(av, bv)))
        coordinate_changes += bool(changed)
        am = {k: v for k, v in a.items() if k not in ('image_landmarks', 'world_landmarks')}
        bm = {k: v for k, v in b.items() if k not in ('image_landmarks', 'world_landmarks')}
        metadata_changes += am != bm
        if changed or am != bm:
            differences.append(dict(stage='landmarks', key='|'.join(key), fields=';'.join(changed + (['metadata'] if am != bm else []))))
    report['landmarks'] = dict(reference_records=len(left), repeat_records=len(right),
        records_with_coordinates=coordinate_records, different_coordinate_records=coordinate_changes,
        different_metadata_records=metadata_changes, maximum_absolute_coordinate_difference=maximum)
    report['all_compared_outputs_equal'] = not differences and report['settings_equal']
    files = ['settings.json', 'baseline.csv', 'processed.csv', 'per_image.csv', 'summary.csv', 'retry_landmarks.jsonl']
    report['evidence_sha256'] = {label: {name: sha256((run/name).read_bytes()).hexdigest() for name in files}
                                 for label, run in [('reference', reference), ('repeat', repeat)]}
    report['scope'] = 'Two executions on the same workstation. Baseline coordinates were not exported; coordinate comparison covers retry detections. Absolute output paths are excluded from content comparisons.'
    (repeat / 'repeatability.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (repeat / 'repeatability_differences.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stage', 'key', 'fields'])
        writer.writeheader()
        writer.writerows(differences)
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--repeat', required=True, type=Path)
    args = parser.parse_args()
    compare(args.reference.resolve(), args.repeat.resolve())
