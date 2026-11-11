"""Three resumable stages for paired hand-detection recovery experiments."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import cv2
import mediapipe
from PIL import Image

from research.data import Detector, decode_image, model_path, project_lock
from research.kaggle import LETTERS, _split_folder, _training_entries
from research.morphology_pilot import variants
from experiments.hand_detection_recovery.path_utils import resolve_path, stored_path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('erosie', 'dilatatie', 'canny_overlay')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(db, stage, key, value):
    with db:
        db.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?)',
                   (stage, key, json.dumps(value, allow_nan=False)))


def records(db, stage):
    return {key: json.loads(value) for key, value in db.execute(
        'SELECT key, value FROM records WHERE stage=? ORDER BY key', (stage,))}


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configuration(args):
    source = _split_folder(args.source.resolve(), 'asl_alphabet_train')
    entries = list(_training_entries(source, 500))
    if not entries:
        raise ValueError('No A-Z training images found.')
    selected = entries[:args.limit] if args.limit else entries
    inventory = [(stored_path(e['source']), e['source'].stat().st_size,
                  e['source'].stat().st_mtime_ns) for e in selected]
    return dict(source=stored_path(args.source), limit=args.limit,
                path_base='repository',
                inventory_sha256=sha256(json.dumps(inventory).encode()).hexdigest(),
                images=len(selected), mediapipe=mediapipe.__version__, opencv=cv2.__version__,
                detector_sha256=digest(model_path()), detection_confidence=0.5,
                presence_confidence=0.5, num_hands=2, mode='IMAGE',
                kernel=[3, 3], iterations=1, canny_thresholds=[50, 150],
                blur=[5, 5], overlay_weights=[1.0, 0.25],
                code_hashes={name: digest(ROOT / 'research' / name) for name in
                             ('data.py', 'morphology_pilot.py', 'recovery_experiment.py')},
                path_utils_sha256=digest(ROOT / 'experiments/hand_detection_recovery/path_utils.py')), selected


def baseline(db, run, entries):
    completed = records(db, 'baseline')
    detector = Detector()
    try:
        for i, entry in enumerate(entries, 1):
            path = entry['source']
            key = stored_path(path)
            if key not in completed:
                image = decode_image(path.read_bytes())
                analysis = detector.analyze(image)
                row = dict(id=sha256(str(image.size).encode() + image.tobytes()).hexdigest()[:24],
                           filename=path.name, letter=entry['label'], original_path=key,
                           source_sha256=digest(path), status=analysis['status'],
                           n_hands=analysis['n_hands'])
                save(db, 'baseline', key, row)
            if i % 250 == 0:
                print(f'Baseline {i}/{len(entries)}', flush=True)
    finally:
        detector.close()
    rows = list(records(db, 'baseline').values())
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source pixels found. Resolve duplicates before continuing.')
    fields = list(rows[0])
    write_csv(run / 'baseline.csv', rows, fields)
    write_csv(run / 'no_hand.csv', [r for r in rows if r['status'] == 'no_hand'], fields)
    save(db, 'state', 'baseline_complete', True)


def preprocess(db, run):
    completed = records(db, 'preprocess')
    missed = [r for r in records(db, 'baseline').values() if r['status'] == 'no_hand']
    for i, row in enumerate(missed, 1):
        source = resolve_path(row['original_path'])
        if digest(source) != row['source_sha256']:
            raise ValueError(f'Source image changed: {source}')
        for name, pixels in variants(decode_image(source.read_bytes())).items():
            key = row['id'] + ':' + name
            if key in completed:
                if digest(resolve_path(completed[key]['processed_path'])) != completed[key]['processed_sha256']:
                    raise ValueError(f'Processed image changed: {key}')
                continue
            path = run / 'processed' / name / row['letter'] / (row['id'] + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            Image.fromarray(pixels).save(temporary, format='PNG')
            temporary.replace(path)
            save(db, 'preprocess', key, dict(row, variant=name, processed_path=stored_path(path),
                                             processed_sha256=digest(path)))
        if i % 250 == 0:
            print(f'Preprocess {i}/{len(missed)}', flush=True)
    rows = list(records(db, 'preprocess').values())
    write_csv(run / 'processed.csv', rows,
              ['id', 'filename', 'letter', 'variant', 'original_path', 'processed_path', 'processed_sha256'])
    save(db, 'state', 'preprocess_complete', True)


def export_results(db, run):
    missed = [r for r in records(db, 'baseline').values() if r['status'] == 'no_hand']
    results = records(db, 'retry')
    paired = []
    for row in missed:
        statuses = {name: results[row['id'] + ':' + name]['retry_status'] for name in VARIANTS}
        paired.append(dict(row, **statuses, recovered_variants=';'.join(
            name for name in VARIANTS if statuses[name] == 'ok')))
    fields = ['id', 'filename', 'letter', 'original_path', 'status', *VARIANTS, 'recovered_variants']
    write_csv(run / 'per_image.csv', paired, fields)
    write_csv(run / 'unrecovered.csv', [r for r in paired if not r['recovered_variants']], fields)
    # recovered.csv and still_no_hand.csv are written once, by analyze.py.
    summary = []
    for name in (*VARIANTS, 'any_variant_unique'):
        count = sum(bool(r['recovered_variants']) if name == 'any_variant_unique'
                    else r[name] == 'ok' for r in paired)
        summary.append(dict(variant=name, original_misses=len(missed), recovered=count,
                            recovery_percent=round(100 * count / len(missed), 3) if missed else 0))
    write_csv(run / 'summary.csv', summary,
              ['variant', 'original_misses', 'recovered', 'recovery_percent'])
    with (run / 'retry_landmarks.jsonl').open('w', encoding='utf-8') as handle:
        for row in results.values():
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def retry(db, run):
    completed = records(db, 'retry')
    processed = records(db, 'preprocess')
    detector = Detector()
    try:
        for i, (key, row) in enumerate(processed.items(), 1):
            path = resolve_path(row['processed_path'])
            if digest(path) != row['processed_sha256']:
                raise ValueError(f'Processed image changed: {path}')
            if key not in completed:
                analysis = detector.analyze(decode_image(path.read_bytes()))
                analysis.pop('features', None)
                status = analysis.pop('status')
                save(db, 'retry', key, dict(row, retry_status=status, detection=analysis))
            if i % 250 == 0:
                print(f'Retry {i}/{len(processed)}', flush=True)
    finally:
        detector.close()
    export_results(db, run)
    save(db, 'state', 'retry_complete', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baseline', 'preprocess', 'retry'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/asl-alphabet-download')
    parser.add_argument('--limit', type=int, help='Small smoke test only; use the same limit in all stages.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    run = args.run.resolve()
    with project_lock(run):
        db = sqlite3.connect(run / 'progress.sqlite')
        try:
            db.execute('CREATE TABLE IF NOT EXISTS records (stage TEXT, key TEXT, value TEXT, PRIMARY KEY(stage,key))')
            config, entries = configuration(args)
            state = records(db, 'state')
            if 'config' not in state:
                if args.stage != 'baseline':
                    raise ValueError('Run baseline first.')
                save(db, 'state', 'config', config)
                (run / 'settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            elif state['config'] != config:
                raise ValueError('Source, settings or code changed. Use a new --run directory.')
            required = {'preprocess': 'baseline_complete', 'retry': 'preprocess_complete'}
            if args.stage in required and not state.get(required[args.stage]):
                raise ValueError(f'Complete {required[args.stage]} first.')
            if args.stage == 'baseline':
                baseline(db, run, entries)
            elif args.stage == 'preprocess':
                preprocess(db, run)
            else:
                retry(db, run)
            print(f'Completed {args.stage}. Output: {run}', flush=True)
        finally:
            db.close()


if __name__ == '__main__':
    main()
