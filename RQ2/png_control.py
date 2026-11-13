"""Paired original/unchanged-PNG detector control, separate from historical runs."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import mediapipe
import PIL

from research.data import Detector, decode_image, model_path
from experiments.hand_detection_recovery.path_utils import resolve_path
from experiments.hand_detection_recovery.verify_results import read_csv, verify
from RQ2.paper_analysis import digest, output_path, write_table

ROOT = Path(__file__).resolve().parents[1]


def environment(settings):
    current = dict(mediapipe=mediapipe.__version__, opencv=cv2.__version__, detector_sha256=digest(model_path()))
    for key, value in current.items():
        if settings[key] != value:
            raise ValueError(f'Historical {key} differs from this environment; use matching versions/model.')
    if settings['mode'] != 'IMAGE' or settings['num_hands'] != 2:
        raise ValueError('Unsupported historical detector configuration.')
    files = ['research/data.py', 'research/landmarks.py', 'research/finger_pilot.py', 'RQ2/png_control.py']
    return dict(**current, python=sys.version, pillow=PIL.__version__,
                code_sha256={name: digest(ROOT / name) for name in files},
                historical_code_matches={name: digest(ROOT / 'research' / name) == value
                                         for name, value in settings['code_hashes'].items()})


def check_image(row, output, detector):
    source = resolve_path(row['original_path'])
    if digest(source) != row['source_sha256']:
        raise ValueError(f'Source image changed: {row["filename"]}')
    original = decode_image(source.read_bytes())
    png = output / 'images' / (row['id'] + '.png')
    original.save(png, format='PNG')
    reloaded = decode_image(png.read_bytes())
    pixels_equal = original.size == reloaded.size and original.tobytes() == reloaded.tobytes()
    if not pixels_equal:
        raise ValueError('PNG round-trip changed decoded pixels.')
    direct, retry = detector.analyze(original), detector.analyze(reloaded)
    return dict(id=row['id'], filename=row['filename'], letter=row['letter'],
                historical_status='no_hand', original_status=direct['status'], png_status=retry['status'],
                pixels_equal=True, source_sha256=row['source_sha256'], png_sha256=digest(png),
                original_detection=direct, png_detection=retry)


def control(run, output, limit=None, resume=False):
    run = Path(run).resolve()
    output = output_path(run, output)
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive.')
    if output.exists() and not resume:
        raise FileExistsError('Output already exists; choose a new directory or explicitly use --resume.')
    if resume and not (output / 'progress.sqlite').is_file():
        raise ValueError('No control checkpoint to resume.')
    verify(run)
    settings = json.loads((run / 'settings.json').read_text(encoding='utf-8'))
    selected = sorted(read_csv(run, 'no_hand.csv'), key=lambda r: (r['letter'], r['filename'], r['id']))
    selected = selected[:limit] if limit is not None else selected
    config = dict(source_exports={n: digest(run / n) for n in ['settings.json', 'baseline.csv', 'no_hand.csv']},
                  limit=limit, selected_images=len(selected), environment=environment(settings),
                  detection_confidence=settings['detection_confidence'], presence_confidence=settings['presence_confidence'])
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive lock; never remove another process's lock. After a forced kill,
    # inspect the process before manually removing a stale control.lock.
    lock = output / 'control.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write('PNG control in progress\n')
    try:
        with closing(sqlite3.connect(output / 'progress.sqlite')) as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, value TEXT)')
            existing = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if existing and json.loads(existing[0]) != config:
                raise ValueError('Source, limit, environment or code changed; use a new control directory.')
            if not existing:
                db.execute("INSERT INTO metadata VALUES ('config', ?)", (json.dumps(config),))
                db.commit()
                (output / 'settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            done = {key: json.loads(value) for key, value in db.execute('SELECT id,value FROM records')}
            (output / 'images').mkdir(exist_ok=True)
            detector = None
            try:
                for index, row in enumerate(selected, 1):
                    if digest(resolve_path(row['original_path'])) != row['source_sha256']:
                        raise ValueError(f'Source image changed: {row["filename"]}')
                    if row['id'] in done:
                        if digest(output / 'images' / (row['id'] + '.png')) != done[row['id']]['png_sha256']:
                            raise ValueError('Saved control PNG changed.')
                        continue
                    if detector is None:
                        detector = Detector(min_hand_detection_confidence=settings['detection_confidence'],
                                            min_hand_presence_confidence=settings['presence_confidence'])
                    result = check_image(row, output, detector)
                    with db:
                        db.execute('INSERT INTO records VALUES (?,?)', (row['id'], json.dumps(result)))
                    done[row['id']] = result
                    if index % 100 == 0:
                        print(f'PNG control {index}/{len(selected)}', flush=True)
            finally:
                if detector is not None:
                    detector.close()
            results = [done[r['id']] for r in selected]
        fields = ['id', 'filename', 'letter', 'historical_status', 'original_status', 'png_status', 'pixels_equal', 'source_sha256', 'png_sha256']
        write_table(output / 'per_image.csv', [{k: r[k] for k in fields} for r in results], fields)
        with (output / 'detections.jsonl').open('w', encoding='utf-8') as handle:
            for row in results:
                handle.write(json.dumps(row) + '\n')
        summary = dict(images=len(results), complete=True, limited=limit is not None,
                       original_statuses=dict(Counter(r['original_status'] for r in results)),
                       png_statuses=dict(Counter(r['png_status'] for r in results)),
                       png_recoveries_vs_historical=sum(r['png_status']=='ok' for r in results),
                       original_recoveries_vs_historical=sum(r['original_status']=='ok' for r in results),
                       paired_status_changes=sum(r['png_status']!=r['original_status'] for r in results),
                       paired_png_recoveries=sum(r['original_status']=='no_hand' and r['png_status']=='ok' for r in results),
                       all_pixels_equal=all(r['pixels_equal'] for r in results),
                       note='Zero recovery is not assumed. Original replays distinguish historical drift from paired PNG differences; sequential calls do not isolate nondeterminism.')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(control(args.run, args.output, args.limit, args.resume), indent=2))


if __name__ == '__main__':
    main()
