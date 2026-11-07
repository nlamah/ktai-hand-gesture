"""Replay the fixed 7-Nov G/H captures and illustrate remaining errors, without fitting."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from research.data import atomic_json
from research.feature_review import hand_crop
from research.inputs import write_csv
from research.live_model import DEFAULT_MODEL, load_bundle, predict_letter
from research.mirroring import pixel_hash

OUT = Path('results/gh_examples_20261107')
FIGURE = Path('paper/figures/gh_recent_webcam_examples.pdf')
SESSIONS = (
    Path('captures/gh_viewpoint/GH_views_20261107T150906Z_7cacb76c'),
    Path('captures/gh_viewpoint/GH_views_20261107T151003Z_3a6394bb'),
)
MODELS = {'reference': DEFAULT_MODEL, 'gh_y180': Path('models/asl_shin_angles_GH_y180.joblib')}
VIEWS = {'schuin_handrug': 'schuin naar de handrug', 'handrug': 'handrug naar camera'}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (OUT / 'withdrawn.json').exists():
        raise SystemExit('Deze G/H-opnamen zijn door de gebruiker ingetrokken wegens een onjuiste uitvoering. Gebruik een nieuwe opnameronde; de oude voorbeelden worden niet opnieuw geëxporteerd.')
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    bundles = {name: load_bundle(path) for name, path in MODELS.items()}
    model_hashes = {name: digest(path) for name, path in MODELS.items()}
    rows, captures, plans = [], {}, []
    for directory in SESSIONS:
        plan_path = directory / 'session.json'
        plan = json.loads(plan_path.read_text())
        tasks = {task['id']: task for task in plan['tasks']}
        for name, path in MODELS.items():
            assert plan['frozen_models'][str(path.resolve())] == model_hashes[name]
        count = 0
        for path in sorted(directory.glob('*/capture.json')):
            record = json.loads(path.read_text())
            task = record['protocol']['task']
            assert task == tasks[task['id']]
            assert record['expected_letter'] == task['letter']
            assert record['model']['sha256'] == model_hashes['reference']
            for filename, info in record['files'].items():
                assert digest(path.parent / filename) == info['sha256']
            detection = record['detection']
            with Image.open(path.parent / 'model_input.png') as image:
                size = list(image.size)
            assert detection['image_size'] == size
            predicted = {name: predict_letter(bundle, detection) for name, bundle in bundles.items()}
            letters = {name: value['letter'] if value else '?' for name, value in predicted.items()}
            assert letters['reference'] == (record['predicted_letter'] or '?')
            row = {'id': record['id'], 'session': directory.name, 'task': task['id'],
                   'participant': task['participant'], 'split': task['split'],
                   'label': task['letter'], 'requested_view': task['view'],
                   'requested_hand': task['requested_hand'], 'repeat': task['repeat'],
                   'saved_at_utc': record['saved_at_utc'], 'status': detection['status'], **letters,
                   'capture_json': str(path), 'capture_sha256': digest(path),
                   'model_input_sha256': digest(path.parent / 'model_input.png')}
            assert row['id'] not in captures
            captures[row['id']] = (path.parent, record)
            rows.append(row)
            count += 1
        plans.append({'path': str(plan_path), 'sha256': digest(plan_path),
                      'planned': len(tasks), 'saved': count,
                      'instruction_version': plan.get('viewpoint_instruction_version')})
    rows.sort(key=lambda row: (row['saved_at_utc'], row['id']))
    assert rows and all(row['participant'] == 'P01' and row['split'] == 'development' for row in rows)
    write_csv(OUT / 'predictions.csv', rows)

    # Fixed session and first attempt in each requested view/letter cell. Do not
    # select the worst-looking prediction or silently mix restarted participants.
    selected = []
    for view in ('schuin_handrug', 'handrug'):
        for letter in ('G', 'H'):
            eligible = [r for r in rows if r['session'] == SESSIONS[1].name
                        and r['requested_view'] == view and r['label'] == letter]
            row = min(eligible, key=lambda r: (r['repeat'], r['saved_at_utc'], r['id']))
            assert row['status'] == 'ok'
            selected.append(row)

    original = json.loads(Path('results/appendix_visuals_01/provenance.json').read_text())
    dataset = [next(r for r in original['sources']
                    if r['kind'] == 'dataset_illustration' and r['label'] == letter)
               for letter in ('G', 'H')]
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), layout='constrained')
    for axis, item in zip(axes[0], dataset):
        with Image.open(item['path']) as image:
            source = image.convert('RGB')
        assert pixel_hash(source) == item['pixel_sha256']
        axis.imshow(source)
        axis.axis('off')
        axis.set_title(f"Datasetlabel {item['label']} · {item['filename']}\nOorspronkelijk trainingsaanzicht", fontsize=10)
    selected_provenance = []
    for axis, row in zip(axes[1:].flat, selected):
        folder, record = captures[row['id']]
        with Image.open(folder / 'model_input.png') as image:
            crop, box = hand_crop(image, record['detection']['image_landmarks'])
        assert row['reference'] != row['label']
        outcome = 'hersteld' if row['gh_y180'] == row['label'] else 'blijft fout'
        axis.imshow(crop)
        axis.axis('off')
        axis.set_title(f"Beoogd {row['label']} · {VIEWS[row['requested_view']]}\n"
                       f"Referentie [2]: {row['reference']} | Rotatie [9]: {row['gh_y180']} · {outcome}", fontsize=10)
        selected_provenance.append({**row, 'crop_box': box, 'outcome': outcome})
    fig.savefig(FIGURE, bbox_inches='tight')
    fig.savefig(OUT / 'gh_recent_webcam_examples.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    atomic_json(OUT / 'provenance.json', {
        'sessions': plans, 'models': {str(path): model_hashes[name] for name, path in MODELS.items()},
        'selection_policy': 'First saved attempt per G/H and oblique-back/back view in the second fixed session; no score-based ranking.',
        'selected': selected_provenance, 'dataset_illustrations': dataset,
        'figure_sha256': digest(FIGURE),
        'source_sha256': {path: digest(path) for path in
                          ('research/studies/recent_gh_examples.py', 'research/feature_review.py', 'research/live_model.py')},
        'scope': 'Selected diagnostic examples from incomplete P01 development sessions; not independent evaluation, ASL label validation or an estimate of population accuracy.',
    })
    atomic_json(OUT / 'verification.json', {
        'status': 'passed', 'capture_records_replayed': len(rows),
        'capture_and_all_three_image_hashes_checked': True,
        'frozen_session_model_hashes_match': True,
        'stored_reference_predictions_reproduced': True,
        'new_detector_calls': 0, 'models_fitted': 0,
        'selected_camera_examples': len(selected),
        'selected_remaining_errors': sum(r['gh_y180'] != r['label'] for r in selected),
        'selected_recovered_errors': sum(r['gh_y180'] == r['label'] for r in selected),
        'all_capture_prediction_counts': {name: dict(Counter(r[name] for r in rows)) for name in MODELS},
        'all_captures_role': 'P01 development',
        'figure_sha256': digest(FIGURE), 'predictions_sha256': digest(OUT / 'predictions.csv'),
        'provenance_sha256': digest(OUT / 'provenance.json'),
    })
    print(f'Replayed {len(rows)} saved attempts; exported four recent examples (two G errors, two recovered H cases).')


if __name__ == '__main__':
    main()
