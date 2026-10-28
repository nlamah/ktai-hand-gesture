"""Development-only inspection and explicit, separate human annotations."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

from gesturelab.datasets.data import read_manifest
from gesturelab.research.evaluation import classify_records, score_evaluation
from gesturelab.research.models import ModelHandle, compatible_models
from gesturelab.research.reporting import export_evaluation, write_csv
from gesturelab.validation.metrics import read_csv
from gesturelab.validation.report import read_json, write_json, new_output


def analyze_development(model_a, project, out, model_b=None, letters=None):
    models = [ModelHandle.load(model_a), ModelHandle.load(model_b or model_a)]
    compatibility = compatible_models(*models)
    all_samples = read_manifest(project)['samples']
    samples = [s for s in all_samples if s['role'] == 'development'
               and s.get('provenance', {}).get('protocol_role') != 'test' and s.get('included', True)]
    if letters:
        samples = [s for s in samples if s['label'] in letters]
    if not samples:
        raise ValueError('No selected development photos; reserved test photos are never shown here.')
    if any(s.get('alphabet', s.get('provenance', {}).get('language')) not in (None, compatibility['alphabet']) for s in samples):
        raise ValueError('Photo alphabet differs from the selected model alphabet.')
    classes = list(letters or sorted({s['label'] for s in samples}))
    if not set(classes) <= set(compatibility['labels']):
        raise ValueError('Development labels fall outside the original model classes.')
    out = new_output(out, [project] + [m.metadata['experiment'] for m in models])
    rows = classify_records(samples, models)
    result = score_evaluation(rows, classes, compatibility['labels'])
    result.update(phase='development_exploration', models=[m.metadata for m in models],
        intended_labels='Researcher-supplied labels, never automatically replaced by predictions.',
        skipped_reserved_test=sum(s['role'] == 'test' for s in all_samples),
        limitation='Viewed development images are not an independent final test; manual notes are hypotheses.')
    export_evaluation(out, result, rows, samples, project)
    return out


def import_notes(analysis, notes):
    analysis = Path(analysis)
    allowed = {r['id'] for r in read_csv(analysis / 'predictions.csv')}
    incoming = read_json(notes)
    if any(row.get('id') not in allowed or not str(row.get('note', '')).strip() for row in incoming):
        raise ValueError('Every human note needs an existing evaluated photo ID and nonempty text.')
    path = analysis / 'human_notes.json'
    previous = read_json(path) if path.exists() else []
    previous.extend({**row, 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'assessment_type': 'Explicit human annotation, not a proven automatic cause.'} for row in incoming)
    write_json(path, previous)
    return path


def export_live(source, out):
    source = Path(source)
    out = new_output(out, [source])
    rows = []
    for file in sorted(source.glob('*/observation.json')):
        observation = read_json(file)
        row = dict(frame_id=observation['frame_id'], intended_label=observation['intended_label'],
            a=observation['raw_predictions'][0], b=observation['raw_predictions'][1],
            category=observation['category'], detection_status=observation['detection']['status'])
        row.update({chr(97 + i) + '_model_sha256': model.get('model_sha256') for i, model in enumerate(observation['models'])})
        rows.append(row)
        if row['a'] != row['b']:
            folder = out / 'differences' / row['frame_id']
            folder.mkdir(parents=True)
            shutil.copyfile(file.parent / 'frame.png', folder / 'frame.png')
            shutil.copyfile(file, folder / 'observation.json')
    write_csv(out / 'selected_frames.csv', rows)
    write_json(out / 'summary.json', dict(selected_frames=len(rows), categories=dict(Counter(r['category'] for r in rows)),
        note='Descriptive selected development frames only. Adjacent video frames are not independent experiments; agreement without a label is not correctness.'))
    return out
