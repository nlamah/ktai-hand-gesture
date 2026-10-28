"""Locked same-photo evaluation; detector failures stay visible and model errors count as errors."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
from PIL import Image

from gesturelab.datasets.data import read_manifest, fingerprint
from gesturelab.research.live import predict_one, correctness
from gesturelab.research.models import ModelHandle, compatible_models
from gesturelab.validation.metrics import UNKNOWN, score_rows
from gesturelab.validation.report import read_json, sha256, write_json, new_output


def _attempt_metadata(sample):
    keys = ('alphabet', 'role', 'captured_at', 'capture_order', 'repetition', 'pixel_sha256', 'image_path')
    return {**{key: sample.get(key) for key in keys},
            'retake_of': sample.get('provenance', {}).get('retake_of')}


def classify_records(records, models):
    """Every model gets the same saved detection object; intended labels never reach predict()."""
    rows = []
    for sample in records:
        detection = {key: sample[key] for key in ('status', 'image_landmarks', 'world_landmarks', 'handedness',
            'handedness_score', 'feature_version', 'features', 'n_hands') if key in sample}
        results = [predict_one(model, detection, sample['width'], sample['height']) for model in models]
        row = dict(id=sample['id'], label=sample['label'], condition=sample['condition'], **_attempt_metadata(sample),
            nominal_angle=sample.get('nominal_angle'), participant=sample['participant'], session=sample['session'],
            detection_status=sample['status'], usable=sample['status'] == 'ok',
            invalid_reason=sample.get('invalid_reason'), included=sample.get('included', True),
            shared_detection_sha256=hashlib.sha256(json.dumps({k: sample.get(k) for k in
                ('status', 'image_landmarks', 'world_landmarks', 'detector_configuration')}, sort_keys=True).encode()).hexdigest())
        for index, result in enumerate(results):
            letter = chr(97 + index)
            row.update({letter: result['prediction'] or UNKNOWN, letter + '_status': result['status'],
                        letter + '_error': result['error']})
        if len(results) == 2:
            row['category'] = correctness(row['a'], row['b'], sample['label']) if row['usable'] else 'detection_failure'
        rows.append(row)
    return rows


def score_evaluation(rows, evaluation_classes, model_classes):
    usable = [r for r in rows if r['usable'] and r['included']]
    scores = {}
    for side in ('a', 'b'):
        predictions = [dict(label=r['label'], prediction=r[side]) for r in usable]
        scores[side] = score_rows(predictions, evaluation_classes, prediction_labels=model_classes) if predictions else None
    coverage = defaultdict(lambda: dict(offered=0, included=0, detected=0, usable=0))
    for row in rows:
        for key in (('condition', row['condition']), ('letter', row['label']), ('condition_letter', row['condition'] + ':' + row['label'])):
            coverage[key]['offered'] += 1
            coverage[key]['included'] += int(row['included'])
            coverage[key]['detected'] += int(row['usable'])
            coverage[key]['usable'] += int(row['included'] and row['usable'])
    conditions = {}
    for condition in sorted({r['condition'] for r in rows}):
        subset = [r for r in usable if r['condition'] == condition]
        conditions[condition] = {side: score_rows([dict(label=r['label'], prediction=r[side]) for r in subset],
            evaluation_classes, prediction_labels=model_classes) if subset else None for side in ('a', 'b')}
    return dict(offered=len(rows), included=sum(r['included'] for r in rows), usable=len(usable), scores=scores,
        conditions=conditions, coverage=[dict(kind=k[0], value=k[1], **v,
            detection_coverage=v['detected'] / v['offered'] if v['offered'] else None) for k, v in coverage.items()],
        paired_categories=dict(Counter(r['category'] for r in usable)),
        model_errors={side: sum(r[side + '_status'] == 'model_error' for r in usable) for side in ('a', 'b')},
        missing_evaluation_classes=sorted(set(evaluation_classes) - {r['label'] for r in usable}),
        policy='Accuracy and fixed-class macro-F1 use the same included, detector-usable IDs. Out-of-stress model answers and model-specific failures count as wrong. Missing classes contribute zero to fixed-class macro-F1; coverage is separate.')


def training_overlap(metadata, records):
    plan_path = Path(metadata['experiment']) / 'split.json'
    if not plan_path.is_file():
        return dict(status='not_checked', reason='Training split unavailable.')
    plan = read_json(plan_path)
    project = Path(plan['project'])
    if not (project / 'manifest.json').is_file():
        return dict(status='not_checked', reason='Training manifest unavailable.')
    if fingerprint(project) != plan['dataset_fingerprint']:
        return dict(status='not_checked', reason='Training manifest changed since the frozen training split.')
    if metadata.get('split_sha256') and sha256(plan_path) != metadata['split_sha256']:
        return dict(status='not_checked', reason='Training split changed since model registration.')
    manifest = {s['id']: s for s in read_manifest(project)['samples']}
    training = [manifest[i] for i in plan['development_ids'] if i in manifest]
    ids = set(plan['development_ids']) & {r['id'] for r in records}
    hashes = {r['pixel_sha256'] for r in training if r.get('pixel_sha256')}
    pixels = [r['id'] for r in records if r.get('pixel_sha256') in hashes]
    complete = len(training) == len(plan['development_ids']) and all(r.get('pixel_sha256') for r in training + records)
    return dict(status='failed' if ids or pixels else ('passed' if complete else 'not_checked'),
        overlapping_ids=sorted(ids), overlapping_pixels=pixels,
        limitation='Stored exact pixel hashes only; near-duplicates, real identities and unlogged human exposure remain unverified.')


def _append_exposure(project, event):
    path = Path(project) / 'research_access.jsonl'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event) + '\n')


def exposure_history(project, protocol):
    path = Path(project) / 'research_access.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and json.loads(line).get('protocol_id') == protocol['id']] if path.exists() else []


def _verify_test_pixels(project, samples):
    for sample in samples:
        with Image.open(Path(project) / sample['image_path']) as source:
            image = source.convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest != sample['pixel_sha256'] or image.size != (sample['width'], sample['height']):
            raise ValueError('Test analysis pixels changed after recording.')
    settings = {json.dumps(s.get('detector_configuration'), sort_keys=True) for s in samples}
    if len(settings) != 1 or not samples[0].get('detector_configuration'):
        raise ValueError('Test detector configurations differ or are missing; do not pool them silently.')


def evaluate_test(seal, project, out, allow_incomplete=False, acknowledge_unchecked_overlap=False):
    frozen = read_json(seal)
    if sha256(frozen['protocol_file']) != frozen['protocol_sha256']:
        raise ValueError('The protocol no longer matches the pre-evaluation seal.')
    protocol = read_json(frozen['protocol_file'])
    models = [ModelHandle.load(metadata) for metadata in frozen['models']]
    compatibility = compatible_models(*models)
    samples = [s for s in read_manifest(project)['samples'] if s.get('provenance', {}).get('protocol_id') == protocol['id']]
    _validate_test_samples(samples, protocol, frozen, allow_incomplete)
    _verify_test_pixels(project, samples)
    history = exposure_history(project, protocol)
    if frozen.get('viewed_retest') and not history:
        raise ValueError('A viewed-set retest needs a recorded earlier evaluation of this protocol.')
    overlaps = [training_overlap(model.metadata, samples) for model in models]
    if any(o['status'] == 'failed' for o in overlaps):
        raise ValueError('Training/test photo or pixel overlap detected.')
    if not acknowledge_unchecked_overlap and any(o['status'] != 'passed' for o in overlaps):
        raise ValueError('Overlap could not be fully checked; restore provenance or explicitly acknowledge the missing evidence.')
    out = new_output(out, [project] + [m.metadata['experiment'] for m in models])
    rows = classify_records(samples, models)
    result = score_evaluation(rows, frozen['evaluation_classes'], compatibility['labels'])
    result.update(phase='final_test', seal=frozen, protocol=protocol, models=[m.metadata for m in models], overlap_checks=overlaps,
        previous_evaluations=len(history), previously_viewed_retest=bool(history) or frozen.get('viewed_retest', False),
        evaluated_at=datetime.now(timezone.utc).isoformat(), dataset_manifest_sha256=sha256(Path(project) / 'manifest.json'),
        limitation='Repeated evaluation is permitted and logged. Adaptations based on viewed results need new unseen data for independent confirmation.')
    _append_exposure(project, dict(at=result['evaluated_at'], protocol_id=protocol['id'], seal_sha256=sha256(seal),
                                  model_hashes=[m.metadata['model_sha256'] for m in models], output=str(out), event='final_test_evaluation'))
    from gesturelab.research.reporting import export_evaluation
    export_evaluation(out, result, rows, samples, project)
    return out


def _validate_test_samples(samples, protocol, frozen, allow_incomplete):
    if not samples or protocol['role'] != 'test':
        raise ValueError('A nonempty, sealed test protocol is required.')
    for sample in samples:
        provenance = sample.get('provenance', {})
        if (sample['role'] != 'test' or sample['alphabet'] != protocol['alphabet']
                or provenance.get('protocol_sha256') != frozen['protocol_sha256']
                or provenance.get('predictions_exposed') or sample['label'] not in frozen['evaluation_classes']):
            raise ValueError('Test attempt metadata differs from the locked protocol or predictions were exposed.')
        if datetime.fromisoformat(sample['captured_at']) < datetime.fromisoformat(frozen['frozen_at']) and not frozen.get('viewed_retest'):
            raise ValueError('Test pictures predate the frozen model/protocol selection.')
        _validate_trial(sample, protocol)
    missing = set(t['order'] for t in protocol['trials']) - {s['capture_order'] for s in samples}
    if missing and not allow_incomplete:
        raise ValueError(f'{len(missing)} planned attempts missing; finish capture or explicitly allow an incomplete evaluation.')


def _validate_trial(sample, protocol):
    trial = next((t for t in protocol['trials'] if t['order'] == sample['capture_order']), None)
    if not trial or any(sample[key] != trial[key] for key in ('label', 'condition', 'nominal_angle', 'repetition')):
        raise ValueError('Recorded intended label/condition differs from its planned trial.')
    if any(sample[key] != protocol[key] for key in ('participant', 'session')):
        raise ValueError('Participant/session differs from the protocol.')
