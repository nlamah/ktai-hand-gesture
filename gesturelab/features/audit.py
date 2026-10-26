"""Compare human finger-state annotations with the independent geometry module."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gesturelab.datasets.data import read_manifest,atomic_json,project_lock
from gesturelab.features.inspection import features_for_sample
from gesturelab.features.finger_features import FINGERS,validate_thresholds

STATES={'extended':'extended','bent':'bent','intermediate':'intermediate','unscorable':'unscorable'}


def load_thresholds(project,path=None):
    path=Path(path) if path else Path(project)/'finger_thresholds.json'
    return validate_thresholds(json.loads(path.read_text()) if path.exists() else None)


def read_annotations(project):
    path=Path(project)/'feature_checks.json'
    return json.loads(path.read_text()).get('annotations',{}) if path.exists() else {}


def save_annotation(project,sample_id,states,note=''):
    if set(states)!=set(FINGERS) or any(s not in STATES for s in states.values()):
        raise ValueError('Assess all five fingers, or select unscorable.')
    with project_lock(project):
        sample=next((s for s in read_manifest(project)['samples'] if s['id']==sample_id),None)
        if sample is None or sample['role']!='development':
            raise ValueError('Use development photos to inspect and tune features.')
        annotations=read_annotations(project)
        annotations[sample_id]={'states':states,'note':note,'updated_at':datetime.now(timezone.utc).isoformat()}
        atomic_json(Path(project)/'feature_checks.json',{'schema':1,'annotations':annotations})


def safe_ratio(a,b):
    return float(a/b) if b else None


def _annotation_records(project, annotations, thresholds):
    records, skipped = [], 0
    for sample in read_manifest(project)['samples']:
        if not sample['included'] or sample['role'] != 'development' or sample['id'] not in annotations:
            continue
        features = features_for_sample(sample, thresholds) if sample['status'] == 'ok' else None
        for finger, truth in annotations[sample['id']]['states'].items():
            if truth == 'unscorable':
                skipped += 1
                continue
            prediction = 'no_prediction'
            if features:
                prefix = f'geom_{finger}_'
                prediction = 'extended' if features[prefix + 'extended'] else 'bent' if features[prefix + 'bent'] else 'intermediate'
            records.append({'id': sample['id'], 'letter': sample['label'], 'finger': finger,
                            'truth': truth, 'prediction': prediction, 'detected': features is not None})
    return records, skipped


def _finger_scores(records):
    rows = []
    for finger in FINGERS:
        subset = [record for record in records if record['finger'] == finger]
        for target in ('extended', 'bent'):
            tp = sum(r['truth'] == target and r['prediction'] == target for r in subset)
            fp = sum(r['truth'] != target and r['prediction'] == target for r in subset)
            fn = sum(r['truth'] == target and r['prediction'] != target for r in subset)
            rows.append({'finger': finger, 'target': target, 'n': len(subset),
                         'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
                         'precision': safe_ratio(tp, tp + fp), 'recall': safe_ratio(tp, tp + fn),
                         'f1': safe_ratio(2 * tp, 2 * tp + fp + fn)})
    return pd.DataFrame(rows)


def evaluate_annotations(project, thresholds=None):
    """Evaluate independent human assessments, excluding reserved test photos."""
    thresholds = validate_thresholds(thresholds)
    records, skipped = _annotation_records(project, read_annotations(project), thresholds)
    meta = {
        'evaluated_fingers': len(records), 'unscorable_skipped': skipped,
        'coverage': safe_ratio(sum(r['detected'] for r in records), len(records)),
        'state_accuracy': safe_ratio(sum(r['truth'] == r['prediction'] for r in records), len(records)),
        'thresholds': thresholds,
        'note': 'Development diagnostic using manual assessments; not an independent test.',
        'undefined_metrics': 'null/blank cell when the denominator is zero; no artificial score of 0 or 1.',
    }
    return _finger_scores(records), pd.DataFrame(records), meta


def write_audit(project,out,thresholds=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    table,records,meta=evaluate_annotations(project,thresholds)
    if records.empty:
        raise ValueError('No scorable manual finger labels yet. Use apps/feature_check_app.py first.')
    table.to_csv(out/'finger_precision_recall.csv',index=False)
    records.to_csv(out/'finger_predictions.csv',index=False)
    atomic_json(out/'feature_audit.json',meta)
    return out
