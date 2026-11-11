"""Read-only installation/data check and independent shared-DS score audit."""
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

from baseline.runner import ROOT


def digest(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def status():
    from baseline.shin import VERSION
    required = ['models/hand_landmarker.task', 'assets/asl_feature_references.xlsx', 'config/thresholds.json']
    versions = {}
    for name in ['numpy', 'scikit-learn', 'mediapipe', 'lightgbm', 'streamlit']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'NOT INSTALLED'
    files = {p: (ROOT / p).exists() for p in required}
    data = {p: (ROOT / p).exists() for p in ['data/asl_kaggle/manifest.json',
            'data/asl_kaggle/research/inputs.npz', 'data/asl_kaggle/research/summary.json',
            'data/asl_kaggle/research/inputs_metadata.json', 'runs/mirror_01/detections']}
    return {'baseline': 'Shin DS: 190 distances + 630 directions = 820 inputs',
            'feature_version': VERSION, 'versions': versions, 'required_files': files,
            'local_reproduction_data': data,
            'contributions': {rq: json.loads((ROOT / rq / 'contribution.json').read_text())['status']
                              for rq in ['RQ1', 'RQ2', 'RQ3']}}


def verify_inputs():
    """Verify an extracted input bundle, or the canonical four paper input files."""
    bundle_record = ROOT / 'reproduction_inputs.json'
    if bundle_record.exists():
        expected = json.loads(bundle_record.read_text())['files_sha256']
    else:
        protocol = json.loads((ROOT / 'results/feature_factorial_01/protocol.json').read_text())
        expected = {**protocol['input_sha256'],
                    'data/asl_kaggle/manifest.json': protocol['historical_input_hashes']['manifest']}
    for name, value in expected.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('data', 'runs'):
            raise ValueError(f'Ongeldig invoerpad in bundelmanifest: {name}')
        path = ROOT / relative
        if not path.is_file() or digest(path) != value:
            raise ValueError(f'Ontbrekende of gewijzigde paperinvoer: {name}')
    return {'status': 'passed', 'input_files_verified': len(expected),
            'scope': 'Byte-identical supplied inputs; no redetection or model fitting.'}


def verify_baseline():
    """Recompute both DS baselines from held-out predictions, without model fits."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    path = ROOT / 'results/feature_factorial_01'
    frame = pd.read_csv(path / 'predictions.csv', keep_default_na=False)
    scores = pd.read_csv(path / 'metrics.csv', keep_default_na=False)
    output = []
    reference_ids = None
    for model in ['svm', 'lightgbm']:
        rows = frame[(frame.model == model) & (frame.variant == 'DS')]
        if len(rows) != 60251 or rows.id.duplicated().any():
            raise ValueError('DS predictions must contain 60,251 unique sources.')
        if reference_ids is None:
            reference_ids = rows[['id', 'label', 'group', 'fold']].reset_index(drop=True)
        else:
            pd.testing.assert_frame_equal(reference_ids, rows[['id', 'label', 'group', 'fold']].reset_index(drop=True))
        for fold in ['pooled', '1', '2', '3', '4', '5']:
            part = rows if fold == 'pooled' else rows[rows.fold == int(fold)]
            p, r, f, _ = precision_recall_fscore_support(part.label, part.prediction,
                                                       labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), zero_division=0)
            measured = {'images': len(part), 'correct': int((part.label == part.prediction).sum()),
                        'accuracy': float((part.label == part.prediction).mean()),
                        'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean())}
            saved = scores[(scores.model == model) & (scores.variant == 'DS') & (scores.fold == fold)]
            if len(saved) != 1:
                raise ValueError(f'Ontbrekende score: {model}, fold {fold}')
            for key, value in measured.items():
                np.testing.assert_allclose(value, saved.iloc[0][key], atol=1e-12, rtol=0)
            if fold == 'pooled':
                output.append({'model': model, **measured})
    return {'status': 'passed', 'independently_checked_score_rows': 12,
            'scope': '60251 accepted original sources; shared DS baseline, not DSJ',
            'predictions_sha256': digest(path / 'predictions.csv'), 'pooled': output}
