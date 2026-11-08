"""Fresh DS replication on an explicit cache, isolated from historical outputs."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from baseline.check import digest
from baseline.data import load_shin_inputs
from baseline.models import build_classifier
from baseline.runner import ROOT
from research.data import atomic_json
from research.inputs import LABELS, write_csv


def train_cache(cache, output, models):
    cache = Path(cache).resolve()
    output = Path(output).resolve()
    X, y, ids, groups, folds = load_shin_inputs(cache)
    protocol = {'version': 'shared-ds-replication-v1', 'inputs': 820,
                'cache_sha256': {name: digest(cache / name) for name in ['inputs.npz', 'inputs_metadata.json', 'summary.json']},
                'models': list(models), 'source_groups': sorted(set(groups.tolist())),
                'source_sha256': {p: digest(ROOT / p) for p in ['baseline/train.py', 'baseline/models.py', 'baseline/data.py',
                                                               'research/shin.py', 'research/classifiers.py',
                                                               'research/studies/compare_models.py']},
                'scope': 'New replication; no inherited historical score or fixed detection count assumed.'}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('This output directory uses different inputs/settings. Choose a new directory.')
    atomic_json(path, protocol)
    metrics, predictions = [], []
    for name in models:
        pred = np.full(len(y), '', dtype='<U1')
        assignment = np.zeros(len(y), int)
        for fold, tr, te in folds:
            model_path = output / f'{name}_fold{fold}.joblib'
            model = joblib.load(model_path) if model_path.exists() else build_classifier(name, len(tr)).fit(X[tr], y[tr])
            if not model_path.exists():
                joblib.dump(model, model_path, compress=3)
            pred[te] = model.predict(X[te])
            assignment[te] = fold
            print(f'{name} fold {fold}: {len(tr)} train, {len(te)} toets', flush=True)
        for fold in ['pooled', *sorted(set(assignment))]:
            keep = np.ones(len(y), bool) if fold == 'pooled' else assignment == fold
            p, r, f, _ = precision_recall_fscore_support(y[keep], pred[keep], labels=LABELS, zero_division=0)
            metrics.append({'model': name, 'variant': 'DS', 'fold': fold, 'images': int(keep.sum()),
                            'correct': int((y[keep] == pred[keep]).sum()),
                            'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean())})
        predictions.extend({'model': name, 'id': ids[i], 'label': y[i], 'group': groups[i],
                            'fold': int(assignment[i]), 'prediction': pred[i]} for i in range(len(y)))
    write_csv(output / 'metrics.csv', metrics)
    write_csv(output / 'predictions.csv', predictions)
    print(json.dumps([r for r in metrics if r['fold'] == 'pooled'], indent=2), flush=True)
