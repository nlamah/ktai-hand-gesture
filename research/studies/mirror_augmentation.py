"""Frozen five-fold original-versus-RGB-mirror training, including a balanced test mix."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight

from research.data import atomic_json
from research.inputs import ANGLE_COLUMNS, LABELS, write_csv
from research.live_model import BUNDLE_VERSION, SHIN_VERSION, load_bundle
from research.mirroring import weighted_training_rows
from research.studies.compare_angles import classifier

SEED = 20261106
ARMS = ('original_training', 'mirror_training')
VIEWS = ('original', 'mirrored_rgb')


def load_extracted(work, out):
    completion = json.loads((out / 'extraction_complete.json').read_text())
    paths = sorted((work / 'detections').glob('chunk_*.npz'))
    if len(paths) != completion['chunks']:
        raise ValueError('Incomplete detection cache.')
    n = completion['source_images']
    features = np.empty((n, 2, 835), dtype=np.float32)
    status = np.empty((n, 2), dtype='<U24')
    hands = np.empty((n, 2), dtype='<U5')
    ids, labels, groups, offset = [], [], [], 0
    for path in paths:
        with np.load(path) as data:
            if str(data['protocol_hash']) != completion['protocol_hash']:
                raise ValueError('Mixed extraction protocols.')
            count = len(data['ids']); end = offset + count
            features[offset:end] = data['features']
            status[offset:end] = data['status']; hands[offset:end] = data['handedness']
            ids.extend(data['ids'].tolist()); labels.extend(data['labels'].tolist()); groups.extend(data['groups'].tolist())
            offset = end
    if offset != n or len(set(ids)) != n:
        raise ValueError('Incomplete or duplicate source population.')
    if not np.isfinite(features[status == 'ok']).all():
        raise ValueError('Accepted descriptors contain missing or infinite values.')
    return features, np.array(labels), np.array(ids), np.array(groups), status, hands


def assignments_for(groups, folds):
    assignments = np.zeros(len(groups), dtype=int)
    for fold in folds:
        train = np.isin(groups, fold['train_groups'])
        test = np.isin(groups, fold['test_groups'])
        if (train & test).any() or not (train | test).all() or assignments[test].any():
            raise ValueError('Invalid or overlapping source-group folds.')
        assignments[test] = fold['fold']
    if not assignments.all():
        raise ValueError('A source has no held-out fold.')
    return assignments


def balanced_views(labels, assignments, seed=SEED):
    """One reproducible view per source, half mirrored within each letter/fold."""
    labels, assignments = np.asarray(labels), np.asarray(assignments)
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(labels), dtype=np.int8)
    for fold in sorted(set(assignments)):
        for letter in LABELS:
            indices = np.flatnonzero((assignments == fold) & (labels == letter))
            selected[rng.permutation(indices)[:len(indices)//2]] = 1
    return selected


def fit_mirrored(baseline, originals, mirrors, labels, mirror_ok):
    """Hold preprocessing and SVM gamma/class priors fixed to the original fit."""
    rows, targets, weights = weighted_training_rows(originals, mirrors, labels, mirror_ok)
    preprocessing = deepcopy(baseline[0])
    learner = clone(baseline[-1])
    if hasattr(baseline[-1], '_gamma'):
        classes = np.unique(labels)
        prior = compute_class_weight('balanced', classes=classes, y=labels)
        learner.set_params(gamma=baseline[-1]._gamma, class_weight=dict(zip(classes, prior)))
    learner.fit(preprocessing.transform(rows), targets, sample_weight=weights)
    return make_pipeline(preprocessing, learner), len(rows)


def prediction_views(model, features, statuses):
    predictions = np.full(statuses.shape, '?', dtype='<U1')
    for view in range(2):
        accepted = statuses[:, view] == 'ok'
        if accepted.any():
            predictions[accepted, view] = model.predict(features[accepted, view])
    return predictions


def metrics(truth, pred):
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=LABELS, zero_division=0)
    return {'images': len(truth), 'correct': int((truth == pred).sum()),
            'accuracy': float(accuracy_score(truth, pred)), 'macro_precision': float(p.mean()),
            'macro_recall': float(r.mean()), 'macro_f1': float(f.mean())}, (p, r, f, n)


def export_results(out, labels, ids, groups, status, hands, assignments, selected, all_predictions):
    rows, per_letter, paired, hand_metrics = [], [], [], []
    selection = np.arange(len(labels))
    for model, arms in all_predictions.items():
        for presentation in (*VIEWS, 'balanced_mix'):
            view = np.full(len(labels), VIEWS.index(presentation)) if presentation in VIEWS else selected
            accepted = status[selection, view] == 'ok'
            baseline = arms[ARMS[0]][selection, view]
            candidate = arms[ARMS[1]][selection, view]
            for hand in ('Left', 'Right'):
                ix = accepted & (hands[selection, view] == hand)
                for arm, pred in zip(ARMS, (baseline, candidate)):
                    values, _ = metrics(labels[ix], pred[ix])
                    hand_metrics.append({'model': model, 'training': arm, 'presentation': presentation,
                        'detector_handedness': hand, 'scope': 'accepted_only_detector_label_not_ground_truth', **values})
            for scope in ('all_attempts', 'accepted_only', 'paired_both_accepted'):
                mask = np.ones(len(labels), bool) if scope == 'all_attempts' else accepted if scope == 'accepted_only' else (status == 'ok').all(axis=1)
                for fold in ('pooled', *sorted(set(assignments))):
                    ix = mask if fold == 'pooled' else mask & (assignments == fold)
                    if not ix.any():
                        continue
                    truth = labels[ix]
                    a, b = baseline[ix], candidate[ix]
                    for arm, pred in zip(ARMS, (a, b)):
                        values, detail = metrics(truth, pred)
                        row = {'model': model, 'training': arm, 'presentation': presentation,
                               'scope': scope, 'fold': fold, 'coverage': float(accepted[ix].mean()), **values}
                        rows.append(row)
                        if fold == 'pooled':
                            per_letter.extend({'model': model, 'training': arm, 'presentation': presentation,
                                'scope': scope, 'letter': letter, 'images': int(detail[3][j]),
                                'precision': float(detail[0][j]), 'recall': float(detail[1][j]), 'f1': float(detail[2][j])}
                                for j, letter in enumerate(LABELS))
                            matrix = confusion_matrix(truth, pred, labels=[*LABELS, '?'])
                            write_csv(out / f'confusion_{model}_{arm}_{presentation}_{scope}.csv',
                                [{'true_letter': l, **dict(zip([*LABELS, '?'], map(int, r)))}
                                 for l, r in zip([*LABELS, '?'], matrix)])
                    paired.append({'model': model, 'presentation': presentation, 'scope': scope, 'fold': fold,
                        'images': len(truth), 'fixed': int(((a != truth) & (b == truth)).sum()),
                        'lost': int(((a == truth) & (b != truth)).sum()),
                        'net': int((b == truth).sum() - (a == truth).sum())})
    write_csv(out / 'metrics.csv', rows); write_csv(out / 'per_letter.csv', per_letter)
    write_csv(out / 'paired_changes.csv', paired)
    write_csv(out / 'handedness_metrics.csv', hand_metrics)
    import csv
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fields = ['model', 'id', 'label', 'group', 'fold', 'view', 'status', 'detected_handedness',
                  'balanced_selected', *ARMS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for model, arms in all_predictions.items():
            for i in range(len(labels)):
                for view in range(2):
                    writer.writerow(dict(model=model, id=ids[i], label=labels[i], group=groups[i],
                        fold=assignments[i], view=VIEWS[view], status=status[i, view],
                        detected_handedness=hands[i, view], balanced_selected=int(selected[i] == view),
                        **{arm: arms[arm][i, view] for arm in ARMS}))
    detection_rows = []
    for view, name in enumerate(VIEWS):
        for letter in ['ALL', *LABELS]:
            ix = np.ones(len(labels), bool) if letter == 'ALL' else labels == letter
            good = ix & (status[:, view] == 'ok')
            detection_rows.append({'view': name, 'letter': letter, 'images': int(ix.sum()),
                'accepted': int(good.sum()), 'coverage': float(good.sum()/ix.sum()),
                'detected_Left': int((good & (hands[:, view] == 'Left')).sum()),
                'detected_Right': int((good & (hands[:, view] == 'Right')).sum()),
                'no_hand': int((ix & (status[:, view] == 'no_hand')).sum()),
                'other_rejections': int((ix & ~good & (status[:, view] != 'no_hand')).sum())})
    write_csv(out / 'detection_counts.csv', detection_rows)
    good_both = (status == 'ok').all(axis=1)
    summary = {'source_images': len(labels), 'folds': len(set(assignments)),
        'balanced_original': int((selected == 0).sum()), 'balanced_mirrored': int((selected == 1).sum()),
        'balanced_accepted': int((status[selection, selected] == 'ok').sum()),
        'balanced_detected_Left': int(((status[selection, selected] == 'ok') & (hands[selection, selected] == 'Left')).sum()),
        'balanced_detected_Right': int(((status[selection, selected] == 'ok') & (hands[selection, selected] == 'Right')).sum()),
        'both_views_accepted': int(good_both.sum()),
        'handedness_swaps_when_both_accepted': int((good_both & (hands[:, 0] != hands[:, 1])).sum()),
        'detection': [r for r in detection_rows if r['letter'] == 'ALL'],
        'metrics': [r for r in rows if r['fold'] == 'pooled'],
        'paired': [r for r in paired if r['fold'] == 'pooled'], 'handedness_metrics': hand_metrics}
    atomic_json(out / 'summary.json', summary)
    return summary


def training_protocol(out, folds_path):
    extraction = json.loads((out / 'extraction_protocol.json').read_text())
    return {'experiment': 'original-only versus original + horizontal RGB mirror',
        'seed': SEED, 'features': 835, 'models': ['svm', 'lightgbm'],
        'folds_sha256': sha256(folds_path.read_bytes()).hexdigest(),
        'extraction_sha256': sha256((out / 'extraction_protocol.json').read_bytes()).hexdigest(),
        'test_mix': 'Exactly one view per source; half mirrored within each letter/fold, fixed seed, shared by learners',
        'paired_test': 'Also evaluate both views of every held-out source separately; never treat views as independent sources',
        'training_sources': 'Freshly accepted originals only; add mirror only for these same fitting sources if accepted',
        'weights': 'Original and mirror each .5 when both accepted; original alone 1 otherwise',
        'preprocessing': 'Fit on original fitting rows only; preserve for both models',
        'svm': 'C=10 RBF; gamma and balanced class weights taken from original fitting rows',
        'lightgbm': '100 trees, 31 leaves, learning_rate .1, min_child_samples 20, seed 42, 4 threads',
        'rejections': 'No prediction (?) for failed detection; all-attempt metrics include these failures',
        'policy': 'One predefined mirror recipe; no tuning or candidate selection from held-out outcomes',
        'scope': 'Synthetic mirroring robustness, not validated real left-handed signer performance',
        'training_source_sha256': {p: sha256(Path(p).read_bytes()).hexdigest() for p in
            ('research/studies/mirror_augmentation.py', 'research/mirroring.py',
             'research/studies/compare_angles.py', 'research/studies/compare_models.py', 'research/classifiers.py')},
        'source_manifest_sha256': extraction['manifest_sha256']}


def save_live(model, path, protocol, training_images, training_rows, augmentation):
    if path.exists():
        raise FileExistsError(f'Live artifact exists: {path}; choose a new path.')
    metadata = {'version': BUNDLE_VERSION, 'shin_version': SHIN_VERSION, 'input_variant': 'shin_angles',
        'angle_columns': list(ANGLE_COLUMNS), 'sklearn_version': importlib.metadata.version('scikit-learn'),
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_images': training_images,
        'training_rows': training_rows, 'source_hashes': {'manifest': protocol['source_manifest_sha256'],
            'extraction_protocol': protocol['extraction_sha256']},
        'augmentation': {'recipe': 'horizontal_rgb_mirror' if augmentation else 'fresh_original_only',
                         'mirror_training': augmentation},
        'experiment_protocol': protocol,
        'scope': 'Full-data deployment fit; cross-validation scores belong to the recipe, not this fitted model.'}
    temporary = path.with_suffix('.partial.joblib')
    joblib.dump({**metadata, 'estimator': model}, temporary, compress=3)
    load_bundle(temporary); temporary.replace(path)
    atomic_json(path.with_suffix('.json'), metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cv', 'train'), required=True)
    parser.add_argument('--work', type=Path, default=Path('runs/mirror_01'))
    parser.add_argument('--out', type=Path, default=Path('results/mirror_01'))
    parser.add_argument('--folds', type=Path, default=Path('data/asl_kaggle/research/summary.json'))
    args = parser.parse_args()
    protocol = training_protocol(args.out, args.folds)
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Training configuration changed; choose a new experiment.')
    atomic_json(protocol_path, protocol)
    X, labels, ids, groups, status, hands = load_extracted(args.work, args.out)
    folds = json.loads(args.folds.read_text())['folds']
    assignments = assignments_for(groups, folds)
    selected = balanced_views(labels, assignments)
    write_csv(args.out / 'test_mix.csv', [{'id': id_, 'letter': labels[i], 'fold': int(assignments[i]),
        'view': VIEWS[selected[i]]} for i, id_ in enumerate(ids)])
    all_predictions, fitting = {}, []
    if args.stage == 'cv':
        for name in protocol['models']:
            predictions = {arm: np.full((len(labels), 2), '?', dtype='<U1') for arm in ARMS}
            for fold in sorted(set(assignments)):
                train = (assignments != fold) & (status[:, 0] == 'ok')
                test = assignments == fold
                if set(groups[train]) & set(groups[test]):
                    raise ValueError('Source group leaked across training and evaluation.')
                base_path = args.work / f'{name}_original_fold{fold}.joblib'
                candidate_path = args.work / f'{name}_mirrored_fold{fold}.joblib'
                started = time.monotonic()
                baseline = joblib.load(base_path) if base_path.exists() else classifier(name, 'shin_angles', train.sum()).fit(X[train, 0], labels[train])
                if not base_path.exists(): joblib.dump(baseline, base_path)
                if candidate_path.exists():
                    candidate = joblib.load(candidate_path)
                    n_rows = int(train.sum() + ((status[:, 1] == 'ok') & train).sum())
                else:
                    candidate, n_rows = fit_mirrored(baseline, X[train, 0], X[train, 1], labels[train], status[train, 1] == 'ok')
                    joblib.dump(candidate, candidate_path)
                for arm, fitted in zip(ARMS, (baseline, candidate)):
                    predictions[arm][test] = prediction_views(fitted, X[test], status[test])
                fitting.append({'model': name, 'fold': int(fold), 'original_sources': int(train.sum()),
                    'mirror_rows': int(n_rows-train.sum()), 'augmented_rows': n_rows,
                    'training_source_ids_sha256': sha256('\n'.join(ids[train]).encode()).hexdigest(),
                    'held_out_source_ids_sha256': sha256('\n'.join(ids[test]).encode()).hexdigest(),
                    'held_out_sources': int(test.sum()), 'seconds': time.monotonic()-started})
                print(json.dumps(fitting[-1]), flush=True)
                del baseline, candidate
            all_predictions[name] = predictions
        write_csv(args.out / 'fitting.csv', fitting)
        summary = export_results(args.out, labels, ids, groups, status, hands, assignments, selected, all_predictions)
        for r in summary['metrics']:
            if r['scope'] == 'all_attempts': print(json.dumps(r), flush=True)
    else:
        train = status[:, 0] == 'ok'
        base_path = args.work / 'svm_original_full.joblib'
        baseline = joblib.load(base_path) if base_path.exists() else classifier('svm', 'shin_angles', train.sum()).fit(X[train, 0], labels[train])
        if not base_path.exists(): joblib.dump(baseline, base_path)
        candidate, n_rows = fit_mirrored(baseline, X[train, 0], X[train, 1], labels[train], status[train, 1] == 'ok')
        for fitted, path, rows, augmented in [(baseline, Path('models/asl_shin_angles_mirror_baseline.joblib'), int(train.sum()), False),
                 (candidate, Path('models/asl_shin_angles_mirrored.joblib'), n_rows, True)]:
            save_live(fitted, path, protocol, int(train.sum()), rows, augmented)
            print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
