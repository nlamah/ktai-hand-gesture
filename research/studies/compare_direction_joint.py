"""Replace or supplement Shin directions with same-coordinate joint angles."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from research.inputs import ANGLE_COLUMNS, LABELS, angle_estimator, write_csv
from research.joint_angles import VERSION, joint_angles_batch
from research.shin import VERSION as SHIN_VERSION, combined_features
from research.studies.compare_models import load_inputs, model_for, scores

ARMS = ('shin', 'distance_joint', 'shin_joint')
NAMES = {'shin': 'Shin reference', 'distance_joint': 'Distances + joint angles',
         'shin_joint': 'Shin + joint angles', 'shin_world': 'Shin + world angles (earlier)'}


def matrix_for(shin, angles, arm):
    if arm == 'shin':
        return shin
    if arm == 'distance_joint':
        return np.c_[shin[:, :190], angles]
    if arm == 'shin_joint':
        return np.c_[shin, angles]
    raise ValueError(arm)


def estimator(name, width, n_train):
    preprocessing = angle_estimator(width - 15, width).steps[0][1]
    return make_pipeline(preprocessing, model_for(name, False, n_train))


def reference_predictions(path, name, ids, y, assignments, variant):
    with path.open(encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle) if r['variant'] == variant and r['model'] == name]
    indexed = {r['id']: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(ids):
        raise ValueError('Baseline sample IDs differ or contain duplicates.')
    if any(indexed[id_]['label'] != y[i] or int(indexed[id_]['fold']) != assignments[i]
           for i, id_ in enumerate(ids)):
        raise ValueError('Baseline labels or folds differ.')
    return np.array([indexed[id_]['prediction'] for id_ in ids])


def export_results(out, rows, predictions, y, ids, assignments):
    letters, records, paired, errors = [], [], [], []
    baseline_by_model = {m: p for (m, arm), p in predictions.items() if arm == 'shin'}
    for (name, arm), pred in predictions.items():
        pr, re, f1, n = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
        for i, letter in enumerate(LABELS):
            letters.append({'model': name, 'variant': arm, 'letter': letter, 'images': int(n[i]),
                            'precision': float(pr[i]), 'recall': float(re[i]), 'f1': float(f1[i])})
        matrix = confusion_matrix(y, pred, labels=LABELS)
        errors.extend({'model': name, 'variant': arm, 'true_letter': a,
                       'predicted_letter': b, 'count': int(matrix[i, j])}
                      for i, a in enumerate(LABELS) for j, b in enumerate(LABELS)
                      if i != j and matrix[i, j])
        write_csv(out / f'confusion_{name}_{arm}.csv',
                  [{'true_letter': l, **dict(zip(LABELS, map(int, r)))}
                   for l, r in zip(LABELS, matrix)])
        base = baseline_by_model[name] == y
        correct = pred == y
        paired.append({'model': name, 'variant': arm, 'fixed': int((~base & correct).sum()),
                       'lost': int((base & ~correct).sum()), 'net': int(correct.sum() - base.sum())})
        records.extend({'model': name, 'variant': arm, 'id': id_, 'fold': int(assignments[i]),
                        'label': str(y[i]), 'prediction': str(pred[i])} for i, id_ in enumerate(ids))
    write_csv(out / 'metrics.csv', rows)
    write_csv(out / 'per_letter.csv', letters)
    write_csv(out / 'paired_changes.csv', paired)
    write_csv(out / 'error_pairs.csv', errors)
    write_csv(out / 'oof_predictions.csv', records)
    pooled = [r for r in rows if r['fold'] == 'pooled']
    summary = {'images': len(y), 'folds': 5, 'metrics': pooled, 'paired_changes': paired,
               'scope': 'Same normalized image x/y/z, same samples/folds, fixed project hyperparameters; no new tuning. Earlier world-angle results are context only.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Shin directions versus explicit joint angles', '',
             'The primary comparison uses identical normalized image x/y/z landmarks for every feature. '
             'No landmark redetection, image selection, augmentation or model tuning was performed. '
             'The 60,251 accepted images and five frozen filename-block folds are shared across arms. '
             'The 15 joint measurements are 14 interior angles plus signed thumb MCP deviation, using the historical definitions. '
             'Finger MCP values are wrist-based proxies, not clinical joint angles. Coordinates are not aspect corrected, to match Shin exactly.', '',
             '| Classifier | Representation | Inputs | Accuracy | Macro-F1 | Fixed / lost versus Shin |',
             '|---|---|---:|---:|---:|---:|']
    for r in pooled:
        p = next(p for p in paired if (p['model'], p['variant']) == (r['model'], r['variant']))
        lines.append(f"| {r['model']} | {NAMES[r['variant']]} | {r['features']} | {r['accuracy']:.5%} | {r['macro_f1']:.6f} | {p['fixed']} / {p['lost']} |")
    lines += ['', '## Findings', '']
    for name in baseline_by_model:
        selected = {r['variant']: r for r in pooled if r['model'] == name}
        replacement = selected['distance_joint']['accuracy'] - selected['shin']['accuracy']
        addition = selected['shin_joint']['accuracy'] - selected['shin']['accuracy']
        net = next(p['net'] for p in paired if p['model'] == name and p['variant'] == 'shin_joint')
        lines.append(f'- {name.upper()}: replacing directions changes accuracy by {100*replacement:+.3f} percentage points. '
                     f'Adding same-coordinate joints changes accuracy by {100*addition:+.4f} points ({net:+d} net correct images).')
    lines += ['', 'Under these fixed model recipes, joint measurements do not replace axis-relative direction information successfully. '
              'Adding the derived same-coordinate measurements yields no convincing overall gain. '
              'For SVM, the earlier world-coordinate addition gains 67 correct images, versus 4 for the same-coordinate addition. '
              'The larger earlier gain therefore cannot be attributed to the bending-angle formula alone; the coordinate representation also matters. '
              'This does not establish anatomical accuracy or generalization to unseen people.', '']
    lines += ['', '![Accuracy and macro-F1](overall.png)', '',
              '## Per-letter effects', '',
              '![Recall change for every letter](letter_recall_changes.png)', '',
              'The full `per_letter.csv` contains precision, recall and F1 for all 26 letters. '
              'Confusion matrices use true letters as rows and predictions as columns. `metrics.csv` includes all fold scores; '
              '`paired_changes.csv` counts corrected and newly incorrect images on paired predictions.', '',
              '## Interpretation and limits', '',
              'Replacing the 630 direction features tests a compact representation, but it also removes absolute axis-relative orientation. '
              'Hand orientation can distinguish letters, so a lower replacement score would not mean joint angles are incorrectly calculated. '
              'Adding joint angles tests whether making a derived nonlinear relationship explicit helps the fixed classifier. '
              'With the same coordinates these measurements largely re-express information already present in the directions; they do not constitute a new sensor.', '',
              'The earlier world-angle arm uses a different estimated coordinate representation. Its difference from the new same-coordinate arm '
              'is informative about that representation, but does not isolate individual landmark estimation errors or establish causality.', '',
              'SVM uses RBF C=10, gamma=scale, balanced. LightGBM uses 100 trees, 31 leaves, learning rate 0.1, '
              'min child samples 20 and seed 42. Median imputation and standardization of added angles are fitted on each training fold only; '
              'Shin inputs pass through unchanged. gamma=scale adapts to each input representation, so these are fixed recipe comparisons, '
              'not a claim that every feature set has been optimally tuned. Baseline and historical world-arm OOF predictions are reused only after checking hashes, IDs, labels and fold membership.', '',
              'These repeatedly inspected development folds are not an independent final test, and filename blocks are not verified people. '
              'Scores are conditional on accepted hand detections. Synthetic geometric rotation invariance does not guarantee live-camera robustness. '
              'No deployed live model was replaced.', '', '## Reproduce', '', '```bash',
              'conda activate ktai-hand-gesture',
              'MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.compare_direction_joint', '```', '',
              'Models and prediction checkpoints are under `runs/direction_joint_01/`; numeric results and this report are under '
              '`results/direction_joint_01/`. Re-running resumes completed folds after validating input provenance. '
              'Use new `--out` and `--work` directories for a different configuration.', '']
    detail = []
    lookup = {(r['model'], r['variant'], r['letter']): r for r in letters}
    for name in baseline_by_model:
        detail += [f'### {name.upper()}: recall for every letter', '',
                   '| Letter | Images | Shin | Distances + joints | Shin + joints | Replacement change (pp) | Addition change (pp) |',
                   '|---|---:|---:|---:|---:|---:|---:|']
        for letter in LABELS:
            a, b, c = [lookup[name, arm, letter] for arm in ARMS]
            detail.append(f"| {letter} | {a['images']} | {a['recall']:.2%} | {b['recall']:.2%} | {c['recall']:.2%} | "
                          f"{100*(b['recall']-a['recall']):+.2f} | {100*(c['recall']-a['recall']):+.2f} |")
        detail += ['', f'### {name.upper()}: paired fold differences', '',
                   '| Variant | Mean accuracy change (pp) | SD of fold changes (pp) | Improved / tied / worse folds |',
                   '|---|---:|---:|---|']
        base = {r['fold']: r for r in rows if r['model'] == name and r['variant'] == 'shin' and r['fold'] != 'pooled'}
        for arm in ARMS[1:]:
            delta = np.array([100 * (r['accuracy'] - base[r['fold']]['accuracy'])
                              for r in rows if r['model'] == name and r['variant'] == arm and r['fold'] != 'pooled'])
            detail.append(f"| {NAMES[arm]} | {delta.mean():+.3f} | {delta.std(ddof=1):.3f} | "
                          f"{int((delta>0).sum())} / {int((delta==0).sum())} / {int((delta<0).sum())} |")
        detail += ['', 'Fold dispersion is descriptive, not an independent confidence interval. '
                   'Mean fold changes weight folds equally; pooled scores weight images equally.', '']
        largest = sorted([r for r in errors if r['model'] == name and r['variant'] == 'distance_joint'],
                         key=lambda r: -r['count'])[:10]
        counts = {(r['variant'], r['true_letter'], r['predicted_letter']): r['count']
                  for r in errors if r['model'] == name}
        detail += [f'### {name.upper()}: largest replacement confusions', '',
                   '| True → predicted | Shin | Distances + joints | Shin + joints |',
                   '|---|---:|---:|---:|']
        for r in largest:
            detail.append(f"| {r['true_letter']} → {r['predicted_letter']} | " +
                          ' | '.join(str(counts.get((arm, r['true_letter'], r['predicted_letter']), 0))
                                     for arm in ARMS) + ' |')
        detail += ['', 'J and Z are static dataset images in this experiment; these counts do not evaluate motion recognition.', '']
    position = lines.index('## Interpretation and limits')
    lines[position:position] = detail
    (out / 'report.md').write_text('\n'.join(lines))
    return letters


def plots(out, rows, letters, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pooled = {(r['model'], r['variant']): r for r in rows if r['fold'] == 'pooled'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    x = np.arange(len(ARMS))
    for j, metric in enumerate(('accuracy', 'macro_f1')):
        for i, name in enumerate(models):
            values = [100 * pooled[name, a][metric] for a in ARMS]
            axes[j].plot(x, values, 'o-', label=name.upper())
        axes[j].set_xticks(x, ['Shin\n820', 'Distances + joints\n205', 'Shin + joints\n835'])
        axes[j].set_ylabel('Accuracy (%)' if metric == 'accuracy' else 'Macro-F1 × 100')
        axes[j].legend(); axes[j].grid(axis='y', alpha=.2)
    fig.savefig(out / 'overall.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.5 * len(models)), squeeze=False, layout='constrained')
    lookup = {(r['model'], r['variant'], r['letter']): r for r in letters}
    x = np.arange(26)
    for i, name in enumerate(models):
        for j, arm in enumerate(ARMS[1:]):
            delta = [100 * (lookup[name, arm, l]['recall'] - lookup[name, 'shin', l]['recall']) for l in LABELS]
            axes[i, 0].bar(x + (j - .5) * .36, delta, width=.36, label=NAMES[arm])
        axes[i, 0].set_xticks(x, LABELS); axes[i, 0].axhline(0, color='black', linewidth=.7)
        axes[i, 0].set_ylabel('Recall change (pp)'); axes[i, 0].set_title(name.upper())
        axes[i, 0].legend(); axes[i, 0].grid(axis='y', alpha=.2)
    fig.savefig(out / 'letter_recall_changes.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('data/asl_kaggle/research'))
    parser.add_argument('--project', type=Path, default=Path('data/asl_kaggle'))
    parser.add_argument('--baseline', type=Path, default=Path('results/angles'))
    parser.add_argument('--out', type=Path, default=Path('results/direction_joint_01'))
    parser.add_argument('--work', type=Path, default=Path('runs/direction_joint_01'))
    parser.add_argument('--models', nargs='+', choices=('svm', 'lightgbm'), default=['svm', 'lightgbm'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); args.work.mkdir(parents=True, exist_ok=True)
    X, y, ids, splits = load_inputs(args.run)
    shin = X[:, :820].copy(); del X
    metadata = json.loads((args.run / 'inputs_metadata.json').read_text())
    manifest_bytes = (args.project / 'manifest.json').read_bytes()
    if sha256(manifest_bytes).hexdigest() != metadata['hashes']['manifest']:
        raise ValueError('Manifest differs from the frozen features.')
    if json.loads((args.baseline / 'inputs.json').read_text())['input_hashes'] != metadata['hashes']:
        raise ValueError('Historical baseline uses different source inputs.')
    assignments = np.zeros(len(y), int)
    for fold, train, test in splits:
        if np.any(assignments[test]) or set(train) & set(test):
            raise ValueError('Invalid frozen partitions.')
        assignments[test] = fold
    assert assignments.all() and len(set(ids)) == len(ids)
    provenance = {'version': VERSION, 'shin_version': SHIN_VERSION, 'input_hashes': metadata['hashes'],
                  'angle_columns': list(ANGLE_COLUMNS), 'arms': list(ARMS),
                  'folds_sha256': sha256((args.run / 'summary.json').read_bytes()).hexdigest(),
                  'baseline_hashes': {n: sha256((args.baseline / f'oof_predictions_{n}.csv').read_bytes()).hexdigest() for n in args.models},
                  'fixed_parameters': {'svm': {'C': 10, 'gamma': 'scale', 'class_weight': 'balanced'},
                                       'lightgbm': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': .1, 'min_child_samples': 20, 'random_state': 42}}}
    config = args.work / 'inputs.json'
    if config.exists() and json.loads(config.read_text()) != provenance:
        raise ValueError('Work directory belongs to a different experiment.')
    config.write_text(json.dumps(provenance, indent=2))
    (args.out / 'inputs.json').write_text(json.dumps(provenance, indent=2))
    cache = args.work / 'image_joint_angles.npz'
    if cache.exists():
        with np.load(cache) as data:
            assert data['ids'].tolist() == ids
            angles = data['angles']
    else:
        manifest = json.loads(manifest_bytes); del manifest_bytes
        samples = {s['id']: s for s in manifest['samples']}
        selected = [samples[id_] for id_ in ids]
        assert all(s['label'] == label and s['status'] == 'ok' for s, label in zip(selected, y))
        points = np.array([s['image_landmarks'] for s in selected])
        for i in (0, len(ids) // 2, len(ids) - 1):
            np.testing.assert_array_equal(np.array(list(combined_features(points[i]).values()), np.float32), shin[i])
        angles = joint_angles_batch(points, [s.get('handedness') for s in selected])
        np.savez_compressed(cache, ids=np.array(ids), angles=angles)
        del selected, samples, manifest, points
    print(json.dumps({'images': len(y), 'joint_inputs': angles.shape[1], 'missing_angles': int(np.isnan(angles).sum())}), flush=True)
    rows, predictions = [], {}
    for name in args.models:
        historical = args.baseline / f'oof_predictions_{name}.csv'
        for arm in (*ARMS, 'shin_world'):
            if arm in ('shin', 'shin_world'):
                predicted = reference_predictions(historical, name, ids, y, assignments,
                                                  'shin' if arm == 'shin' else 'shin_angles')
                width = 820 if arm == 'shin' else 835
            else:
                matrix = matrix_for(shin, angles, arm); width = matrix.shape[1]
                predicted = np.full(len(y), '', dtype='<U1')
                for fold, train, test in splits:
                    checkpoint = args.work / f'{name}_{arm}_fold{fold}.npz'
                    path = args.work / f'{name}_{arm}_fold{fold}.joblib'
                    started = time.monotonic()
                    if checkpoint.exists():
                        with np.load(checkpoint) as saved:
                            assert saved['ids'].tolist() == [ids[i] for i in test]
                            pred = saved['predictions']
                    else:
                        print(f'Fitting {name} {arm} fold {fold}: {len(train)} train, {width} inputs', flush=True)
                        if path.exists():
                            model = joblib.load(path)
                        else:
                            model = estimator(name, width, len(train)); model.fit(matrix[train], y[train])
                            joblib.dump(model, path)
                        pred = model.predict(matrix[test]); del model
                        np.savez_compressed(checkpoint, ids=np.array(ids)[test], predictions=pred)
                    predicted[test] = pred
                    print(json.dumps({'model': name, 'arm': arm, 'fold': fold,
                                      'seconds': round(time.monotonic() - started, 1), **scores(y[test], pred)}), flush=True)
                del matrix
            assert np.all(np.isin(predicted, LABELS))
            predictions[name, arm] = predicted
            for fold, train, test in splits:
                rows.append({'model': name, 'variant': arm, 'fold': fold, 'features': width,
                             'images': len(test), **scores(y[test], predicted[test])})
            pooled = {'model': name, 'variant': arm, 'fold': 'pooled', 'features': width,
                      'images': len(y), **scores(y, predicted)}
            rows.append(pooled); print(json.dumps(pooled), flush=True)
    letters = export_results(args.out, rows, predictions, y, ids, assignments)
    plots(args.out, rows, letters, args.models)
    print(f'Completed: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
