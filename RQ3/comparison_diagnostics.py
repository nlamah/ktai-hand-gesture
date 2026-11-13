"""Describe existing held-out feature comparisons without fitting new models."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from baseline.runner import ROOT
from research.data import atomic_json
from research.inputs import write_csv


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def scores(truth, prediction, labels):
    precision, _, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0)
    return precision, f1


def diagnose(frame):
    """Pair source IDs, truth and folds before counting corrected/new errors."""
    if frame.duplicated(['model', 'variant', 'id']).any():
        raise ValueError('Duplicate held-out predictions.')
    folds, letters, summary = [], [], []
    shared_sources = None
    for model, model_rows in frame.groupby('model', sort=True):
        reference = model_rows[model_rows.variant == 'DS'].sort_values('id').reset_index(drop=True)
        if reference.empty:
            raise ValueError(f'Missing DS reference: {model}')
        sources = reference[['id', 'label', 'group', 'fold']]
        if shared_sources is not None and not sources.equals(shared_sources):
            raise ValueError('Classifiers do not share sources, labels and folds.')
        shared_sources = sources
        labels = sorted(reference.label.unique())
        truth = reference.label.to_numpy()
        baseline = reference.prediction.to_numpy()
        baseline_correct = baseline == truth
        base_precision, base_f1 = scores(truth, baseline, labels)
        for variant in sorted(set(model_rows.variant) - {'DS'}):
            candidate = model_rows[model_rows.variant == variant].sort_values('id').reset_index(drop=True)
            if not candidate[['id', 'label', 'group', 'fold']].equals(sources):
                raise ValueError(f'Unpaired sources, labels or folds: {model}/{variant}')
            prediction = candidate.prediction.to_numpy()
            correct = prediction == truth
            rows = []
            for fold in ['pooled', *sorted(reference.fold.unique())]:
                keep = np.ones(len(truth), dtype=bool) if fold == 'pooled' else reference.fold.eq(fold).to_numpy()
                bp, bf = scores(truth[keep], baseline[keep], labels)
                cp, cf = scores(truth[keep], prediction[keep], labels)
                fixed = int((~baseline_correct[keep] & correct[keep]).sum())
                lost = int((baseline_correct[keep] & ~correct[keep]).sum())
                rows.append(dict(model=model, variant=variant, fold=str(fold),
                                 images=int(keep.sum()), fixed=fixed, lost=lost,
                                 net_correct=fixed-lost,
                                 accuracy_delta_pp=100*(fixed-lost)/int(keep.sum()),
                                 precision_delta_pp=100*float(cp.mean()-bp.mean()),
                                 f1_delta_pp=100*float(cf.mean()-bf.mean())))
            cp, cf = scores(truth, prediction, labels)
            per_letter = []
            for i, letter in enumerate(labels):
                keep = truth == letter
                fixed = int((~baseline_correct[keep] & correct[keep]).sum())
                lost = int((baseline_correct[keep] & ~correct[keep]).sum())
                per_letter.append(dict(model=model, variant=variant, letter=letter,
                                       images=int(keep.sum()), fixed=fixed, lost=lost,
                                       net_correct=fixed-lost,
                                       precision_delta_pp=100*float(cp[i]-base_precision[i]),
                                       f1_delta_pp=100*float(cf[i]-base_f1[i])))
            if sum(r['net_correct'] for r in per_letter) != rows[0]['net_correct']:
                raise ValueError('Letter counts do not sum to the pooled contrast.')
            detail = rows[1:]
            if any(sum(r[key] for r in detail) != rows[0][key] for key in ['images', 'fixed', 'lost', 'net_correct']):
                raise ValueError('Fold counts do not sum to the pooled contrast.')
            summary.append(dict(rows[0], folds=len(detail),
                                f1_positive_folds=sum(r['f1_delta_pp'] > 0 for r in detail),
                                net_positive_folds=sum(r['net_correct'] > 0 for r in detail),
                                fold_f1_min_pp=min(r['f1_delta_pp'] for r in detail),
                                fold_f1_max_pp=max(r['f1_delta_pp'] for r in detail),
                                net_by_fold={r['fold']: r['net_correct'] for r in detail},
                                largest_gains=[{'letter': r['letter'], 'net_correct': r['net_correct']}
                                               for r in sorted(per_letter, key=lambda r: (-r['net_correct'], r['letter']))
                                               if r['net_correct'] > 0][:3],
                                largest_losses=[{'letter': r['letter'], 'net_correct': r['net_correct']}
                                                for r in sorted(per_letter, key=lambda r: (r['net_correct'], r['letter']))
                                                if r['net_correct'] < 0][:3]))
            folds.extend(rows)
            letters.extend(per_letter)
    return folds, letters, summary


def run(source, out, verify=False):
    frame = pd.read_csv(source / 'predictions.csv', keep_default_na=False, dtype={'fold': str})
    folds, letters, comparisons = diagnose(frame)
    saved = pd.read_csv(source / 'metrics.csv', keep_default_na=False, dtype={'fold': str})
    paired = pd.read_csv(source / 'paired_changes.csv', keep_default_na=False, dtype={'fold': str})
    for row in folds:
        arm = saved[(saved.model == row['model']) & (saved.variant == row['variant']) & (saved.fold == row['fold'])]
        base = saved[(saved.model == row['model']) & (saved.variant == 'DS') & (saved.fold == row['fold'])]
        pair = paired[(paired.model == row['model']) & (paired.variant == row['variant']) & (paired.fold == row['fold'])]
        if len(arm) != 1 or len(base) != 1 or len(pair) != 1:
            raise ValueError('Missing or duplicate saved comparison rows.')
        a, b, c = arm.iloc[0], base.iloc[0], pair.iloc[0]
        np.testing.assert_array_equal([row['images'], row['fixed'], row['lost'], row['net_correct']],
                                      [a.images, c.fixed, c.lost, c.net])
        np.testing.assert_allclose([row['precision_delta_pp'], row['f1_delta_pp'], row['accuracy_delta_pp']],
                                   100*np.array([a.macro_precision-b.macro_precision,
                                                 a.macro_f1-b.macro_f1, a.accuracy-b.accuracy]), atol=1e-11, rtol=0)
    summary = {'version': 'paired-feature-diagnostics-v1',
               'inputs_sha256': {name: digest(source / name) for name in
                                 ['predictions.csv', 'metrics.csv', 'paired_changes.csv']},
               'source_sha256': digest(__file__), 'comparisons': comparisons,
               'interpretation': 'Descriptive paired differences across the existing folds and letters. '
                                 'Folds share training data and filename groups are not verified people; '
                                 'these ranges are not confidence intervals or significance tests.',
               'new_model_fits': 0}
    expected = {'fold_changes.csv': folds, 'letter_changes.csv': letters}
    if verify:
        if json.loads((out / 'summary.json').read_text()) != summary:
            raise ValueError('Diagnostic summary differs from current inputs/code.')
        for name, rows in expected.items():
            stored = pd.read_csv(out / name, keep_default_na=False, dtype={'fold': str})
            pd.testing.assert_frame_equal(stored, pd.DataFrame(rows), check_exact=False, atol=1e-11, rtol=0)
    else:
        if (out / 'summary.json').exists() and json.loads((out / 'summary.json').read_text()) != summary:
            raise ValueError('Different diagnostic inputs/code; choose a new --out directory.')
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in expected.items():
            write_csv(out / name, rows)
        atomic_json(out / 'summary.json', summary)
    return {'status': 'passed', 'comparisons': len(comparisons), 'fold_rows': len(folds),
            'letter_rows': len(letters), 'new_model_fits': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/feature_factorial_01')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/feature_diagnostics_01')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, args.verify), indent=2))


if __name__ == '__main__':
    main()
