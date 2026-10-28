"""Independent integer-count metric implementation: no sklearn/reporting helpers."""
from collections import Counter
import csv
import math
from pathlib import Path
import statistics

UNKNOWN = '__unknown__'


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def score_rows(rows, labels, supported_macro=False, prediction_labels=None):
    """CV macro includes every frozen class; incomplete final-test macro uses truth-present classes."""
    if not rows or not labels or len(set(labels)) != len(labels) or UNKNOWN in labels:
        raise ValueError('Need nonempty predictions and a distinct, explicit class list.')
    if any(r['label'] not in labels for r in rows):
        raise ValueError('Ground truth contains a class outside the frozen labels.')
    prediction_labels = list(prediction_labels or labels)
    if not set(labels) <= set(prediction_labels) or UNKNOWN in prediction_labels:
        raise ValueError('Prediction classes must include the evaluation classes and exclude the rejection token.')
    order = prediction_labels + [UNKNOWN]
    counts = Counter((r['label'], r['prediction'] if r['prediction'] in prediction_labels else UNKNOWN) for r in rows)
    per_class = [_class_counts(label, labels, counts) for label in labels]
    macro = [r for r in per_class if r['support']] if supported_macro else per_class
    accuracy = sum(counts[label, label] for label in labels) / len(rows)
    return dict(accuracy=accuracy, macro_f1=statistics.mean(r['f1'] for r in macro),
                balanced_accuracy=statistics.mean(r['recall'] for r in macro),
                coverage=sum(r['prediction'] in prediction_labels for r in rows) / len(rows), n=len(rows),
                labels=list(labels), predicted_columns=order, per_class=per_class,
                confusion=[[counts[label, column] for column in order] for label in labels])


def _class_counts(label, labels, counts):
    tp = counts[label, label]
    fp = sum(counts[other, label] for other in labels if other != label)
    fn = sum(value for (truth, prediction), value in counts.items() if truth == label and prediction != label)
    return dict(label=label, tp=tp, fp=fp, fn=fn, support=tp + fn,
                precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn), f1=_ratio(2 * tp, 2 * tp + fp + fn))


def fold_summary(rows, labels):
    """Compute unweighted fold means and sample SD (ddof=1), separately from pooling."""
    folds = sorted({int(r['fold']) for r in rows})
    scores = {str(f): score_rows([r for r in rows if int(r['fold']) == f], labels) for f in folds}
    summary = {}
    for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
        values = [s[key] for s in scores.values()]
        summary[key + '_mean'] = statistics.mean(values)
        summary[key + '_std'] = statistics.stdev(values) if len(values) > 1 else None
    return dict(folds=scores, summary=summary)


def numeric_difference(expected, observed, tolerance=1e-9, undefined=False):
    if expected is None or undefined:
        return None if observed in ('', None, 'nan', 'NaN') else dict(expected=None, observed=observed)
    try:
        number = float(observed)
    except (TypeError, ValueError):
        return dict(expected=expected, observed=observed)
    if not math.isfinite(number):
        return dict(expected=expected, observed=str(number), absolute_difference=None)
    if abs(number - expected) > tolerance:
        return dict(expected=expected, observed=number, absolute_difference=abs(number - expected))
    return None
