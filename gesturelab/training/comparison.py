"""Evaluate the paired feature variants and write comparison reports."""
import base64
import html
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from gesturelab.datasets.data import atomic_json
from gesturelab.training.experiments import metrics, heatmap
from gesturelab.training.experiment_plan import (
    file_hash, load_experiment,
)
from gesturelab.training.rules import UNKNOWN
from gesturelab.features.training_inputs import DEFAULT_VARIANTS, training_input, input_table


def _load_training_results(experiment, split_hash, variants):
    """Reject mismatched splits or models modified after training."""
    training, model_hashes = {}, {}
    for variant in variants:
        folder = experiment / variant
        info = json.loads((folder / 'training.json').read_text())
        if info['split_sha256'] != split_hash:
            raise ValueError('The variants use different splits.')
        actual_hash = file_hash(folder / 'model.joblib')
        if actual_hash != info['model_sha256']:
            raise ValueError('A saved model has changed; results are no longer comparable.')
        training[variant] = info
        model_hashes[variant] = actual_hash
    return training, model_hashes


def _saved_final_test_matches(out, model_hashes, split_hash):
    """Reuse a final test only for the identical frozen model pair."""
    if not (out / 'result.json').exists():
        return False
    previous = json.loads((out / 'result.json').read_text())
    if previous['model_hashes'] != model_hashes or previous['split_sha256'] != split_hash:
        raise ValueError('A final test exists for different models. Use new independent test data.')
    return True


def _final_test_predictions(experiment, plan, records, variants):
    """Keep detection failures in the final test as unknown predictions."""
    if not plan['test_ids']:
        raise ValueError('This experiment has no separate test set.')
    test = [records[sample_id] for sample_id in plan['test_ids']]
    predictions = {}
    for variant in variants:
        table = input_table(test, plan['feature_config'], variant)
        bundle = joblib.load(experiment / variant / 'model.joblib')
        valid = {}
        if not table.empty:
            predicted = bundle['estimator'].predict(table[bundle['config']['columns']].to_numpy())
            valid = dict(zip(table.id, map(str, predicted)))
        predictions[variant] = pd.DataFrame([
            {'id': sample['id'], 'label': sample['label'],
             'prediction': valid.get(sample['id'], UNKNOWN),
             'condition': sample['condition'], 'session': sample['session'],
             'participant': sample['participant']}
            for sample in test
        ])
    return predictions


def _cv_predictions(experiment, variants):
    return {
        variant: pd.read_csv(
            experiment / variant / 'cv_predictions.csv',
            dtype={'label': str, 'prediction': str, 'id': str},
        )
        for variant in variants
    }


def _pair_predictions(predictions):
    """Match unique photo IDs and ground truth before writing any report."""
    if len(predictions) != 2:
        raise ValueError("Choose exactly two distinct training input variants.")
    first, second = predictions.values()
    if set(first.id) != set(second.id) or any(
        table.id.duplicated().any() for table in predictions.values()
    ):
        raise ValueError('The two results do not have exactly the same unique photo IDs.')
    paired = first[['id', 'label', 'prediction']].rename(columns={'prediction': 'first'}).merge(
        second[['id', 'label', 'prediction']].rename(columns={'prediction': 'second'}),
        on=['id', 'label'], validate='one_to_one',
    )
    if len(paired) != len(first):
        raise ValueError('Ground-truth labels differ between the two results.')
    paired['first_correct'] = paired['label'] == paired['first']
    paired['second_correct'] = paired['label'] == paired['second']
    return paired


def _report_note(plan, final_test):
    note = ('Separate final test; detection failures included' if final_test
            else 'Development CV; pooled validation predictions')
    return 'SYNTHETIC SOFTWARE TEST · ' + note if plan['synthetic'] else note


def _write_variant_report(out, variant, predictions, labels, note):
    """Write per-photo, per-label and confusion results for one variant."""
    predictions.to_csv(out / (variant + '_predictions.csv'), index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.label, predictions.prediction, labels=labels, zero_division=0,
    )
    per_label = pd.DataFrame({
        'label': labels, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support,
    })
    per_label.to_csv(out / (variant + '_per_label.csv'), index=False)
    order = labels + [UNKNOWN]
    confusion = pd.DataFrame(
        confusion_matrix(predictions.label, predictions.prediction, labels=order)[:len(labels)],
        index=labels, columns=order,
    )
    confusion.to_csv(out / (variant + '_confusion.csv'))
    heatmap(
        confusion.div(confusion.sum(axis=1).replace(0, 1), axis=0),
        out / (variant + '_confusion.png'),
        training_input(variant).title + ' · ' + note + '\nRows actual; columns predicted',
    )
    return per_label.set_index('label').recall.rename(training_input(variant).title)


def plot_comparison(table, out, note):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(table))
    for ax, key, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        ax.bar(x, table[key], color=['#397cd5', '#20a28c'])
        ax.set_xticks(x, [training_input(variant).title.replace(' finger ', '\nfinger ')
                         for variant in table.variant])
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.set_ylabel('Score (0–1)')
        for index, value in enumerate(table[key]):
            ax.text(index, value + .025, f'{value:.3f}', ha='center')
    fig.suptitle(note, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


def _write_scores_and_charts(out, predictions, training, plan, note):
    rows, recalls = [], []
    for variant, predicted in predictions.items():
        info = training[variant]
        rows.append({
            'variant': variant, 'n_features': info['n_features'],
            **metrics(predicted.label, predicted.prediction, plan['labels']),
            'cv_macro_f1_mean': info['cv_macro_f1_mean'],
            'cv_macro_f1_std': info['cv_macro_f1_std'],
        })
        recalls.append(_write_variant_report(out, variant, predicted, plan['labels'], note))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'metrics.csv', index=False)
    plot_comparison(summary, out, note)
    heatmap(pd.concat(recalls, axis=1), out / 'recall_per_letter.png', 'Recall per letter · ' + note)
    return summary


def _comparison_result(stage, split_hash, model_hashes, plan, summary, paired, note):
    result = {
        'stage': stage, 'variants': summary.variant.tolist(),
        'split_sha256': split_hash, 'model_hashes': model_hashes,
        'synthetic': plan['synthetic'], 'n': len(paired),
        'macro_f1_difference_second_minus_first': float(summary.iloc[1].macro_f1 - summary.iloc[0].macro_f1),
        'only_second_correct': int((paired.second_correct & ~paired.first_correct).sum()),
        'only_first_correct': int((paired.first_correct & ~paired.second_correct).sum()),
        'both_wrong': int((~paired.first_correct & ~paired.second_correct).sum()),
        'excluded_development_detections': len(plan['excluded_development_ids']),
        'interpretation': note + '. No claim of statistical significance.',
    }
    if tuple(summary.variant) == DEFAULT_VARIANTS:
        # Preserve established field names for the original two-variant experiment.
        result.update({
            'macro_f1_difference_with_minus_without': result['macro_f1_difference_second_minus_first'],
            'only_with_correct': result['only_second_correct'],
            'only_without_correct': result['only_first_correct'],
        })
    return result


def _write_html_report(out, summary, result, note):
    images = []
    for name in ['comparison.png', 'recall_per_letter.png', *[v + '_confusion.png' for v in summary.variant]]:
        encoded = base64.b64encode((out / name).read_bytes()).decode()
        images.append('<img style="max-width:100%" src="data:image/png;base64,' + encoded + '">')
    html_text = '<html lang="en"><meta charset="utf-8"><title>Training input comparison</title><body style="max-width:1100px;margin:40px auto;font-family:Arial">'
    html_text += '<h1>Training input comparison</h1><p>' + html.escape(note) + '</p>' + summary.to_html(index=False)
    html_text += '<p>Both variants use the same labels, photos, splits and model settings. Their input features are defined in training_inputs.py.</p>'
    html_text += '<pre>' + html.escape(json.dumps(result, indent=2, ensure_ascii=False)) + '</pre>' + ''.join(images) + '</body></html>'
    (out / 'report.html').write_text(html_text, encoding='utf-8')


def _validated_variants(variants):
    variants = tuple(variants)
    if len(variants) != 2 or len(set(variants)) != 2:
        raise ValueError('Choose exactly two distinct training input variants.')
    for variant in variants:
        training_input(variant)
    return variants


def compare_variants(experiment, final_test=False, project_override=None, variants=DEFAULT_VARIANTS):
    """Validate the paired experiment, evaluate it and write a comparison report."""
    variants = _validated_variants(variants)
    experiment, _, plan, records = load_experiment(experiment, project_override)
    split_hash = file_hash(experiment / 'split.json')
    training, model_hashes = _load_training_results(experiment, split_hash, variants)
    stage = 'final_test' if final_test else 'comparison_cv'
    folder = stage if variants == DEFAULT_VARIANTS else stage + '_' + '_vs_'.join(variants)
    out = experiment / folder
    if final_test and _saved_final_test_matches(out, model_hashes, split_hash):
        return out

    predictions = (_final_test_predictions(experiment, plan, records, variants) if final_test
                   else _cv_predictions(experiment, variants))
    paired = _pair_predictions(predictions)
    note = _report_note(plan, final_test)
    out.mkdir(exist_ok=True)
    summary = _write_scores_and_charts(out, predictions, training, plan, note)
    exported_pairs = paired.rename(columns={
        'first': 'without', 'second': 'with',
        'first_correct': 'without_correct', 'second_correct': 'with_correct',
    }) if variants == DEFAULT_VARIANTS else paired
    exported_pairs.to_csv(out / 'paired_predictions.csv', index=False)
    result = _comparison_result(stage, split_hash, model_hashes, plan, summary, paired, note)
    atomic_json(out / 'result.json', result)
    _write_html_report(out, summary, result, note)
    return out
