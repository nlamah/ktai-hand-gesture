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
from gesturelab.training.predictions import (
    pair_predictions as _pair_predictions, align_predictions, validate_predictions, pairwise_scores,
)
from gesturelab.training.reporting import summary_html


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


def _report_note(plan, final_test):
    note = ('Separate final test; detection failures included' if final_test
            else 'Development CV; pooled validation predictions')
    if final_test and plan.get('missing_test_labels'):
        note += '. No test photos for: ' + ', '.join(plan['missing_test_labels'])
        note += '; macro scores cover only letters present in the test set'
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
    per_label.loc[per_label.support == 0, ['recall', 'f1']] = np.nan
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


def plot_comparison(table, out, note, final_test=False):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, max(4, 1.5 + len(table) * .6)))
    x = np.arange(len(table))
    for ax, key, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        values = table[key] if final_test else table['cv_' + key + '_mean']
        errors = None if final_test else table['cv_' + key + '_std']
        ax.barh(x, values, xerr=errors, capsize=3, color='#397cd5')
        ax.set_yticks(x, [training_input(variant).title for variant in table.variant])
        ax.set_ylim(len(table) - .5, -.5)
        ax.set_xlim(0, 1.1)
        ax.set_title(title)
        ax.set_xlabel('Score (0–1)' if final_test else 'Fold mean ± SD')
        for index, value in enumerate(values):
            ax.text(value + .015, index, f'{value:.3f}', va='center')
    fig.suptitle(note, fontsize=10)
    if not final_test:
        fig.text(.5, .015, 'Error bars show variability across folds, not a confidence interval.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(out / 'comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def _prediction_scores(predictions, plan, final_test):
    scores = metrics(predictions.label, predictions.prediction, plan['labels'])
    if final_test and plan.get('missing_test_labels'):
        present = sorted(set(predictions.label))
        supported = metrics(predictions.label, predictions.prediction, present)
        for key in ('macro_f1', 'balanced_accuracy'):
            scores[key] = supported[key]
    return scores


def _write_scores_and_charts(out, predictions, training, plan, note, final_test=False):
    rows, recalls = [], []
    for variant, predicted in predictions.items():
        info = training[variant]
        rows.append({
            'variant': variant, 'n_features': info['n_features'],
            'index_middle_tip_weight': info.get('index_middle_tip_weight', 1.0),
            'mirror_training': info.get('mirror_training', False),
            'n_training_rows': info.get('n_training_rows', info['n_train']),
            **_prediction_scores(predicted, plan, final_test),
            'cv_macro_f1_mean': info['cv_macro_f1_mean'],
            'cv_macro_f1_std': info['cv_macro_f1_std'],
            **{key: info[key] for key in ('cv_accuracy_mean', 'cv_accuracy_std', 'n_train',
                'cv_fit_seconds_total', 'final_fit_seconds', 'model_training_seconds', 'feature_extraction_seconds')},
        })
        recalls.append(_write_variant_report(out, variant, predicted, plan['labels'], note))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'metrics.csv', index=False)
    plot_comparison(summary, out, note, final_test)
    heatmap(pd.concat(recalls, axis=1), out / 'recall_per_letter.png', 'Recall per letter · ' + note)
    return summary


def _comparison_result(stage, split_hash, model_hashes, plan, summary, paired, note):
    result = {
        'stage': stage, 'variants': summary.variant.tolist(),
        'split_sha256': split_hash, 'model_hashes': model_hashes,
        'synthetic': plan['synthetic'], 'n': len(paired),
        'evaluation_note': plan.get('evaluation_note', ''),
        'missing_test_labels': plan.get('missing_test_labels', []) if stage == 'final_test' else [],
        'classifier': plan['model'], 'classifier_profile': plan.get('classifier_profile', 'project'),
        'index_middle_tip_weight': plan.get('index_middle_tip_weight', 1.0),
        'mirror_training': plan.get('mirror_training', False),
        'augmentation_definition': plan.get('augmentation_definition'),
        'excluded_development_detections': len(plan['excluded_development_ids']),
        'interpretation': note + '. No claim of statistical significance.',
    }
    if len(summary) == 2:
        result.update({
            'macro_f1_difference_second_minus_first': float(summary.iloc[1].macro_f1 - summary.iloc[0].macro_f1),
            'only_second_correct': int((paired.second_correct & ~paired.first_correct).sum()),
            'only_first_correct': int((paired.first_correct & ~paired.second_correct).sum()),
            'both_wrong': int((~paired.first_correct & ~paired.second_correct).sum()),
        })
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
    html_text += '<h1>Training input comparison</h1><p>' + html.escape(note) + '</p>' + summary_html(summary)
    if result.get('evaluation_note'):
        html_text += '<p>' + html.escape(result['evaluation_note']) + '</p>'
    html_text += '<p>All variants use the same labels, photo IDs and grouped folds. The classifier profile and resolved settings are saved with each model. Paper presets may use different hyperparameters per input.</p>'
    html_text += '<p>The table accuracy/macro-F1 pool predictions. Charts show development fold means and standard deviations; final-test charts show test scores without CV error bars. CV includes successful detections only. See detection_counts.csv for failures.</p>'
    html_text += '<p>Downloadable files beside this report: metrics.csv, cv_folds.csv (development only), pairwise_comparisons.csv, paired_predictions.csv, detection_counts.csv, and per-variant predictions, per-letter scores and confusion matrices.</p>'
    html_text += '<pre>' + html.escape(json.dumps(result, indent=2, ensure_ascii=False)) + '</pre>' + ''.join(images) + '</body></html>'
    (out / 'report.html').write_text(html_text, encoding='utf-8')


def _validated_variants(variants):
    variants = tuple(variants)
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError('Choose at least two distinct training input variants.')
    for variant in variants:
        training_input(variant)
    return variants


def _detection_report(out, plan, records):
    ids = plan['development_ids'] + plan['excluded_development_ids'] + plan['test_ids']
    data = pd.DataFrame([{'role': records[i]['role'], 'label': records[i]['label'],
                          'status': records[i]['status']} for i in ids])
    counts = data.groupby(['role', 'label', 'status']).size().rename('n').reset_index()
    counts.to_csv(out / 'detection_counts.csv', index=False)
    return counts.groupby(['role', 'status']).n.sum().reset_index().to_dict('records')


def _write_fold_report(experiment, out, variants):
    tables = [pd.read_csv(experiment / v / 'cv_folds.csv').assign(variant=v) for v in variants]
    pd.concat(tables, ignore_index=True).to_csv(out / 'cv_folds.csv', index=False)


def _write_comparison(experiment, out, plan, records, predictions, training, model_hashes, split_hash, final_test):
    paired = align_predictions(predictions)
    note = _report_note(plan, final_test)
    out.mkdir(exist_ok=True)
    summary = _write_scores_and_charts(out, predictions, training, plan, note, final_test)
    exported = paired.rename(columns={'first': 'without', 'second': 'with',
        'first_correct': 'without_correct', 'second_correct': 'with_correct'}) if tuple(predictions) == DEFAULT_VARIANTS else paired
    exported.to_csv(out / 'paired_predictions.csv', index=False)
    stage = 'final_test' if final_test else 'comparison_cv'
    result = _comparison_result(stage, split_hash, model_hashes, plan, summary, paired, note)
    pairs = pairwise_scores(predictions, summary)
    pairs.to_csv(out / 'pairwise_comparisons.csv', index=False)
    result['pairwise_comparisons'] = pairs.to_dict('records')
    result['detection_counts'] = _detection_report(out, plan, records)
    if not final_test:
        _write_fold_report(experiment, out, predictions)
    atomic_json(out / 'result.json', result)
    _write_html_report(out, summary, result, note)
    return out


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
    validate_predictions(predictions, plan, records, final_test)
    return _write_comparison(experiment, out, plan, records, predictions, training, model_hashes, split_hash, final_test)
