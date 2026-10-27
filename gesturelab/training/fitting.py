"""Cross-validation and final model fitting for each feature variant."""
import time

import joblib
import pandas as pd

from gesturelab.datasets.data import atomic_json, fingerprint
from gesturelab.features.training_inputs import training_input, input_table, model_input_metadata
from gesturelab.training.experiments import estimator, metrics
from gesturelab.training.classifiers import classifier_parameters
from gesturelab.training.experiment_plan import (
    file_hash, load_experiment,
)

def _training_data(plan, records, variant):
    """Build development inputs only, using the frozen feature settings."""
    samples = [records[sample_id] for sample_id in plan['development_ids']]
    training = input_table(samples, plan['feature_config'], variant).set_index('id', drop=False)
    columns = training_input(variant).columns(training.columns)
    return training, columns


def _estimator_for(plan, variant, n_train):
    return estimator(plan['model'], n_train, plan['seed'], variant=variant,
                     profile=plan.get('classifier_profile', 'project'))


def _fold_predictions(validation, predicted, index):
    return [{
        'id': row.id, 'fold': index, 'label': row.label, 'prediction': str(prediction),
        'condition': row.condition, 'session': row.session, 'participant': row.participant,
    } for row, prediction in zip(validation.itertuples(), predicted)]


def _cross_validate(training, columns, plan, variant):
    """Fit a fresh classifier per frozen fold and collect held-out predictions."""
    scores = []
    predictions = []
    for index, fold in enumerate(plan['folds']):
        train = training.loc[fold['train_ids']]
        validation = training.loc[fold['validation_ids']]
        model = _estimator_for(plan, variant, len(train))

        start = time.perf_counter()
        model.fit(train[columns].to_numpy(), train.label.to_numpy())
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predicted = model.predict(validation[columns].to_numpy())
        milliseconds = 1000 * (time.perf_counter() - start) / len(validation)
        scores.append({
            'fold': index,
            **metrics(validation.label, predicted, plan['labels']),
            'fit_seconds': fit_seconds,
            'classifier_ms_per_image': milliseconds,
        })
        predictions.extend(_fold_predictions(validation, predicted, index))
    return pd.DataFrame(scores), pd.DataFrame(predictions)


def _fit_final_model(training, columns, plan, variant):
    """Refit the selected classifier on all development photos, without test data."""
    model = _estimator_for(plan, variant, len(training))
    model.fit(training[columns].to_numpy(), training.label.to_numpy())
    return model


def _fit_models(training, columns, plan, variant):
    start = time.perf_counter()
    fold_scores, predictions = _cross_validate(training, columns, plan, variant)
    final_start = time.perf_counter()
    model = _fit_final_model(training, columns, plan, variant)
    times = {'final_fit_seconds': time.perf_counter() - final_start,
             'model_training_seconds': time.perf_counter() - start}
    return model, fold_scores, predictions, times


def _model_bundle(model, columns, plan, variant, split_hash):
    """Package everything needed to reproduce the training inputs at prediction."""
    return {
        'estimator': model,
        'config': {'model': plan['model'], 'columns': columns, 'feature_set': variant,
                   'classifier_profile': plan.get('classifier_profile', 'project'),
                   'parameters': classifier_parameters(plan['model'], variant,
                       plan.get('classifier_profile', 'project'), len(plan['development_ids']), plan['seed'])},
        'labels': plan['labels'],
        'candidate': variant, 'synthetic': plan['synthetic'],
        'split_sha256': split_hash,
        **model_input_metadata(variant, plan['feature_config'], plan['code_hashes']),
    }


def _training_summary(out, training, bundle, fold_scores, predictions, plan, times):
    columns = bundle['config']['columns']
    return {
        'variant': bundle['candidate'], 'n_features': len(columns),
        'feature_columns': columns, 'n_train': len(training),
        'split_sha256': bundle['split_sha256'],
        'model_sha256': file_hash(out / 'model.joblib'),
        'model': plan['model'], 'parameters': str(bundle['estimator']),
        'classifier_profile': plan.get('classifier_profile', 'project'),
        'resolved_parameters': bundle['config']['parameters'],
        'synthetic': plan['synthetic'],
        'cv_accuracy_mean': float(fold_scores.accuracy.mean()),
        'cv_accuracy_std': float(fold_scores.accuracy.std(ddof=1)),
        'cv_macro_f1_mean': float(fold_scores.macro_f1.mean()),
        'cv_macro_f1_std': float(fold_scores.macro_f1.std(ddof=1)),
        'cv_pooled': metrics(predictions.label, predictions.prediction, plan['labels']),
        'cv_fit_seconds_total': float(fold_scores.fit_seconds.sum()), **times,
        'note': 'Development/CV only. The final model was trained on all development photos. Test photos have not been predicted yet.',
    }


def _save_training_results(out, training, bundle, fold_scores, predictions, plan, times):
    """Persist model, inputs, fold scores and predictions without touching older runs."""
    out.mkdir()
    joblib.dump(bundle, out / 'model.joblib')
    training[bundle['config']['columns']].to_csv(out / 'training_features.csv', index=True)
    fold_scores.to_csv(out / 'cv_folds.csv', index=False)
    predictions.to_csv(out / 'cv_predictions.csv', index=False)
    atomic_json(out / 'training.json', _training_summary(out, training, bundle, fold_scores, predictions, plan, times))


def train_variant(experiment, variant, project_override=None):
    """Train one feature variant against a frozen, shared experiment plan.

    Cross-validation and final fitting use development photos only. Results
    are saved only after confirming that the dataset did not change mid-run.
    """
    experiment, project, plan, records = load_experiment(experiment, project_override)
    training_input(variant)
    out = experiment / variant
    if out.exists():
        raise ValueError('This variant has already run. Use a new experiment directory to retrain.')

    start = time.perf_counter()
    training, columns = _training_data(plan, records, variant)
    feature_seconds = time.perf_counter() - start
    model, fold_scores, predictions, times = _fit_models(training, columns, plan, variant)
    times['feature_extraction_seconds'] = feature_seconds
    bundle = _model_bundle(model, columns, plan, variant, file_hash(experiment / 'split.json'))

    if fingerprint(project) != plan['dataset_fingerprint']:
        raise ValueError('The dataset changed during training. Stop recording and start a new experiment.')
    _save_training_results(out, training, bundle, fold_scores, predictions, plan, times)
    return out


