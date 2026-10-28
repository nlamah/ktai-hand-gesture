"""Inspect actual trusted local models and preserve their identity without retraining."""
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.svm import SVC

from gesturelab.features.training_inputs import model_features, validate_model_input
from gesturelab.training.experiments import predict_bundle
from gesturelab.validation.report import read_json, sha256, write_json


def new_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing file is preserved: {path}')
    write_json(path, value)
    return path


def _classifier(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, 'steps') else estimator


def _parameters(estimator):
    return {key: value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)
            for key, value in estimator.get_params(deep=False).items()}


def _provenance(bundle, plan_path):
    if not plan_path.is_file():
        return {}, None
    plan = read_json(plan_path)
    digest = sha256(plan_path)
    if bundle.get('split_sha256') and bundle['split_sha256'] != digest:
        raise ValueError('Selected experiment split does not belong to this saved model.')
    if set(plan['labels']) != set(map(str, _classifier(bundle['estimator']).classes_)):
        raise ValueError('Experiment classes differ from the actual fitted model.')
    return plan, digest


def _missing_metadata(bundle, plan, alphabet):
    result = ['Historical detector asset/options are not stored in the model.']
    if not alphabet: result.append('Alphabet unknown; class names do not identify sign language.')
    if not plan: result.append('Source experiment unavailable; training membership cannot be verified.')
    if not bundle.get('feature_code_hashes'): result.append('Historical feature code hashes unavailable.')
    if not bundle.get('split_sha256'): result.append('Saved model-to-split hash unavailable; selected plan association is unverified.')
    return result


def inspect_model(model, experiment=None, alphabet=None, name=None):
    model = Path(model).resolve()
    bundle = joblib.load(model)
    classifier = _classifier(bundle['estimator'])
    experiment = Path(experiment).resolve() if experiment else model.parent.parent
    plan_path = experiment / 'split.json'
    plan, split_digest = _provenance(bundle, plan_path)
    labels = list(map(str, getattr(classifier, 'classes_', bundle.get('labels', []))))
    columns = bundle.get('config', {}).get('columns', [])
    try:
        validate_model_input(bundle)
        compatible, error = True, None
    except (ValueError, KeyError) as problem:
        compatible, error = False, str(problem)
    variant = bundle.get('training_input', bundle.get('config', {}).get('feature_set'))
    info = dict(name=name or model.parent.name, model_file=str(model), model_sha256=sha256(model),
        experiment=str(experiment), split_sha256=split_digest,
        dataset_fingerprint=plan.get('dataset_fingerprint'), code_hashes=plan.get('code_hashes'),
        training_environment={key: plan.get(key) for key in ('seed', 'python_version', 'sklearn_version', 'numpy_version', 'pandas_version', 'classifier_versions')},
        input_code_hashes=bundle.get('feature_code_hashes'), classifier=type(classifier).__name__,
        input_definition=bundle.get('input_definition'), feature_config=bundle.get('feature_config'),
        input_preprocessing=bundle.get('input_preprocessing'),
        parameters=_parameters(classifier), variant=variant, features=len(columns), columns=columns,
        preprocessing=[dict(name=n, type=type(step).__name__, parameters=_parameters(step)) for n, step in getattr(bundle['estimator'], 'steps', [])[:-1]],
        labels=labels, alphabet=alphabet, alphabet_evidence='Researcher declaration; class names alone do not establish sign language.',
        current_input_compatible=compatible, compatibility_error=error,
        baseline_svm_shin=isinstance(classifier, SVC) and variant == 'shin2021_combined' and len(columns) == 820,
        feature_version=bundle.get('feature_version'), synthetic=bundle.get('synthetic', False), historical_detector_requirements=None,
        missing_metadata=_missing_metadata(bundle, plan, alphabet))
    return info


def register_model(model, out, experiment=None, alphabet=None, name=None, require_baseline=False):
    info = inspect_model(model, experiment, alphabet, name)
    if require_baseline and not info['baseline_svm_shin']:
        raise ValueError('The actual classifier/input is not SVM + combined 820-value Shin features.')
    return new_json(out, info)


@dataclass
class ModelHandle:
    metadata: dict
    bundle: dict

    @classmethod
    def load(cls, registration):
        metadata = read_json(registration) if isinstance(registration, (str, Path)) else registration
        if sha256(metadata['model_file']) != metadata['model_sha256']:
            raise ValueError('Selected model bytes changed after registration.')
        bundle = joblib.load(metadata['model_file'])
        validate_model_input(bundle)
        if (metadata['columns'] != bundle['config']['columns'] or metadata['features'] != bundle['estimator'].n_features_in_
                or metadata['classifier'] != type(_classifier(bundle['estimator'])).__name__):
            raise ValueError('Registered input/classifier schema differs from the actual model.')
        if list(map(str, _classifier(bundle['estimator']).classes_)) != metadata['labels']:
            raise ValueError('Registered classes differ from the actual model.')
        return cls(metadata, bundle)

    def predict(self, detection, width, height):
        values = model_features(self.bundle, {**detection, 'width': width, 'height': height})
        return str(predict_bundle(self.bundle, [values])[0])


def compatible_models(first, second):
    a, b = first.metadata, second.metadata
    if not a.get('alphabet') or a['alphabet'] != b.get('alphabet'):
        raise ValueError('A/B requires the same explicitly registered alphabet.')
    if a['labels'] != b['labels'] or a['feature_version'] != b['feature_version']:
        raise ValueError('A/B requires compatible class lists and landmark feature versions.')
    if (a.get('historical_detector_requirements') and b.get('historical_detector_requirements')
            and a['historical_detector_requirements'] != b['historical_detector_requirements']):
        raise ValueError('Recorded detector requirements differ; no implicit conversion is allowed.')
    return dict(alphabet=a['alphabet'], labels=a['labels'],
                limitation='A shared current detector is used. Historical detector requirements are unverified when missing.')
