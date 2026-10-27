"""Own model-input composition, extraction and input-code provenance.

Training, evaluation and camera code use this interface without knowing which
measurements an input contains. Add variants and dependencies here only.
"""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Callable

import pandas as pd

from gesturelab.features.inspection import FEATURE_VERSION
from gesturelab.features.landmark_features import landmarks_for_sample
from gesturelab.features.finger_features import finger_features_for_sample
from gesturelab.features.paper_shin2021 import distance_for_sample, angle_for_sample, combined_for_sample, VERSION


@dataclass(frozen=True)
class TrainingInput:
    title: str
    extractors: tuple[Callable, ...]
    prefixes: tuple[str, ...]
    expected_features: int
    dependencies: tuple[str, ...] = ()
    standardize: bool = True
    definition: str = 'project'

    def columns(self, available):
        # Prefix order explicitly defines the model's column order.
        selected = list(dict.fromkeys(
            name for prefix in self.prefixes for name in available if name.startswith(prefix)
        ))
        if len(selected) != self.expected_features:
            raise ValueError(f'Expected {self.expected_features} features; found {len(selected)}.')
        return selected

    def features(self, sample, thresholds):
        available = {}
        for extractor in self.extractors:
            values = extractor(sample, thresholds)
            duplicates = available.keys() & values.keys()
            if duplicates:
                raise ValueError(f'Feature extractors produced duplicate columns: {sorted(duplicates)}')
            available.update(values)
        return {name: available[name] for name in self.columns(available)}

    def source_files(self):
        """Track extractor modules automatically; declare additional helpers here."""
        package = Path(__file__).resolve().parents[1]
        paths = {Path(inspect.getfile(extractor)).resolve() for extractor in self.extractors}
        paths.update((package / name).resolve() for name in self.dependencies)
        if any(not path.is_relative_to(package) or path.suffix != ".py" or not path.is_file() for path in paths):
            raise ValueError('Input extractors and dependencies must be Python files in gesturelab.')
        return {path.relative_to(package).as_posix() for path in paths}


DEFAULT_VARIANTS = ('without_features', 'with_features')
PAPER_VARIANTS = ('shin2021_distance', 'shin2021_angle', 'shin2021_combined')
PAPER_COMPARISON = DEFAULT_VARIANTS + PAPER_VARIANTS
TRAINING_INPUTS = {
    'finger_features_only': TrainingInput(
        title='Finger features only', extractors=(finger_features_for_sample,),
        prefixes=('geom_',), expected_features=81,
    ),
    'without_features': TrainingInput(
        title='Without finger features', extractors=(landmarks_for_sample,),
        prefixes=('norm_', 'world_'), expected_features=126,
    ),
    'with_features': TrainingInput(
        title='With finger features',
        extractors=(landmarks_for_sample, finger_features_for_sample),
        prefixes=('norm_', 'geom_', 'world_'), expected_features=207,
    ),
    'shin2021_distance': TrainingInput(
        'Shin 2021: distances', (distance_for_sample,), ('shin_dist_',), 190,
        standardize=False, definition=VERSION,
    ),
    'shin2021_angle': TrainingInput(
        'Shin 2021: axis cosines', (angle_for_sample,), ('shin_angle_',), 630,
        standardize=False, definition=VERSION,
    ),
    'shin2021_combined': TrainingInput(
        'Shin 2021: distances + axis cosines', (combined_for_sample,), ('shin_dist_', 'shin_angle_'), 820,
        standardize=False, definition=VERSION,
    ),
}


def training_input(name):
    try:
        return TRAINING_INPUTS[name]
    except KeyError:
        raise ValueError(f'Unknown training input variant: {name!r}.') from None


def input_source_files():
    """All source files to freeze alongside a shared experiment plan."""
    return {'features/training_inputs.py'} | set().union(*(spec.source_files() for spec in TRAINING_INPUTS.values()))


def input_table(samples, thresholds, variant):
    """Build deterministic per-photo inputs; fitting and splitting happen elsewhere."""
    spec = training_input(variant)
    metadata_keys = ('id', 'label', 'participant', 'session', 'condition', 'role', 'image_path')
    rows = []
    for sample in samples:
        if sample['status'] == 'ok':
            metadata = {key: sample[key] for key in metadata_keys}
            try:
                features = spec.features(sample, thresholds)
            except ValueError as error:
                raise ValueError(f'Cannot build {variant} for photo {sample["id"]}: {error}') from error
            rows.append({**metadata, **features})
    return pd.DataFrame(rows)


def model_input_metadata(variant, thresholds, code_hashes):
    """Describe input reconstruction without exposing its internals to the trainer."""
    sources = training_input(variant).source_files() | {'features/training_inputs.py'}
    return {
        'training_input': variant,
        'feature_config': thresholds,
        'feature_version': FEATURE_VERSION,
        'input_definition': training_input(variant).definition,
        'feature_code_hashes': {name: code_hashes[name] for name in sorted(sources)},
    }


def validate_model_input(bundle):
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('This model uses a different feature version. Retrain it.')
    for name, expected in bundle.get('feature_code_hashes', {}).items():
        source = Path(__file__).resolve().parents[1] / name
        if not source.is_file():
            raise ValueError('Model input code moved or changed. Retrain the model.')
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Model input code changed since training. Retrain the model.')


def model_features(bundle, sample):
    """Reconstruct inputs using the same definition used during training."""
    if bundle.get('training_input'):
        return training_input(bundle['training_input']).features(sample, bundle['feature_config'])
    # Older saved bundles remain readable without branching in camera code.
    if bundle.get('input_mode') == 'image_and_world':
        from gesturelab.features.inspection import features_for_sample
        return features_for_sample(sample, bundle['feature_config'])
    return sample['features']
