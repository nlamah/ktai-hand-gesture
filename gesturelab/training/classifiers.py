"""Classifier construction and explicitly named, frozen parameter profiles."""
from importlib.metadata import version

from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from gesturelab.features.training_inputs import training_input
from gesturelab.training.distance_weighting import validate_tip_weight, tip_weighting_step
from gesturelab.training.neural_network import BalancedMLPClassifier, mlp_parameters

MODEL_NAMES = {'knn': 'k-NN', 'svm': 'SVM (RBF)', 'logistic': 'Logistic regression',
               'random_forest': 'Random forest', 'decision_tree': 'Decision tree', 'lightgbm': 'LightGBM',
               'mlp': 'MLP (64, 32)'}
PROFILES = ('project', 'shin2021_asl')


def _paper_parameters(name, variant):
    """Tables 5/7 ASL Alphabet rows; baseline fallback uses the combined settings."""
    if name == 'svm':
        gamma = .001 if variant in ('shin2021_distance', 'shin2021_angle') else 'scale'
        return dict(C=1000, gamma=gamma, kernel='rbf', class_weight=None)
    if name == 'lightgbm':
        rounds = {'shin2021_distance': 12, 'shin2021_angle': 13}.get(variant, 14)
        return dict(num_leaves=100, learning_rate=.1, min_child_samples=25, n_estimators=rounds)
    raise ValueError('The shin2021_asl profile supports only svm and lightgbm.')


def classifier_parameters(name, variant=None, profile='project', n_train=100, seed=42):
    """Resolve settings without fitting or consulting validation/test outcomes."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown classifier profile: {profile}.')
    defaults = {
        'knn': dict(n_neighbors=min(5, n_train), weights='distance'),
        'svm': dict(C=10, gamma='scale', kernel='rbf', class_weight='balanced'),
        'logistic': dict(C=1, max_iter=2000, class_weight='balanced', random_state=seed),
        'random_forest': dict(n_estimators=150, min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=-1),
        'decision_tree': dict(max_depth=6, min_samples_leaf=2, class_weight='balanced', random_state=seed),
        'lightgbm': dict(num_leaves=31, learning_rate=.1, min_child_samples=20, n_estimators=100),
        'dummy': dict(strategy='most_frequent'),
        'mlp': mlp_parameters(seed),
    }
    if name not in defaults:
        raise ValueError(f'Unknown model: {name}')
    parameters = _paper_parameters(name, variant) if profile == 'shin2021_asl' else defaults[name]
    if name == 'lightgbm':
        parameters.update(random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    return parameters


def _lightgbm(parameters):
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError) as error:
        raise ValueError('Install requirements.txt for LightGBM. On macOS also install llvm-openmp in Conda; see docs/PAPER_SHIN2021.md.') from error
    return LGBMClassifier(**parameters)


def _preprocessor(variant, columns):
    spec = training_input(variant) if variant is not None else None
    if spec is not None and spec.scaled_prefixes:
        if columns is None:
            raise ValueError('Column names are required for selective input scaling.')
        scaled = [i for i, column in enumerate(columns) if column.startswith(spec.scaled_prefixes)]
        unscaled = [i for i in range(len(columns)) if i not in scaled]
        if not scaled:
            raise ValueError('No columns match the input scaling definition.')
        return ColumnTransformer([('paper', 'passthrough', unscaled),
                                  ('additions', StandardScaler(), scaled)], sparse_threshold=0)
    return StandardScaler() if spec is None or spec.standardize else None


def estimator(name, n_train=100, seed=42, variant=None, profile='project', columns=None, index_middle_tip_weight=1.0):
    """Keep learned scaling inside each fit; paper inputs already define their own normalization."""
    weight = validate_tip_weight(index_middle_tip_weight, name)
    parameters = classifier_parameters(name, variant, profile, n_train, seed)
    factories = {'knn': KNeighborsClassifier, 'svm': SVC, 'logistic': LogisticRegression,
                 'random_forest': RandomForestClassifier, 'decision_tree': DecisionTreeClassifier,
                 'dummy': DummyClassifier, 'lightgbm': _lightgbm, 'mlp': BalancedMLPClassifier}
    model = _lightgbm(parameters) if name == 'lightgbm' else factories[name](**parameters)
    preprocessing = _preprocessor(variant, columns) if name in ('knn', 'svm', 'logistic', 'mlp') else None
    steps = [preprocessing] if preprocessing is not None else []
    if weight != 1:
        steps.append(tip_weighting_step(preprocessing, columns, weight))
    return make_pipeline(*steps, model) if steps else model


def classifier_versions(name):
    return {'lightgbm': version('lightgbm')} if name == 'lightgbm' else {}
