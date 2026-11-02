"""Full-data reference and joint-angle SVM bundles with matching live encoding."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import pickle
from pathlib import Path
import tempfile

import joblib
import numpy as np
import sklearn

from research.geometry import measure_hand_features
from research.image_angles import projected_angles, PIXEL_ANGLE_COLUMNS, VERSION as PIXEL_VERSION
from research.inputs import ANGLE_COLUMNS, LABELS, encode_angles
from research.shin import VERSION as SHIN_VERSION, combined_features
from research.studies.compare_angles import classifier

BUNDLE_VERSION = 'shin-angles-live-v1'
IMAGE_ANGLE_BUNDLE_VERSION = 'shin-angles-image2d-live-v1'
BASELINE_BUNDLE_VERSION = 'shin-live-v1'
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models/asl_shin_angles.joblib'
DEFAULT_BASELINE_MODEL = DEFAULT_MODEL.with_name('asl_shin.joblib')


def detection_inputs(detection, variant="shin_angles"):
    """820, 835 or 849 float32 inputs matching the frozen experiment schema."""
    if detection.get('status') != 'ok':
        raise ValueError('Exactly one valid hand must be visible.')
    shin = list(combined_features(detection['image_landmarks']).values())
    if variant == 'shin':
        return np.asarray(shin,dtype=np.float32).reshape(1,820)
    if variant not in ('shin_angles','shin_angles_2d'):
        raise ValueError('Unknown model input.')
    angles = encode_angles(measure_hand_features(detection['world_landmarks'], detection.get('handedness')))
    row = np.r_[shin, angles].astype(np.float32).reshape(1, 835)
    if not np.isfinite(row[:, :820]).all() or np.isinf(row[:, 820:]).any():
        raise ValueError('Invalid hand geometry for letter recognition.')
    if variant=='shin_angles_2d':
        row=np.c_[row,projected_angles(detection['image_landmarks'],detection.get('image_size'))[None,:]]
    return row


def load_bundle(path):
    # Only load trusted local artifacts produced by this training command.
    try:
        bundle = joblib.load(path)
    except (EOFError, pickle.UnpicklingError, AttributeError, KeyError, TypeError, IndexError) as error:
        raise ValueError("The model file is damaged or has an invalid format.") from error
    if not isinstance(bundle, dict) or bundle.get('version') not in (BUNDLE_VERSION, BASELINE_BUNDLE_VERSION, IMAGE_ANGLE_BUNDLE_VERSION):
        raise ValueError('This file is not a supported live ASL model.')
    baseline = bundle['version'] == BASELINE_BUNDLE_VERSION
    expected_angles = () if baseline else ANGLE_COLUMNS
    if bundle.get('shin_version') != SHIN_VERSION or tuple(bundle.get('angle_columns', ())) != expected_angles:
        raise ValueError('The model feature version does not match the code.')
    image_angles = bundle['version']==IMAGE_ANGLE_BUNDLE_VERSION
    if image_angles and (tuple(bundle.get('pixel_angle_columns',()))!=PIXEL_ANGLE_COLUMNS or bundle.get('pixel_angle_version')!=PIXEL_VERSION):
        raise ValueError('The model 2D feature version does not match the code.')
    model = bundle.get('estimator')
    if getattr(model, 'n_features_in_', None) != (820 if baseline else 849 if image_angles else 835) or list(getattr(model, "classes_", ())) != LABELS:
        raise ValueError('The model has an invalid feature width or incomplete A-Z classes.')
    if bundle.get('sklearn_version') != sklearn.__version__:
        raise ValueError('Install the scikit-learn version in requirements.txt or retrain the model.')
    return bundle


def input_variant(bundle):
    if bundle.get('version')==IMAGE_ANGLE_BUNDLE_VERSION:return 'shin_angles_2d'
    return 'shin' if bundle.get('version') == BASELINE_BUNDLE_VERSION else 'shin_angles'


def model_inputs(bundle, detection):
    return detection_inputs(detection,input_variant(bundle))


def preprocessed_inputs(bundle, row):
    from sklearn.pipeline import Pipeline
    model=bundle['estimator']
    return model[:-1].transform(row) if isinstance(model,Pipeline) else row.copy()


def predict_letter(bundle, detection):
    if detection.get('status') != 'ok':
        return None
    row = model_inputs(bundle,detection)
    model = bundle['estimator']
    letter = str(model.predict(row)[0])
    values = np.asarray(model.decision_function(row))[0]
    order = np.argsort(values)[::-1]
    alternatives = [str(model.classes_[i]) for i in order if str(model.classes_[i]) != letter][:2]
    return {'letter': letter, 'alternatives': alternatives}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/asl_kaggle/research'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--variant', choices=('shin','shin_angles'), default='shin_angles')
    parser.add_argument('--force', action='store_true', help='Replace an existing final live model.')
    args = parser.parse_args()
    args.out = args.out or (DEFAULT_BASELINE_MODEL if args.variant=='shin' else DEFAULT_MODEL)
    width = 820 if args.variant=='shin' else 835
    if args.out.exists() and not args.force:
        parser.error('Model already exists; use --force to retrain.')
    meta_bytes = (args.cache / 'inputs_metadata.json').read_bytes()
    metadata = json.loads(meta_bytes)
    with np.load(args.cache / 'inputs.npz') as arrays:
        X = (arrays['X'][:,:820] if width==820 else np.c_[arrays['X'][:, :820], arrays['angles']]).astype(np.float32)
    y = np.array([s['label'] for s in metadata['samples']])
    if X.shape != (60251, width) or len(y) != len(X) or sorted(set(y)) != LABELS:
        raise ValueError('The final fit requires the 60,251 historical detections from all 26 letters.')
    if len({s['id'] for s in metadata['samples']}) != len(y) or not np.isfinite(X).all():
        raise ValueError('Invalid or duplicate training input.')
    model = classifier('svm',args.variant,len(y))
    print(f'Training SVM: {len(y):,} images, {width} features; all accepted development data.', flush=True)
    model.fit(X, y)
    provenance = {
        'version': BASELINE_BUNDLE_VERSION if width==820 else BUNDLE_VERSION, 'input_variant':args.variant,
        'shin_version': SHIN_VERSION, 'angle_columns': [] if width==820 else list(ANGLE_COLUMNS),
        'sklearn_version': sklearn.__version__, 'training_images': len(y),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_hashes': metadata['hashes'], 'metadata_sha256': sha256(meta_bytes).hexdigest(),
        'input_cache_sha256': sha256((args.cache / 'inputs.npz').read_bytes()).hexdigest(),
        'model': 'SVM RBF C=10 gamma=scale balanced' + ('; train-only median imputation and angle standardization' if width==835 else '; original Shin features only'),
        'scope': 'Final deployment fit on all accepted development images. This fit has no independent test score; see the frozen five-fold study.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.out.parent, suffix='.joblib', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump({**provenance, 'estimator': model}, temporary, compress=3)
        load_bundle(temporary)
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Live model saved: {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
