"""Lossless, reproducible manual snapshots for exploratory live error analysis."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from research.inputs import ANGLE_COLUMNS
from research.image_angles import PIXEL_ANGLE_COLUMNS
from research.live_model import model_inputs, preprocessed_inputs, input_variant
from research.shin import combined_features

DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parents[1] / 'captures/confusions'


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_capture(packet, expected, bundle, model_path, model_sha256, preview, output=DEFAULT_CAPTURE_DIR):
    """Save the frozen displayed frame and its original inference, never redetect."""
    expected = expected.strip().upper() if expected else None
    if expected is not None and (len(expected) != 1 or expected not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('De bedoelde letter moet één letter A–Z zijn.')
    if packet.get('protocol') and expected != packet['protocol']['task']['letter']:
        raise ValueError('Annotatie wijkt af van de geplande protocolletter.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    detection = {**packet['detection'], 'image_size':[packet['model_frame'].shape[1],packet['model_frame'].shape[0]]}
    prediction = packet['prediction']
    record = {
        'schema': 'asl-confusion-capture-v1', 'id': capture_id,
        'frame_captured_at_utc': packet['captured_at_utc'],
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
        'expected_letter': expected, 'expected_source': 'user_annotation' if expected else None,
        'predicted_letter': prediction['letter'] if prediction else None,
        'is_confusion': expected != prediction['letter'] if expected and prediction else None,
        'prediction': prediction, 'input_variant': input_variant(bundle), 'detection': detection, 'protocol': packet.get('protocol'),
        'model': {'path': str(Path(model_path).resolve()), 'sha256': model_sha256,
                  'metadata': {k: v for k, v in bundle.items() if k != 'estimator'}},
        'camera': packet.get('camera'), 'session_id': packet.get('session_id'),
        'capture_round': packet.get('capture_round'), 'landmarks_visible': packet.get('landmarks_visible'), 'opencv_version': cv2.__version__,
        'orientation': {'frame.png': 'unmirrored raw camera frame',
                        'model_input.png': 'unmirrored exact detector input, aspect ratio preserved',
                        'preview.png': 'mirrored display with predicted letter'},
        'feature_names': None, 'raw_model_features': None, 'preprocessed_model_features': None,
        'decision_scores': None,
        'notes': 'Manually selected exploratory example, not a representative test sample. SVM decision scores are not probabilities.',
    }
    if packet.get('protocol'):
        record['expected_source'] = 'guided_protocol_confirmed_by_user'
        record['notes'] = 'Planned protocol attempt, including errors and failed detections. Not training data. SVM decision scores are not probabilities.'
        record['orientation']['preview.png'] = 'mirrored preview; prediction hidden during guided collection'
    if packet.get('preview_mirrored') is False:
        record['orientation']['preview.png'] = 'unmirrored display; prediction hidden during guided collection' if packet.get('protocol') else 'unmirrored display with predicted letter'
    if detection.get('status') == 'ok':
        row = model_inputs(bundle,detection)
        model = bundle['estimator']
        record['feature_names'] = list(combined_features(detection['image_landmarks'])) + (list(ANGLE_COLUMNS) if row.shape[1]>=835 else []) + (list(PIXEL_ANGLE_COLUMNS) if row.shape[1]==849 else [])
        record['raw_model_features'] = row[0]
        record['joint_angles_degrees'] = dict(zip(ANGLE_COLUMNS, row[0, 820:835])) if row.shape[1]>=835 else None
        record['projected_angles_degrees'] = dict(zip(PIXEL_ANGLE_COLUMNS,row[0,835:849])) if row.shape[1]==849 else None
        record['preprocessed_model_features'] = preprocessed_inputs(bundle,row)[0]
        values = model.decision_function(row)[0]
        record['decision_scores'] = dict(zip(model.classes_, values))
    directory = Path(tempfile.mkdtemp(prefix='.capture-', dir=output))
    try:
        for filename, bgr in [('frame.png', packet['frame']), ('model_input.png', packet['model_frame']), ('preview.png', preview)]:
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(directory / filename)
        record['files'] = {name: {'sha256': sha256((directory / name).read_bytes()).hexdigest()}
                           for name in ('frame.png', 'model_input.png', 'preview.png')}
        (directory / 'capture.json').write_text(json.dumps(_json_value(record), indent=2, allow_nan=False) + '\n')
        destination = output / capture_id
        directory.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
