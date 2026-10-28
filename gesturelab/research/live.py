"""Same-frame live A/B observations, explicitly development-only and unsmoothed by default."""
from datetime import datetime, timezone
from io import BytesIO
import hashlib
from pathlib import Path
import uuid

from gesturelab.camera.runtime import StableLabel
from gesturelab.research.models import compatible_models
from gesturelab.research.recording import detector_configuration
from gesturelab.validation.report import write_json


def correctness(first, second, intended=None):
    if intended is None:
        return 'same_prediction' if first == second else 'different_predictions'
    a, b = first == intended, second == intended
    return 'both_correct' if a and b else ('only_a_correct' if a else ('only_b_correct' if b else 'both_incorrect'))


def predict_one(model, detection, width, height):
    if detection.get('status') != 'ok':
        return dict(prediction=None, status='no_usable_detection', error=None)
    try:
        return dict(prediction=model.predict(detection, width, height), status='ok', error=None)
    except (ValueError, KeyError, RuntimeError, IndexError, TypeError, ArithmeticError) as error:
        return dict(prediction=None, status='model_error', error=str(error))


class LivePair:
    def __init__(self, first, second, smoothing=1):
        if not isinstance(smoothing, int) or smoothing < 1:
            raise ValueError('Smoothing window must be a positive integer; one means raw.')
        self.compatibility = compatible_models(first, second)
        self.models = first, second
        self.smoothing = smoothing
        self.votes = StableLabel(smoothing), StableLabel(smoothing)
        self.frozen, self.observation, self.frame = False, None, None
        self.sequence, self.intended = 0, None

    def set_intended(self, label):
        if label is not None and label not in self.compatibility['labels']:
            raise ValueError('The intended label must belong to the full model alphabet.')
        self.intended = label

    def process(self, frame, detector):
        if self.frozen and self.observation is not None:
            return self.observation
        self.sequence += 1
        detection = detector.analyze(frame)
        results = [predict_one(model, detection, frame.width, frame.height) for model in self.models]
        stable = [vote.update(result['prediction']) for vote, result in zip(self.votes, results)]
        raw = [result['prediction'] for result in results]
        category = correctness(*raw, self.intended) if all(r['status'] == 'ok' for r in results) else 'unscorable_detection_or_model_error'
        self.observation = dict(frame_id=uuid.uuid4().hex, sequence=self.sequence, at=datetime.now(timezone.utc).isoformat(),
            intended_label=self.intended, detection=detection, models=[m.metadata for m in self.models],
            detector_configuration=detector_configuration(detector), width=frame.width, height=frame.height,
            results=results, raw_predictions=raw, smoothed_predictions=stable if self.smoothing > 1 else None,
            smoothing_window=self.smoothing, category=category, role='development',
            note='Selected live frames are descriptive development/demo observations, not independent trials or test ground truth.')
        self.frame = frame.copy()
        return self.observation

    def save(self, out):
        if self.observation is None:
            raise ValueError('No synchronized frame is available.')
        folder = Path(out) / self.observation['frame_id']
        folder.mkdir(parents=True, exist_ok=False)
        buffer = BytesIO()
        self.frame.convert('RGB').save(buffer, format='PNG')
        data = buffer.getvalue()
        (folder / 'frame.png').write_bytes(data)
        write_json(folder / 'observation.json', {**self.observation, 'image_sha256': hashlib.sha256(data).hexdigest(),
            'saved_at': datetime.now(timezone.utc).isoformat(), 'predictions_exposed': True})
        return folder
