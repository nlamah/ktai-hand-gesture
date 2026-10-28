"""Store every scheduled attempt, including failed detections and explicit retakes."""
from datetime import datetime, timezone
from io import BytesIO
import hashlib
from pathlib import Path
import uuid

from gesturelab.datasets.data import project_lock, read_manifest, atomic_json, MODEL_SHA256
from gesturelab.validation.report import read_json, sha256

ANALYSIS_KEYS = ('status', 'n_hands', 'mediapipe_version', 'feature_version', 'image_landmarks',
                 'world_landmarks', 'handedness', 'handedness_score', 'error')


def detector_configuration(detector):
    options = getattr(detector, 'options', None)
    fields = ('num_hands', 'min_hand_detection_confidence', 'min_hand_presence_confidence')
    return dict(model_sha256=MODEL_SHA256, running_mode='IMAGE',
                options={name: getattr(options, name, None) for name in fields},
                unknown_options=options is None)


def save_attempt(project, frame, detection, protocol, trial, protocol_hash, detector_config, retake_of=None):
    data = BytesIO()
    frame.convert('RGB').save(data, format='PNG')
    payload, image = data.getvalue(), frame.convert('RGB')
    photo_id = uuid.uuid4().hex[:24]
    now = datetime.now(timezone.utc).isoformat()
    sample = dict(id=photo_id, filename=photo_id + '.png', created_at=now, captured_at=now,
        original_path=f'originals/{photo_id}.png', image_path=f'images/{photo_id}.png',
        original_sha256=hashlib.sha256(payload).hexdigest(), pixel_sha256=hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest(),
        width=image.width, height=image.height, label=trial['label'], alphabet=protocol['alphabet'],
        participant=protocol['participant'], session=protocol['session'], condition=trial['condition'],
        nominal_angle=trial['nominal_angle'], repetition=trial['repetition'], capture_order=trial['order'],
        role=protocol['role'], included=True, invalid_reason=None, detector_configuration=detector_config,
        provenance=dict(type='research_protocol', protocol_id=protocol['id'], protocol_sha256=protocol_hash,
            protocol_role=protocol['role'], source_group=f"protocol:{protocol['participant']}:{protocol['session']}",
            trial_order=trial['order'], retake_of=retake_of, predictions_exposed=False,
            angle_method=protocol['angle_method'], label_validity_confirmed=protocol['label_validity_confirmed']))
    sample.update({key: detection[key] for key in ANALYSIS_KEYS if key in detection})
    sample.setdefault('status', 'unavailable')
    with project_lock(project) as project:
        manifest = read_manifest(project)
        if any(s['participant'] == sample['participant'] and s['session'] == sample['session']
               and s['role'] != sample['role'] for s in manifest['samples']):
            raise ValueError('A participant/session cannot mix development and test roles.')
        (project / sample['original_path']).write_bytes(payload)
        (project / sample['image_path']).write_bytes(payload)
        manifest['samples'].append(sample)
        atomic_json(project / 'manifest.json', manifest)
    return sample


def annotate_attempt(project, photo_id, note, invalid=False):
    if not note.strip():
        raise ValueError('A note/reason is required; no silent deletion is allowed.')
    with project_lock(project) as project:
        manifest = read_manifest(project)
        sample = next(s for s in manifest['samples'] if s['id'] == photo_id)
        if sample.get('provenance', {}).get('type') != 'research_protocol':
            raise ValueError('This command annotates new research attempts only; historical datasets are preserved.')
        sample.setdefault('human_annotations', []).append(dict(note=note, invalid=invalid,
            at=datetime.now(timezone.utc).isoformat(), author='researcher', evidence='Explicit command, not an automatically established cause.'))
        if invalid:
            sample.update(included=False, invalid_reason=note)
        atomic_json(project / 'manifest.json', manifest)


def validate_capture(protocol_file, project, seal=None, resume=False):
    protocol = read_json(protocol_file)
    digest = sha256(protocol_file)
    if not protocol.get('label_validity_confirmed'):
        raise ValueError('Review gesture-label validity for the chosen viewpoint method, then create a confirmed protocol.')
    if protocol['role'] == 'test':
        if not seal:
            raise ValueError('Final-test capture requires a model/protocol seal created before recording.')
        frozen = read_json(seal)
        if frozen['protocol_sha256'] != digest or frozen.get('viewed_retest'):
            raise ValueError('Protocol changed after models/settings were locked.')
        if any(sha256(model['model_file']) != model['model_sha256'] for model in frozen['models']):
            raise ValueError('A locked model changed before capture.')
        from gesturelab.research.evaluation import exposure_history
        if exposure_history(project, protocol):
            raise ValueError('This test protocol was already evaluated. New independent collection needs a new protocol/session.')
    existing = [s for s in read_manifest(project)['samples'] if s.get('provenance', {}).get('protocol_id') == protocol['id']]
    if existing and not resume:
        raise ValueError('This protocol already has recordings. Use --resume for missing slots or create a new session/protocol.')
    if any(s['provenance']['protocol_sha256'] != digest for s in existing):
        raise ValueError('Existing attempts have a different protocol hash.')
    return protocol, digest


def resume_schedule(schedule, project):
    existing = [s for s in read_manifest(project)['samples']
                if s.get('provenance', {}).get('protocol_id') == schedule.protocol['id']]
    orders = {s['capture_order'] for s in existing}
    if orders and orders != set(range(1, max(orders) + 1)):
        raise ValueError('Cannot resume: planned capture order has gaps.')
    schedule.index = max(orders, default=0)
    if schedule.index > len(schedule.protocol['trials']):
        raise ValueError('Recorded attempts exceed the planned trial order.')
    schedule.last_id = existing[-1]['id'] if existing else None
    return schedule


class CaptureSchedule:
    def __init__(self, protocol):
        self.protocol, self.index, self.paused = protocol, 0, True
        self.deadline, self.remaining, self.retake_of = None, protocol['countdown'], None
        self.last_id, self.retake_trial = None, None

    @property
    def current(self):
        return self.retake_trial or (self.protocol['trials'][self.index] if self.index < len(self.protocol['trials']) else None)

    def toggle(self, now):
        if self.paused:
            self.paused, self.deadline = False, now + self.remaining
        else:
            self.remaining = max(0, self.deadline - now) if self.deadline is not None else self.protocol['countdown']
            self.paused = True

    def due(self, now):
        return self.current is not None and not self.paused and self.deadline is not None and now >= self.deadline

    def saved(self, photo_id, now):
        previous = self.current
        self.last_id = photo_id
        if self.retake_trial is None:
            self.index += 1
        self.retake_of, self.retake_trial = None, None
        self.remaining = self.protocol['countdown']
        self.deadline = now + self.protocol['interval'] + self.protocol['countdown']
        if self.current is None or self.current['condition'] != previous['condition']:
            self.paused = True

    def retake(self, now):
        if not self.last_id or self.index < 1:
            raise ValueError('No previous attempt is available for an explicit retake.')
        self.retake_trial = self.protocol['trials'][self.index - 1]
        self.retake_of = self.last_id
        self.paused, self.deadline = False, now + self.protocol['countdown']
