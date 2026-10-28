"""Immutable nominal-viewpoint protocol, reference provenance and randomized trial order."""
from datetime import datetime, timezone
from pathlib import Path
import random
import uuid

from gesturelab.research.models import new_json, ModelHandle, compatible_models
from gesturelab.validation.report import read_json, sha256

DEFAULT_CONDITIONS = [('front', 0), ('left15', -15), ('right15', 15), ('left30', -30), ('right30', 30)]
ASL_SOURCE = 'https://www.lifeprint.com/asl101/fingerspelling/images/signlanguageabc02.jpg'
# Display regions of the existing Vicars/Lifeprint chart, not invented sign drawings.
ASL_REGIONS = {'A': (95, 55, 225, 240), 'B': (335, 0, 460, 240), 'C': (580, 35, 710, 240),
 'D': (825, 0, 955, 240), 'E': (1065, 40, 1195, 240), 'F': (205, 165, 335, 385),
 'G': (460, 160, 575, 340), 'H': (700, 190, 820, 350), 'I': (940, 135, 1060, 345),
 'J': (105, 335, 260, 525), 'K': (335, 300, 475, 530), 'L': (575, 330, 715, 530),
 'M': (810, 325, 935, 525), 'N': (1065, 335, 1195, 530), 'O': (215, 520, 335, 695),
 'P': (445, 530, 560, 690), 'Q': (675, 530, 825, 690), 'R': (945, 450, 1060, 690),
 'S': (100, 650, 215, 815), 'T': (330, 645, 455, 820), 'U': (575, 585, 690, 820),
 'V': (815, 585, 935, 820), 'W': (1065, 585, 1195, 825), 'X': (200, 790, 335, 985),
 'Y': (440, 790, 595, 990), 'Z': (680, 815, 845, 990)}


def reference_spec(alphabet, letters, folder=None, source=None):
    if folder:
        if not source:
            raise ValueError('Provide the source/credit for supplied reference images.')
        paths = {label: next((Path(folder) / (label + ext) for ext in ('.png', '.jpg') if (Path(folder) / (label + ext)).is_file()), None) for label in letters}
        if any(path is None for path in paths.values()):
            raise ValueError('A reference image is missing; no replacement will be invented.')
        return {label: dict(file=str(p.resolve()), sha256=sha256(p), crop=None, alphabet=alphabet, source=source) for label, p in paths.items()}
    if alphabet != 'ASL' or any(label not in ASL_REGIONS for label in letters):
        raise ValueError('Supply alphabet-specific references; NGT drawings are never used as an ASL fallback.')
    chart = Path(__file__).resolve().parents[2] / 'assets/asl_alphabet.jpg'
    return {label: dict(file=str(chart), sha256=sha256(chart), crop=ASL_REGIONS[label], alphabet='ASL',
        source=ASL_SOURCE, credit='William Vicars, ASL University/Lifeprint (2010); see assets/SOURCES.md',
        limitation='A crop of the supplied chart; J/Z arrows do not constitute a full movement example.') for label in letters}


def trial_order(letters, conditions, repetitions, seed):
    rng, result = random.Random(seed), []
    for repeat in range(1, repetitions + 1):
        sequence = list(conditions)
        rng.shuffle(sequence)
        for condition in sequence:
            for label in letters:
                result.append(dict(order=len(result) + 1, label=label, condition=condition['name'],
                    nominal_angle=condition['nominal_angle'], repetition=repeat))
    return result


def create_protocol(out, alphabet, letters, participant, session, role='development', repetitions=5,
                    countdown=3., interval=3., camera=0, seed=42, angle_method='camera_view',
                    confirm_label_validity=False, conditions=None, references=None, reference_source=None):
    if role not in ('development', 'test') or angle_method not in ('camera_view', 'gesture_rotation'):
        raise ValueError('Choose a development/test role and explicit viewpoint/gesture method.')
    if not letters or len(set(letters)) != len(letters) or not participant or not session or repetitions < 1:
        raise ValueError('Distinct letters, participant, session and positive repetition count are required.')
    if countdown < 0 or interval < 0:
        raise ValueError('Timing must be nonnegative.')
    conditions = conditions or [dict(name=name, nominal_angle=angle) for name, angle in DEFAULT_CONDITIONS]
    if len({c['name'] for c in conditions}) != len(conditions):
        raise ValueError('Condition names must be distinct.')
    refs = reference_spec(alphabet, letters, references, reference_source)
    protocol = dict(schema=1, id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), alphabet=alphabet,
        letters=list(letters), evaluation_classes=list(letters), participant=participant, session=session, role=role,
        conditions=conditions, repetitions=repetitions, countdown=countdown, interval=interval, camera=camera,
        seed=seed, angle_method=angle_method, label_validity_confirmed=confirm_label_validity,
        angles_are='Nominal arrangements, not measured camera angles; no angle inferred from landmarks.',
        references=refs, trials=trial_order(letters, conditions, repetitions, seed), predictions_exposed=False,
        instruction='Relax your hand and form the gesture again before each separate repetition.')
    return new_json(out, protocol)


def freeze_test(protocol, model_a, model_b, out, hypothesis, viewed_retest=False):
    path = Path(protocol)
    plan = read_json(path)
    if plan['role'] != 'test' or not plan['label_validity_confirmed']:
        raise ValueError('A test protocol with researcher-confirmed label validity is required.')
    first, second = ModelHandle.load(model_a), ModelHandle.load(model_b)
    compatibility = compatible_models(first, second)
    if not set(plan['letters']) <= set(compatibility['labels']):
        raise ValueError('Protocol evaluation classes must be among the models’ full original classes.')
    frozen = dict(schema=1, id=uuid.uuid4().hex, protocol_file=str(path.resolve()), protocol_sha256=sha256(path),
        frozen_at=datetime.now(timezone.utc).isoformat(), hypothesis=hypothesis, viewed_retest=viewed_retest,
        evaluation_classes=plan['evaluation_classes'],
        models=[first.metadata, second.metadata], compatibility=compatibility,
        independence_limitation='The seal records ordering and identities; it does not prove human independence or correct nominal angles.')
    return new_json(out, frozen)
