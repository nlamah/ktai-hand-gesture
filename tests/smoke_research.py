"""SYNTHETIC SOFTWARE TEST of the research workflow; no physical camera or ASL score."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_research import photo, detection
from gesturelab.research.models import ModelHandle, register_model
from gesturelab.research.protocol import create_protocol, freeze_test
from gesturelab.research.recording import save_attempt
from gesturelab.research.live import LivePair
from gesturelab.research.analysis import analyze_development, export_live
from gesturelab.research.improvement import train_improvement
from gesturelab.research.evaluation import evaluate_test
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.fitting import train_variant
from gesturelab.validation.report import read_json, sha256, write_json


def _attempts(root, protocol, color, include_failure=False):
    plan = read_json(protocol)
    project = root / ('test_photos' if plan['role'] == 'test' else 'development_photos')
    for index, trial in enumerate(plan['trials']):
        status = 'no_hand' if include_failure and index == 2 else 'ok'
        save_attempt(project, Image.new('RGB', (32, 32), (color, index, 120)), detection(trial['label'], status),
            plan, trial, sha256(protocol), {'synthetic': True, 'detector': 'not-used'})
    return project


def _protocol(root, role):
    return create_protocol(root / (role + '.json'), 'ASL', list('AJZ'), 'synthetic_person', role + '_session',
        role=role, repetitions=1, countdown=0, confirm_label_validity=True,
        conditions=[dict(name='front', nominal_angle=0), dict(name='left15', nominal_angle=-15)])


def run(out):
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    project = root / 'original'
    for group in range(3):
        for label in 'AJZ': photo(project, group, label)
    experiment = prepare_experiment(project, root / 'baseline', folds=3, allow_no_test=True)
    folder = train_variant(experiment, 'shin2021_combined')
    baseline = register_model(folder / 'model.joblib', root / 'baseline.json', experiment, 'ASL', 'SYNTHETIC baseline', True)
    development = _attempts(root, _protocol(root, 'development'), 201)
    analyze_development(baseline, development, root / 'development_analysis')
    improved = train_improvement(baseline, development, root / 'augmented', root / 'improved',
        'SYNTHETIC SOFTWARE TEST', 'Synthetic added viewpoints', 'Software execution only', all_development=True)
    pair = LivePair(ModelHandle.load(baseline), ModelHandle.load(improved / 'registered_model.json'))
    pair.set_intended('Z')
    pair.process(Image.new('RGB', (32, 32), 'orange'), SimpleNamespace(analyze=lambda _: detection('Z')))
    pair.save(root / 'live')
    export_live(root / 'live', root / 'live_export')
    protocol = _protocol(root, 'test')
    seal = freeze_test(protocol, baseline, improved / 'registered_model.json', root / 'sealed.json', 'SYNTHETIC SOFTWARE TEST')
    test = _attempts(root, protocol, 245, include_failure=True)
    evaluate_test(seal, test, root / 'test_result')
    write_json(root / 'SOFTWARE_TEST.json', dict(scope=__doc__, physical_webcam_tested=False,
        human_label_or_angle_validation=False, usable_scores_are_not_asl_accuracy=True))
    return root / 'test_result/report.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    print(run(parser.parse_args().out))
