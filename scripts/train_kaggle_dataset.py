"""Import, analyze and train both Random Forest variants on every Kaggle letter photo."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from gesturelab.datasets.data import analyze_project, atomic_json, read_manifest
from gesturelab.datasets.kaggle import GROUPING_NOTE, import_kaggle_dataset
from gesturelab.features.training_inputs import DEFAULT_VARIANTS
from gesturelab.training.comparison import compare_variants
from gesturelab.training.experiment_plan import load_experiment, prepare_experiment
from gesturelab.training.fitting import train_variant


class KaggleTrainingRun:
    """Keep long-running stages observable and resume completed work without overwriting it."""

    def __init__(self, args):
        self.args = args
        self.project, self.experiment = Path(args.project).resolve(), Path(args.out).resolve()
        self.status_path = self.project.with_name(self.project.name + '_run.json')
        self.pending, self.last_report = 0, 0

    def report(self, stage, **details):
        status = {'stage': stage, 'pid': os.getpid(), 'project': str(self.project),
                  'experiment': str(self.experiment), 'updated_at': datetime.now(timezone.utc).isoformat(),
                  'grouping_note': GROUPING_NOTE, **details}
        atomic_json(self.status_path, status)
        print(json.dumps(status), flush=True)

    def analysis_progress(self, fraction):
        completed = round(fraction * self.pending)
        if completed - self.last_report >= self.args.checkpoint_every or completed == self.pending:
            self.last_report = completed
            self.report('analysis', completed=completed, pending_at_start=self.pending)

    def import_images(self):
        self.report('import')
        result = import_kaggle_dataset(
            self.args.source, self.project, self.args.block_size, self.args.checkpoint_every,
            resume=self.project.exists(),
            progress=lambda n, total: self.report('import', completed=n, total=total),
            exclude_overlapping_test=True)
        self.report('import_complete', **result)

    def analyze_images(self):
        self.pending = sum(s['status'] == 'pending' for s in read_manifest(self.project)['samples'])
        self.report('analysis', completed=0, pending_at_start=self.pending)
        analyze_project(self.project, self.analysis_progress, self.args.checkpoint_every, cache_features=False)
        samples = read_manifest(self.project)['samples']
        summary = {'total': len(samples), 'statuses': dict(Counter(s['status'] for s in samples)),
                   'roles': dict(Counter(s['role'] for s in samples)), 'grouping_note': GROUPING_NOTE}
        atomic_json(self.project / 'reports/dataset_summary.json', summary)
        self.report('analysis_complete', **summary)

    def train_models(self):
        self.report('prepare_experiment')
        if not self.experiment.exists():
            manifest = read_manifest(self.project)
            overlaps = bool(manifest.get('import', {}).get('test_overlaps'))
            has_test = any(s['role'] == 'test' for s in manifest['samples'])
            prepare_experiment(self.project, self.experiment, folds=self.args.folds,
                               grouping='source', model='random_forest',
                               allow_no_test=overlaps and not has_test, allow_incomplete_test=overlaps)
        load_experiment(self.experiment, self.project)
        for variant in DEFAULT_VARIANTS:
            self.report('training', variant=variant)
            if not (self.experiment / variant / 'training.json').is_file():
                train_variant(self.experiment, variant, self.project)
        self.report('comparison')
        report = compare_variants(self.experiment, project_override=self.project) / 'report.html'
        self.report('complete', report=str(report), final_test_evaluated=False)

    def run(self):
        try:
            self.import_images()
            self.analyze_images()
            self.train_models()
        except BaseException as error:
            self.report('failed', error=str(error) or type(error).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/external/archive-kaggle')
    parser.add_argument('--project', default='data/asl_kaggle')
    parser.add_argument('--out', default='experiments/asl_kaggle_rf_01')
    parser.add_argument('--block-size', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--folds', type=int, default=3)
    KaggleTrainingRun(parser.parse_args()).run()


if __name__ == '__main__':
    main()
