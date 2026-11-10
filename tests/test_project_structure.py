"""Public RQ entry points, shared baseline and independent fresh-run isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from baseline.data import load_shin_inputs
from baseline.runner import ROOT, run_commands
from baseline import shin
from RQ3.workflows import WORKFLOWS, commands_for
from tests.test_shin import known_points


class ProjectStructureTests(unittest.TestCase):
    def test_shared_baseline_reuses_frozen_functions_and_820_columns(self):
        from research import shin as frozen
        self.assertIs(shin.combined_features, frozen.combined_features)
        self.assertEqual(len(shin.combined_features(known_points())), 820)

    def test_active_workflows_do_not_reference_removed_modules(self):
        for plan in WORKFLOWS.values():
            for action in ['reproduce', 'verify', 'report']:
                for command in plan.get(action, []):
                    self.assertEqual(command[0], '-m')
                    self.assertTrue((ROOT / (command[1].replace('.', '/') + '.py')).is_file(), command)

    def test_dry_run_never_executes_and_real_run_uses_project_root(self):
        with patch('baseline.runner.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            command = commands_for('features', 'verify')
            run_commands(command, dry_run=True)
            run.assert_not_called()
            run_commands(command)
            self.assertEqual(run.call_args.kwargs['cwd'], ROOT)
            self.assertTrue(run.call_args.kwargs['check'])

    @staticmethod
    def cache(directory):
        cache = directory / 'cache'
        cache.mkdir()
        rng = np.random.default_rng(9)
        samples = [{'id': f'{letter}-{group}', 'label': letter, 'group': f'block-{group}'}
                   for group in range(5) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        np.savez_compressed(cache / 'inputs.npz', X=rng.normal(size=(130, 916)).astype(np.float32))
        (cache / 'inputs_metadata.json').write_text(json.dumps({'samples': samples}))
        (cache / 'summary.json').write_text(json.dumps({'folds': [
            {'fold': g + 1, 'train_groups': [f'block-{i}' for i in range(5) if i != g], 'test_groups': [f'block-{g}']}
            for g in range(5)]}))
        return cache

    def test_new_baseline_fit_predicts_only_its_recorded_held_out_sources(self):
        from baseline.train import train_cache
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            cache = self.cache(directory)
            output = directory / 'new_results'
            train_cache(cache, output, ['svm'])
            X, y, ids, groups, folds = load_shin_inputs(cache)
            predictions = pd.read_csv(output / 'predictions.csv').set_index('id')
            self.assertEqual(len(predictions), 130)
            for fold, train, test in folds:
                self.assertFalse(set(groups[train]) & set(groups[test]))
                model = joblib.load(output / f'svm_fold{fold}.joblib')
                self.assertEqual(model.n_features_in_, 820)
                np.testing.assert_array_equal(model.predict(X[test]), predictions.loc[ids[test]].prediction)
            # A changed cache cannot silently reuse a previous fit.
            meta = cache / 'inputs_metadata.json'
            meta.write_text(meta.read_text() + '\n')
            with self.assertRaises(ValueError):
                train_cache(cache, output, ['svm'])

    def test_integration_slots_do_not_claim_missing_experiments_are_implemented(self):
        for package in ['RQ1']:
            manifest = json.loads((ROOT / package / 'contribution.json').read_text())
            self.assertNotEqual(manifest['status'], 'implemented')
            self.assertFalse((ROOT / package / 'experiment.py').exists())

    def test_input_loader_rejects_cross_fold_source_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = self.cache(Path(temp))
            path = cache / 'summary.json'
            plan = json.loads(path.read_text())
            plan['folds'][0]['train_groups'].append('block-0')
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_shin_inputs(cache)

    def test_frozen_studies_use_the_exact_required_source_snapshot(self):
        from baseline.frozen_worker import required_snapshot
        from baseline.check import digest
        for study in ['mirrors', 'gh_rotation']:
            path, snapshot = required_snapshot(study)
            self.assertEqual(digest(path), snapshot['sha256'])
            self.assertTrue(snapshot['revision'])

    def test_frozen_worker_does_not_replace_the_actual_live_encoder(self):
        from baseline.frozen_worker import run_frozen
        from baseline.check import digest
        actual = ROOT / 'research/live_model.py'
        before = digest(actual)
        def check_worker(command, cwd, check):
            self.assertTrue(check)
            workspace = Path(cwd)
            self.assertNotEqual(workspace, ROOT)
            snapshot = json.loads((ROOT / 'baseline/frozen_sources/manifest.json').read_text())
            self.assertEqual(digest(workspace / 'research/live_model.py'), snapshot['sha256'])
            self.assertEqual(digest(actual), before)
            self.assertEqual((workspace / 'models').resolve(), ROOT / 'models')
        with patch('baseline.frozen_worker.subprocess.run', side_effect=check_worker), contextlib.redirect_stdout(io.StringIO()):
            run_frozen('gh_rotation', 'verify')
        self.assertEqual(digest(actual), before)

    def test_reproduction_bundle_excludes_private_captures_and_credentials(self):
        import zipfile
        from baseline.bundle import create_bundle
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            names = ['data/asl_kaggle/manifest.json', 'data/asl_kaggle/research/inputs.npz',
                     'data/asl_kaggle/research/inputs_metadata.json', 'data/asl_kaggle/research/summary.json']
            for name in [*names, 'captures/private.png', '.env']:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'test input')
            archive = directory / 'bundle.zip'
            with patch('baseline.bundle.ROOT', directory):
                create_bundle(archive)
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), set(names) | {'reproduction_inputs.json'})
