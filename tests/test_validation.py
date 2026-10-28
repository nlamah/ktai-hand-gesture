"""Independent numerical oracles and deliberately broken research artifacts."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesturelab.validation.metrics import score_rows, fold_summary, numeric_difference
from gesturelab.validation.features import explicit_values, invariant_checks, invalid_checks, composition_checks, paper_table_checks
from gesturelab.validation.pipeline import fit_transform_probe, metadata_probe
from gesturelab.validation.report import AuditReport, new_output
from gesturelab.validation.splits import canonical_assignment, _fold_check, audit_pixels
from gesturelab.validation.scores import prediction_membership
from gesturelab.validation.runner import run_audit
from gesturelab.validation.reproduction import reproduce_variant
from gesturelab.validation.report import read_json, sha256, write_json
from gesturelab.training.experiment_plan import prepare_experiment
from gesturelab.training.fitting import train_variant
from scripts.demo import make_demo


class IndependentMetricTests(unittest.TestCase):
    def test_hand_counted_tp_fp_fn_and_confusion(self):
        rows = [dict(label=a, prediction=b) for a, b in zip('AABBC', ['A', 'B', 'B', '__unknown__', 'A'])]
        result = score_rows(rows, ['A', 'B', 'C'])
        self.assertEqual(result['accuracy'], 2 / 5)
        self.assertEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['coverage'], 4 / 5)
        self.assertEqual(result['confusion'], [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0]])
        self.assertEqual([(r['tp'], r['fp'], r['fn'], r['support']) for r in result['per_class']],
                         [(1, 1, 1, 2), (1, 1, 1, 2), (0, 0, 1, 1)])

    def test_pooled_fold_mean_and_sample_sd_are_different_quantities(self):
        rows = [dict(label=a, prediction=b, fold=f) for a, b, f in
                [('A', 'A', 0), ('A', 'B', 1), ('B', 'A', 1), ('B', 'A', 1)]]
        pooled = score_rows(rows, ['A', 'B'])
        folds = fold_summary(rows, ['A', 'B'])['summary']
        self.assertEqual(pooled['accuracy'], .25)
        self.assertEqual(folds['accuracy_mean'], .5)
        self.assertAlmostEqual(folds['accuracy_std'], math.sqrt(.5))
        self.assertEqual(pooled['macro_f1'], .2)
        self.assertEqual(folds['macro_f1_mean'], .25)

    def test_missing_class_policy_and_outside_stress_alphabet(self):
        rows = [dict(label='A', prediction='A')]
        self.assertEqual(score_rows(rows, ['A', 'B'])['macro_f1'], .5)
        self.assertEqual(score_rows(rows, ['A', 'B'], supported_macro=True)['macro_f1'], 1)
        rows[0]['prediction'] = 'Z'
        result = score_rows(rows, ['A', 'B'], prediction_labels=list('ABZ'))
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['coverage'], 1)
        self.assertEqual(result['confusion'][0], [0, 0, 1, 0])
        self.assertIsNotNone(numeric_difference(.5, 'nan'))


class ValidationContractTests(unittest.TestCase):
    def test_mathematical_contracts_and_actual_fit_transform_probe(self):
        self.assertEqual(explicit_values()['actual']['distance_3_4_5'], 5)
        self.assertEqual(explicit_values()['actual']['joint_right'], 90)
        self.assertTrue(invariant_checks())
        self.assertEqual(len(invalid_checks()), 6)
        self.assertEqual(composition_checks()['all_features'], 1027)
        self.assertEqual(paper_table_checks()['direction_pairs'], 210)
        self.assertEqual(fit_transform_probe()['fit_inputs'], [[1., 3.], [101., 103.]])
        self.assertEqual(metadata_probe()['shin2021_combined'], 820)

    def test_metadata_only_plan_changes_do_not_change_content_assignments(self):
        plan = dict(labels=['B', 'A'], grouping='session', development_ids=['b', 'a'], test_ids=['t'],
                    folds=[dict(train_ids=['b'], validation_ids=['a'])], created_at='old', model='svm')
        other = json.loads(json.dumps(plan))
        other.update(created_at='new', model='lightgbm', extra='irrelevant')
        other['labels'].reverse()
        self.assertEqual(canonical_assignment(plan), canonical_assignment(other))
        other['folds'][0]['train_ids'] = ['a']
        self.assertNotEqual(canonical_assignment(plan), canonical_assignment(other))

    def test_missing_evidence_remains_unknown_and_protected_output_is_rejected(self):
        report = AuditReport('test')
        report.add('missing', 'No file.', 'Evidence', 'Absent')
        self.assertEqual(report.checks[0]['status'], 'not_checked')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                new_output(Path(tmp) / 'audit', [Path(tmp)])

    def test_pixel_duplicates_fail_even_when_photo_ids_differ(self):
        context = dict(project=Path('unused'), plan=dict(development_ids=['a', 'b'], test_ids=[],
            folds=[dict(train_ids=['a'], validation_ids=['b'])]), records={
            'a': dict(pixel_sha256='identical'), 'b': dict(pixel_sha256='identical')})
        report = AuditReport('pixels')
        audit_pixels(report, context)
        self.assertEqual(report.checks[0]['status'], 'failed')

    def test_duplicate_oof_rows_and_wrong_truth_are_detected(self):
        plan = dict(labels=['A', 'B'], development_ids=['a', 'b'], test_ids=[],
                    folds=[dict(train_ids=['b'], validation_ids=['a']), dict(train_ids=['a'], validation_ids=['b'])])
        records = {'a': dict(label='A'), 'b': dict(label='B')}
        rows = [dict(id='a', label='B', prediction='A', fold='0')] * 2
        result = prediction_membership(rows, plan, records)
        self.assertTrue(result['missing_pairs'])
        self.assertTrue(result['extra_pairs'])
        self.assertTrue(result['bad_labels'])


class ValidationIntegrationTests(unittest.TestCase):
    def test_audit_replay_and_deliberately_wrong_saved_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_demo(root / 'synthetic')
            experiment = prepare_experiment(project, root / 'source', folds=3, grouping='participant')
            folder = train_variant(experiment, 'finger_features_only')
            before = {p: sha256(p) for source in (project, experiment) for p in source.rglob('*') if p.is_file()}
            audit = run_audit([experiment], root / 'audit')
            self.assertNotIn('failed', read_json(audit / 'audit.json')['statuses'])
            replay = reproduce_variant(experiment, 'finger_features_only', root / 'replay')
            self.assertEqual(read_json(replay / 'audit.json')['statuses'], {'passed': 3})
            self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))
            info = read_json(folder / 'training.json')
            info['cv_pooled']['accuracy'] = -1
            write_json(folder / 'training.json', info)
            broken = run_audit([experiment], root / 'broken_score')
            checks = read_json(broken / 'audit.json')['checks']
            self.assertTrue(any(c['name'].endswith('scores.training_pooled') and c['status'] == 'failed' for c in checks))
            self.assertEqual(read_json(folder / 'training.json')['cv_pooled']['accuracy'], -1)


if __name__ == '__main__':
    unittest.main()
