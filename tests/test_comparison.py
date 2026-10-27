"""Ensure feature variants are compared using the same photos and ground truth."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesturelab.training.comparison import _pair_predictions, _prediction_scores


class PairedComparisonTests(unittest.TestCase):
    def predictions(self):
        return {
            'without_features': pd.DataFrame([
                {'id': 'photo1', 'label': 'A', 'prediction': 'A'},
                {'id': 'photo2', 'label': 'B', 'prediction': 'A'},
            ]),
            'with_features': pd.DataFrame([
                {'id': 'photo2', 'label': 'B', 'prediction': 'B'},
                {'id': 'photo1', 'label': 'A', 'prediction': 'B'},
            ]),
        }

    def test_pairing_uses_photo_ids_instead_of_csv_row_order(self):
        paired = _pair_predictions(self.predictions()).set_index('id')
        self.assertTrue(paired.loc['photo1', 'first_correct'])
        self.assertFalse(paired.loc['photo1', 'second_correct'])
        self.assertFalse(paired.loc['photo2', 'first_correct'])
        self.assertTrue(paired.loc['photo2', 'second_correct'])

    def test_missing_or_substituted_photos_are_rejected(self):
        for other in ('missing', 'substituted'):
            predictions = self.predictions()
            if other == 'missing':
                predictions['with_features'] = predictions['with_features'].iloc[:1]
            else:
                predictions['with_features'].loc[0, 'id'] = 'another_photo'
            with self.subTest(case=other), self.assertRaisesRegex(ValueError, 'photo IDs'):
                _pair_predictions(predictions)

    def test_duplicate_photo_ids_are_rejected_in_either_variant(self):
        for variant in self.predictions():
            predictions = self.predictions()
            predictions[variant] = pd.concat([predictions[variant], predictions[variant].iloc[:1]])
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'photo IDs'):
                _pair_predictions(predictions)

    def test_incomplete_test_macro_scores_exclude_absent_truth_but_keep_known_label_coverage(self):
        plan = {'labels': ['A', 'B'], 'missing_test_labels': ['B']}
        predictions = pd.DataFrame({'label': ['A'], 'prediction': ['A']})
        final = _prediction_scores(predictions, plan, final_test=True)
        self.assertEqual(final['macro_f1'], 1.0)
        self.assertEqual(final['balanced_accuracy'], 1.0)
        predictions['prediction'] = 'B'
        wrong = _prediction_scores(predictions, plan, final_test=True)
        self.assertEqual(wrong['macro_f1'], 0.0)
        self.assertEqual(wrong['coverage'], 1.0)

    def test_conflicting_ground_truth_for_same_photo_is_rejected(self):
        predictions = self.predictions()
        predictions['with_features'].loc[0, 'label'] = 'C'
        with self.assertRaisesRegex(ValueError, 'Ground-truth labels differ'):
            _pair_predictions(predictions)


if __name__ == '__main__':
    unittest.main()
