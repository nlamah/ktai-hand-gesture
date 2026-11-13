"""Paired diagnostics retain source identity and distinguish gains from losses."""
import unittest

import pandas as pd

from RQ3.comparison_diagnostics import diagnose


class ComparisonDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def predictions():
        rows = []
        for variant, predictions in [('DS', 'ABB'), ('DSJ', 'BAB')]:
            for ident, label, fold, prediction in zip('123', 'AAB', ['1', '1', '2'], predictions):
                rows.append(dict(model='svm', variant=variant, id=ident, label=label,
                                 group='group-' + fold, fold=fold, prediction=prediction))
        return pd.DataFrame(rows)

    def test_corrected_and_new_errors_are_counted_on_the_same_sources(self):
        folds, letters, summary = diagnose(self.predictions())
        pooled = folds[0]
        self.assertEqual((pooled['fixed'], pooled['lost'], pooled['net_correct']), (1, 1, 0))
        self.assertEqual(pooled['f1_delta_pp'], 0)
        self.assertEqual(sum(row['fixed'] for row in folds[1:]), pooled['fixed'])
        self.assertEqual(sum(row['lost'] for row in letters), pooled['lost'])
        self.assertEqual(summary[0]['f1_positive_folds'], 0)

    def test_duplicate_predictions_are_rejected(self):
        frame = self.predictions()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diagnose(pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    def test_mismatched_truth_or_fold_is_rejected(self):
        for column, value in [('label', 'B'), ('fold', '2'), ('group', 'different')]:
            with self.subTest(column=column):
                frame = self.predictions()
                frame.loc[(frame.variant == 'DSJ') & (frame.id == '1'), column] = value
                with self.assertRaisesRegex(ValueError, 'Unpaired'):
                    diagnose(frame)
