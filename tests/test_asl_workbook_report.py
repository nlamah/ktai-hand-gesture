"""Agreement denominators must include unknowns and omit unscored cells."""
import unittest
from scripts.report_asl_workbook import summarize


class WorkbookReportTests(unittest.TestCase):
    def test_unknowns_fail_and_unscored_values_are_excluded(self):
        results = [
            {'status': 'ok', 'comparisons': {'a': {'status': 'match'}, 'b': {'status': 'not_scored'}}},
            {'status': 'ok', 'comparisons': {'a': {'status': 'mismatch'}, 'b': {'status': 'unknown'}}},
            {'status': 'no_hand', 'comparisons': {'a': {'status': 'unknown'}, 'b': {'status': 'not_scored'}}},
        ]
        summary = summarize(results)
        self.assertEqual(summary['images'], 3)
        self.assertEqual(summary['detected_images'], 2)
        self.assertEqual(summary['fully_matching_images'], 1)
        self.assertEqual(summary['scored_feature_values'], 4)
        self.assertEqual(summary['feature_match_fraction_including_unknown'], 0.25)
        self.assertEqual(summary['fully_matching_fraction_all_images'], 1 / 3)
        self.assertEqual(summary['fully_matching_fraction_detected_images'], 0.5)

    def test_empty_summary_has_no_misleading_percentage(self):
        self.assertIsNone(summarize([])['feature_match_fraction_including_unknown'])
