"""Reference calculations and actual invariances of the paper-derived features."""
from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gesturelab.features import paper_shin2021 as paper
from gesturelab.features.training_inputs import PAPER_VARIANTS, training_input


def known_points():
    points = np.array([[i, i * i * .01, i * .02] for i in range(21)], dtype=float)
    points[:6] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [1, 1, 1]]
    return points


class PaperFeatureTests(unittest.TestCase):
    def setUp(self):
        self.points = known_points()

    def test_table2_literal_pairs_and_each_row_count(self):
        expected = [15, 18, 17, 16, 16, 14, 13, 12, 12, 10, 9, 8, 8, 6, 5, 4, 4, 2, 1, 0, 0]
        counts = Counter(i for i, j in paper.DISTANCE_PAIRS)
        self.assertEqual([counts[i] for i in range(21)], expected)
        self.assertEqual(len(paper.DISTANCE_PAIRS), 190)
        self.assertEqual(len(set(paper.DISTANCE_PAIRS)), 190)
        self.assertNotIn((0, 4), paper.DISTANCE_PAIRS)
        self.assertIn((0, 5), paper.DISTANCE_PAIRS)
        self.assertEqual(len(paper.ALL_PAIRS), 210)

    def test_euclidean_distance_and_population_zscore(self):
        self.points[2] = [3, 4, 0]
        raw = paper.raw_distances(self.points)
        self.assertEqual(raw['shin_dist_00_02'], 5.0)
        normalized = paper.distance_features(self.points)
        values = np.array(list(raw.values()))
        np.testing.assert_allclose(list(normalized.values()), (values - values.mean()) / values.std(ddof=0))
        self.assertAlmostEqual(np.mean(list(normalized.values())), 0)
        self.assertAlmostEqual(np.std(list(normalized.values())), 1)

    def test_axis_cosines_and_directed_pair_signs_are_not_joint_angles(self):
        values = paper.angle_features(self.points)
        self.assertEqual([values[f'shin_angle_00_01_cos_{axis}'] for axis in 'xyz'], [1, 0, 0])
        self.assertEqual([values[f'shin_angle_00_02_cos_{axis}'] for axis in 'xyz'], [0, 1, 0])
        self.assertEqual(values['shin_angle_00_04_cos_x'], -1)
        self.assertAlmostEqual(values['shin_angle_01_02_cos_x'], -1 / np.sqrt(2))
        self.assertAlmostEqual(values['shin_angle_01_02_cos_y'], 1 / np.sqrt(2))
        for axis in 'xyz':
            self.assertAlmostEqual(values[f'shin_angle_00_05_cos_{axis}'], 1 / np.sqrt(3))

    def test_counts_order_and_combined_values_use_only_image_landmarks(self):
        sample = {'image_landmarks': self.points, 'world_landmarks': 'must not be read'}
        distance, angle, combined = [training_input(v).features(sample, None) for v in PAPER_VARIANTS]
        self.assertEqual(tuple(map(len, (distance, angle, combined))), (190, 630, 820))
        self.assertEqual(list(combined), list(distance) + list(angle))
        self.assertEqual(combined, {**distance, **angle})
        self.assertEqual(list(distance)[0], 'shin_dist_00_02')
        self.assertEqual(list(angle)[:3], ['shin_angle_00_01_cos_' + a for a in 'xyz'])
        np.testing.assert_array_equal(self.points, known_points())

    def test_translation_and_positive_uniform_scale_leave_features_unchanged(self):
        original = paper.combined_features(self.points)
        moved = paper.combined_features(self.points * 2.5 + [1, -2, 4])
        np.testing.assert_allclose(list(original.values()), list(moved.values()), atol=1e-12)

    def test_rotation_preserves_distances_but_changes_axis_cosines(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = self.points @ rotation.T
        np.testing.assert_allclose(list(paper.distance_features(self.points).values()),
                                   list(paper.distance_features(rotated).values()), atol=1e-12)
        original = paper.angle_features(self.points)
        changed = paper.angle_features(rotated)
        self.assertEqual(original['shin_angle_00_01_cos_x'], 1)
        self.assertEqual(changed['shin_angle_00_01_cos_x'], 0)
        self.assertEqual(changed['shin_angle_00_01_cos_y'], 1)

    def test_invalid_landmarks_zero_vectors_and_zero_variance_are_explicit(self):
        for bad in (np.zeros((20, 3)), np.zeros((21, 2)), np.full((21, 3), np.nan), np.full((21, 3), np.inf)):
            with self.subTest(shape=bad.shape), self.assertRaisesRegex(ValueError, 'finite 21 x 3'):
                paper.combined_features(bad)
        with self.assertRaisesRegex(ValueError, 'variance'):
            paper.distance_features(np.zeros((21, 3)))
        self.points[2] = self.points[1]
        with self.assertRaisesRegex(ValueError, 'zero-length'):
            paper.angle_features(self.points)
        self.assertEqual(len(paper.distance_features(self.points)), 190)


if __name__ == '__main__':
    unittest.main()
