"""Validate same-coordinate joint angles and their relation to Shin directions."""
import unittest
import csv
from pathlib import Path
import tempfile

import numpy as np

from research.geometry import FINGER_JOINTS, measure_hand_features
from research.inputs import encode_angles
from research.joint_angles import joint_angles_batch
from research.shin import angle_features, combined_features
from research.studies.compare_direction_joint import matrix_for, reference_predictions


class SameCoordinateJointAngleTests(unittest.TestCase):
    def test_batch_matches_reference_and_shin_direction_dot_products(self):
        rng = np.random.default_rng(704)
        points = rng.normal(size=(8, 21, 3))
        hands = ['Left', 'Right'] * 4
        actual = joint_angles_batch(points, hands)
        expected = [encode_angles(measure_hand_features(p, h)) for p, h in zip(points, hands)]
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        # PIP uses points 5,6,7: reverse 5->6 and compare with 6->7.
        for p, row in zip(points, actual):
            directions = angle_features(p)
            first = np.array([directions[f'shin_angle_05_06_cos_{a}'] for a in 'xyz'])
            second = np.array([directions[f'shin_angle_06_07_cos_{a}'] for a in 'xyz'])
            angle = np.rad2deg(np.arccos(np.clip(-first @ second, -1, 1)))
            self.assertAlmostEqual(float(row[list(FINGER_JOINTS).index('index_pip')]), angle, places=4)

    def test_angles_survive_rigid_rotation_while_axis_directions_change(self):
        points = np.random.default_rng(705).normal(size=(2, 21, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        moved = points @ rotation.T * 2.5 + [1, 3, 8]
        np.testing.assert_allclose(joint_angles_batch(points, ['Right'] * 2),
                                   joint_angles_batch(moved, ['Right'] * 2), atol=1e-5)
        self.assertFalse(np.allclose(list(angle_features(points[0]).values()),
                                    list(angle_features(moved[0]).values())))
        original_shin = np.array([list(combined_features(p).values()) for p in points])
        moved_shin = np.array([list(combined_features(p).values()) for p in moved])
        # The replacement deliberately loses axis orientation; the combination keeps it.
        original_angles = joint_angles_batch(points, ['Right'] * 2)
        moved_angles = joint_angles_batch(moved, ['Right'] * 2)
        replacement = matrix_for(original_shin, original_angles, 'distance_joint')
        self.assertEqual(replacement.shape, (2, 205))
        np.testing.assert_allclose(replacement, matrix_for(moved_shin, moved_angles, 'distance_joint'), atol=1e-5)
        combined = matrix_for(original_shin, original_angles, 'shin_joint')
        self.assertEqual(combined.shape, (2, 835))
        self.assertFalse(np.allclose(combined, matrix_for(moved_shin, moved_angles, 'shin_joint')))
        self.assertTrue(np.isnan(joint_angles_batch(np.zeros((1, 21, 3)), ['Right'])).all())
        with self.assertRaises(ValueError):
            joint_angles_batch(points, ['Right'])

    def test_reused_predictions_are_aligned_by_id_and_cannot_cross_folds(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.csv'
            rows = [dict(model='svm', variant='shin', id='second', label='B', fold=2, prediction='A'),
                    dict(model='svm', variant='shin', id='first', label='A', fold=1, prediction='A')]
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            predictions = reference_predictions(path, 'svm', ['first', 'second'], ['A', 'B'], [1, 2], 'shin')
            self.assertEqual(predictions.tolist(), ['A', 'A'])
            with self.assertRaisesRegex(ValueError, 'folds differ'):
                reference_predictions(path, 'svm', ['first', 'second'], ['A', 'B'], [2, 1], 'shin')
