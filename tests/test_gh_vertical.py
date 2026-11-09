"""Camera-y180 is a proper 3D turn, not a handedness reflection."""
import unittest
import numpy as np

from research.gh_vertical import rotate_vertical, adaptation_rows
from research.shin import combined_features
from tests.test_shin import known_points


class GHVerticalTests(unittest.TestCase):
    def test_matches_reencoding_rotated_landmarks_and_reverses_palm_normal(self):
        points = known_points()
        row = np.r_[list(combined_features(points).values()), np.arange(15)].astype(np.float32)[None]
        rotation = np.diag([-1., 1., -1.])
        turned = (points - points[0]) @ rotation.T + points[0]
        expected = np.r_[list(combined_features(turned).values()), np.arange(15)]
        result = rotate_vertical(row)
        np.testing.assert_allclose(result[0], expected, atol=2e-6)
        np.testing.assert_array_equal(result[:, :190], row[:, :190])
        np.testing.assert_array_equal(result[:, 820:], row[:, 820:])
        np.testing.assert_array_equal(rotate_vertical(result), row)
        self.assertEqual(np.linalg.det(rotation), 1.)
        normal = np.cross(points[5]-points[0], points[17]-points[0])
        turned_normal = np.cross(turned[5]-turned[0], turned[17]-turned[0])
        np.testing.assert_allclose(turned_normal, normal @ rotation.T, atol=1e-12)
        self.assertAlmostEqual(turned_normal[2], -normal[2])

    def test_depth_also_flips_and_only_GH_are_duplicated_without_calibration(self):
        rng = np.random.default_rng(41)
        X = rng.normal(size=(3, 2, 835)).astype(np.float32)
        rows, labels, weights = adaptation_rows(X[:, 0], X[:, 1], np.array(['G','H','A']), [True,False,True], None)
        self.assertEqual(len(rows), 8)
        for label in ('G','H','A'):
            self.assertAlmostEqual(weights[labels == label].sum(), 1.)
        vectors = X[:,0,190:820].reshape(-1,210,3)
        rotated = rotate_vertical(X[:,0])[:,190:820].reshape(-1,210,3)
        np.testing.assert_array_equal(rotated[:,:,0], -vectors[:,:,0])
        np.testing.assert_array_equal(rotated[:,:,1], vectors[:,:,1])
        np.testing.assert_array_equal(rotated[:,:,2], -vectors[:,:,2])


if __name__ == '__main__': unittest.main()
