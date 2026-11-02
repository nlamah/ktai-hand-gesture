"""Rotation augmentation must match landmark geometry and preserve supervision."""
import unittest
import numpy as np
from research.shin import combined_features
from research.studies.tilt_augmentation import rotate_directions, augment
from tests.test_shin import known_points


class TiltAugmentationTests(unittest.TestCase):
    def test_rotated_features_match_a_rigid_landmark_rotation(self):
        points=known_points();tail=np.arange(15,dtype=np.float32)
        x=np.r_[list(combined_features(points).values()),tail].astype(np.float32)[None,:]
        original=x.copy();a=np.deg2rad(15)
        rotation=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        expected=np.r_[list(combined_features((points-points[0])@rotation.T).values()),tail]
        actual=rotate_directions(x,15)
        np.testing.assert_allclose(actual[0],expected,atol=2e-7)
        np.testing.assert_array_equal(actual[:,:190],x[:,:190])
        np.testing.assert_array_equal(actual[:,820:],x[:,820:])
        np.testing.assert_array_equal(x,original)
        np.testing.assert_allclose(np.linalg.norm(actual[:,190:820].reshape(-1,3),axis=1),1,atol=2e-7)

    def test_weights_preserve_total_weight_per_original_and_labels(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(2,835)).astype(np.float32);y=np.array(['C','O'])
        xx,yy,w=augment(x,y)
        self.assertEqual(xx.shape,(6,835));self.assertEqual(yy.tolist(),['C','O','C','O','C','O'])
        np.testing.assert_array_equal(xx[:2],x)
        np.testing.assert_allclose(w.reshape(3,2).sum(axis=0),np.ones(2))
        for k in range(3):
            np.testing.assert_array_equal(xx[k*2:k*2+2,:190],x[:,:190])
            np.testing.assert_array_equal(xx[k*2:k*2+2,820:],x[:,820:])

    def test_input_schema_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):rotate_directions(np.zeros((2,931)),15)
        with self.assertRaises(ValueError):rotate_directions(np.full((1,835),np.nan),15)

    def test_selective_C_augmentation_leaves_other_examples_single_and_weighted_normally(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(4,835)).astype(np.float32);y=np.array(['A','C','D','C'])
        xx,yy,w=augment(x,y,letters=['C'])
        self.assertEqual(xx.shape,(8,835));self.assertEqual(yy.tolist(),['A','C','D','C','C','C','C','C'])
        np.testing.assert_array_equal(xx[:4],x)
        np.testing.assert_array_equal(w[[0,2]],np.ones(2))
        np.testing.assert_allclose([w[[1,4,6]].sum(),w[[3,5,7]].sum()],np.ones(2))
        np.testing.assert_array_equal(xx[4:6,820:],x[[1,3],820:])

    def test_selective_fit_preserves_original_scaling_gamma_and_balanced_class_priors(self):
        from research.studies.tilt_augmentation import fit_candidate
        from research.studies.compare_angles import classifier
        rng=np.random.default_rng(5);x=rng.normal(size=(12,835)).astype(np.float32);y=np.array(['A']*8+['C']*4)
        x[8:,820:]+=10
        baseline=classifier('svm','shin_angles',len(y)).fit(x,y)
        candidate=fit_candidate(x,y,letters=['C'])
        np.testing.assert_allclose(candidate[-1].class_weight_,baseline[-1].class_weight_)
        self.assertAlmostEqual(candidate[-1]._gamma,baseline[-1]._gamma,places=12)
        np.testing.assert_allclose(candidate[0].transform(x),baseline[0].transform(x),atol=1e-7)

    def test_y_rotation_and_four_variant_GH_weights_match_rigid_geometry(self):
        points=known_points();tail=np.arange(15,dtype=np.float32)
        row=np.r_[list(combined_features(points).values()),tail].astype(np.float32)
        x=np.tile(row,(3,1));y=np.array(['G','H','O'])
        xx,yy,w=augment(x,y,letters=['G','H'],axis='y',angles=(0,-90,90,180))
        self.assertEqual(xx.shape,(9,835))
        self.assertEqual(yy.tolist(),['G','H','O','G','H','G','H','G','H'])
        np.testing.assert_allclose([w[[0,3,5,7]].sum(),w[[1,4,6,8]].sum(),w[2]],[1,1,1])
        for degrees in (-90,90,180):
            a=np.deg2rad(degrees);c,s=np.cos(a),np.sin(a)
            rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            expected=np.r_[list(combined_features((points-points[0])@rotation.T).values()),tail]
            np.testing.assert_allclose(rotate_directions(x[:1],degrees,'y')[0],expected,atol=2e-7)
        np.testing.assert_array_equal(xx[:3],x)
        with self.assertRaises(ValueError):augment(x,y,angles=(15,0))
        with self.assertRaises(ValueError):rotate_directions(x,90,'invalid')
