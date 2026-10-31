"""Check block weighting and fold-local continuous preprocessing."""
import unittest
import numpy as np
from scripts.tune_kaggle_feature_weights import weighted_input,scale_angles,WEIGHTS,bounded_indices


class KaggleFeatureWeightsTests(unittest.TestCase):
    def test_weighting_leaves_shin_unchanged_and_weights_only_named_blocks(self):
        X=np.arange(2*916,dtype=float).reshape(2,916)/100
        angles=np.ones((2,15))
        actual=weighted_input(X,angles,(.25,.5,2))
        self.assertEqual(actual.shape,(2,931))
        np.testing.assert_array_equal(actual[:,:820],X[:,:820])
        np.testing.assert_array_equal(actual[:,820:890],X[:,820:890]*.5)
        np.testing.assert_array_equal(actual[:,890:916],X[:,890:916]*2)
        np.testing.assert_array_equal(actual[:,916:],angles*.25)
        self.assertEqual(WEIGHTS,(.25,.5,1.,2.))

    def test_validation_values_do_not_set_angle_center_or_imputation(self):
        train=np.array([[10.,np.nan],[20.,30.],[30.,50.]])
        validation=np.array([[10000.,np.nan]])
        a,b=scale_angles(train,validation)
        np.testing.assert_allclose(a.mean(axis=0),0,atol=1e-12)
        self.assertAlmostEqual(b[0,1],0)
        self.assertGreater(b[0,0],100)

    def test_bounded_search_sampling_is_reproducible_and_stays_in_partition(self):
        labels=np.array(['A']*200+['B']*200)
        partition=np.r_[np.arange(175),np.arange(200,375)]
        selected=bounded_indices(partition,labels,42)
        np.testing.assert_array_equal(selected,bounded_indices(partition,labels,42))
        self.assertEqual(len(selected),300)
        self.assertTrue(set(selected)<=set(partition))
        self.assertEqual(int((labels[selected]=='A').sum()),150)
