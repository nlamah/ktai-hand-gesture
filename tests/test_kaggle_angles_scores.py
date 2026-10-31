"""Ensure the isolated comparison cannot accidentally include category columns."""
import unittest
import numpy as np
from scripts.compare_kaggle_angles_scores import variant_inputs,classifier


class KaggleAnglesScoresTests(unittest.TestCase):
    def test_variants_select_only_the_documented_feature_blocks(self):
        X=np.tile(np.arange(931,dtype=float),(2,1))
        expected={'shin':820,'shin_angles':835,'shin_excel':846,'shin_angles_excel':861}
        for variant,width in expected.items():
            result=variant_inputs(X,variant)
            self.assertEqual(result.shape,(2,width))
            np.testing.assert_array_equal(result[:,:820],X[:,:820])
            self.assertFalse(set(range(820,890))&set(result[0,820:]))
        np.testing.assert_array_equal(variant_inputs(X,'shin_angles')[0,820:],np.arange(916,931))
        np.testing.assert_array_equal(variant_inputs(X,'shin_excel')[0,820:],np.arange(890,916))
        np.testing.assert_array_equal(variant_inputs(X,'shin_angles_excel')[0,820:],np.arange(890,931))

    def test_both_classifiers_fit_the_isolated_schemas(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(30,931));y=['A','B']*15
        for name in ('svm','lightgbm'):
            for variant in ('shin_angles','shin_excel','shin_angles_excel'):
                matrix=variant_inputs(X,variant)
                model=classifier(name,variant,len(y)).fit(matrix,y)
                self.assertEqual(model.predict(matrix[:2]).shape,(2,))
