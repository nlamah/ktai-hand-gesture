"""Single-factor G/H augmentation keeps every source's weight and model settings."""
import unittest
import numpy as np

from research.gh_vertical import rotate_vertical
from research.studies.compare_angles import classifier
from research.studies.gh_rotation_standalone import augmented_rows, fit_rotated


class StandaloneGHRotationTests(unittest.TestCase):
    def test_only_GH_are_turned_and_original_source_weights_are_preserved(self):
        rng=np.random.default_rng(104)
        X=rng.normal(size=(4,835)).astype(np.float32);y=np.array(['G','A','H','G'])
        rows,labels,weights=augmented_rows(X,y)
        np.testing.assert_array_equal(rows[:4],X)
        np.testing.assert_array_equal(rows[4:],rotate_vertical(X[[0,2,3]]))
        np.testing.assert_array_equal(labels,['G','A','H','G','G','H','G'])
        np.testing.assert_array_equal(weights[:4],[.5,1.,.5,.5])
        for original,copy in ((0,4),(2,5),(3,6)):
            self.assertEqual(weights[original]+weights[copy],1.)
        self.assertEqual((labels=='A').sum(),1)

    def test_candidate_reuses_original_preprocessing_kernel_bandwidth_and_class_priors(self):
        rng=np.random.default_rng(105)
        X=rng.normal(size=(15,835)).astype(np.float32);y=np.array(['A']*9+['G']*4+['H']*2)
        X[y=='G',820:]+=20
        baseline=classifier('svm','shin_angles',len(y)).fit(X,y)
        candidate=fit_rotated(baseline,X,y)
        np.testing.assert_array_equal(candidate[0].transform(X),baseline[0].transform(X))
        self.assertEqual(candidate[-1]._gamma,baseline[-1]._gamma)
        np.testing.assert_array_equal(candidate[-1].class_weight_,baseline[-1].class_weight_)
        self.assertEqual(len(candidate),2)
        self.assertEqual(type(candidate[-1]).__name__,'SVC')


if __name__=='__main__':unittest.main()
