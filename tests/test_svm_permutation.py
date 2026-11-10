"""Reused RBF kernels must predict the same labels as ordinary SVC evaluation."""
import unittest
import numpy as np
from research.feature_study import estimator,permute_group
from research.svm_permutation import RBFPermutation


class RBFPermutationTests(unittest.TestCase):
    def test_raw_and_angle_pipeline_predictions_match_after_group_permutation(self):
        rng=np.random.default_rng(306)
        for variant,width in [('DE',260),('DEJ',275)]:
            x=rng.normal(size=(90,width)).astype(np.float32)
            labels=np.array(list('ABCDE')*18)
            model=estimator('svm',variant,len(x)).fit(x,labels)
            validation=(x[:17]+rng.normal(0,.4,(17,width))).astype(np.float32)
            orders=[rng.permutation(len(validation)) for _ in range(3)]
            with RBFPermutation(model,validation,chunk=7) as fast:
                np.testing.assert_array_equal(fast.baseline(),model.predict(validation))
                for columns in ([0,1,2],list(range(190,width))):
                    predicted=fast.group_predictions(columns,orders)
                    for pred,order in zip(predicted,orders):
                        np.testing.assert_array_equal(pred,model.predict(permute_group(validation,columns,order)))


if __name__=='__main__':unittest.main()
