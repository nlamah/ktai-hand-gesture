"""Direction-preserving G/H rolls, source weighting and guarded 26-class routing."""
import unittest
import numpy as np

from research.gh_view_adapter import INDEX_DIRECTION, GHViewGuard, roll_about_index, adaptation_rows, horizontal_mask
from research.inputs import LABELS
from research.shin import combined_features
from tests.test_shin import known_points


class FakeClassifier:
    classes_=np.array(LABELS)
    n_features_in_=835
    def __init__(self,labels):self.labels=np.array(labels)
    def predict(self,X):return self.labels[np.asarray(X)[:,0].astype(int)]
    def decision_function(self,X):
        result=np.zeros((len(X),26))
        for i,label in enumerate(self.predict(X)):result[i,LABELS.index(label)]=5
        return result


class GHViewAdapterTests(unittest.TestCase):
    def test_roll_is_proper_rotation_preserving_index_direction_distances_and_intrinsic_angles(self):
        points=known_points();row=np.r_[list(combined_features(points).values()),np.arange(15)].astype(np.float32)[None,:]
        original=row.copy();actual=roll_about_index(row)
        axis=points[8]-points[5];axis/=np.linalg.norm(axis)
        rotation=2*np.outer(axis,axis)-np.eye(3)
        self.assertAlmostEqual(np.linalg.det(rotation),1)
        expected=np.r_[list(combined_features(points@rotation.T).values()),np.arange(15)]
        np.testing.assert_allclose(actual[0],expected,atol=2e-6)
        np.testing.assert_allclose(actual[:,190:820].reshape(-1,210,3)[:,INDEX_DIRECTION],row[:,190:820].reshape(-1,210,3)[:,INDEX_DIRECTION],atol=2e-6)
        np.testing.assert_array_equal(actual[:,:190],row[:,:190]);np.testing.assert_array_equal(actual[:,820:],row[:,820:])
        np.testing.assert_allclose(roll_about_index(actual),row,atol=3e-6)
        np.testing.assert_array_equal(row,original)

    def test_only_GH_receive_rolls_and_each_source_keeps_its_declared_total_weight(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(3,2,835)).astype(np.float32)
        cal={'features':rng.normal(size=(2,2,835)).astype(np.float32),'labels':np.array(['G','H']),
             'status':np.array([['ok','ok'],['ok','no_hand']])}
        rows,y,w=adaptation_rows(x[:,0],x[:,1],np.array(['G','H','A']),[True,False,True],cal)
        self.assertTrue(np.isfinite(rows).all())
        self.assertAlmostEqual(w.sum(),43)
        self.assertAlmostEqual(w[y=='G'].sum(),21);self.assertAlmostEqual(w[y=='H'].sum(),21)
        self.assertAlmostEqual(w[y=='A'].sum(),1)
        self.assertEqual((y=='A').sum(),2)

    def test_guard_does_not_turn_vertical_or_other_letter_proposals_into_GH(self):
        X=np.zeros((4,835));X[:,0]=np.arange(4)
        directions=X[:,190:820].reshape(4,210,3)
        directions[:,INDEX_DIRECTION]=[[1,0,0],[0,1,0],[1,0,0],[0,0,1]]
        base=FakeClassifier(['A','U','P','Q']);adapted=FakeClassifier(['G','H','C','G'])
        model=GHViewGuard(base,adapted)
        np.testing.assert_array_equal(horizontal_mask(X),[True,False,True,False])
        self.assertEqual(model.predict(X).tolist(),['G','U','P','Q'])
        scores=model.decision_function(X)
        self.assertEqual([LABELS[i] for i in scores.argmax(axis=1)],['G','U','P','Q'])

    def test_router_roundtrips_as_a_live_pipeline_with_actual_sklearn_estimators(self):
        from copy import deepcopy
        from pathlib import Path
        import tempfile
        import joblib,sklearn
        from sklearn.pipeline import make_pipeline
        from research.studies.compare_angles import classifier
        from research.live_model import BUNDLE_VERSION,SHIN_VERSION,ANGLE_COLUMNS,load_bundle,predict_letter,preprocessed_inputs,model_inputs
        from tests.test_live_model import detection
        rng=np.random.default_rng(8);X=rng.normal(size=(52,835)).astype(np.float32)
        baseline=classifier('svm','shin_angles',52).fit(X,LABELS*2)
        guard=GHViewGuard(baseline[-1],deepcopy(baseline[-1]))
        model=make_pipeline(deepcopy(baseline[0]),guard)
        np.testing.assert_array_equal(model.predict(X),baseline.predict(X))
        bundle={'version':BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
                'sklearn_version':sklearn.__version__,'estimator':model}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(bundle,path);loaded=load_bundle(path)
            result=predict_letter(loaded,detection());self.assertIn(result['letter'],LABELS)
            self.assertEqual(preprocessed_inputs(loaded,model_inputs(loaded,detection())).shape,(1,835))


if __name__=='__main__':unittest.main()
