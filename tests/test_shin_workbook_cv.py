"""Workbook model inputs must not depend on the true target label."""
from pathlib import Path
import unittest
import numpy as np

from gesturelab.features.asl_excel_review import compare_sample,load_reference_workbook,measure_hand_features
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS
from scripts.cross_validate_shin_workbook import encode_additions, encode_angles, angle_estimator, ANGLE_COLUMNS
from tests.test_asl_threshold_tuning import _synthetic_hand


class ShinWorkbookInputTests(unittest.TestCase):
    def test_all_letter_scores_reconcile_with_reference_comparison(self):
        refs=load_reference_workbook((Path(__file__).resolve().parents[1]/'assets/asl_feature_references.xlsx').read_bytes())
        world=_synthetic_hand()
        measurements=measure_hand_features(world,'Right')
        categories,scores=encode_additions(measurements,PARAMETER_DEFAULTS,refs)
        self.assertEqual(categories.sum(),19)
        self.assertEqual(len(scores),26)
        for i,label in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            result=compare_sample({'id':'synthetic','label':label,'status':'ok',
                                   'world_landmarks':world,'handedness':'Right'},refs,PARAMETER_DEFAULTS)
            denominator=result['matches']+result['mismatches']+result['unknown']
            self.assertAlmostEqual(scores[i],result['matches']/denominator)
        self.assertTrue(np.all(np.isfinite(categories)))

    def test_changing_reference_alternatives_does_not_change_measured_categories(self):
        import copy
        refs=load_reference_workbook((Path(__file__).resolve().parents[1]/'assets/asl_feature_references.xlsx').read_bytes())
        modified=copy.deepcopy(refs)
        modified['A']['flexion_index_pip']['allowed']=['bent','partly_bent','extended']
        measurements=measure_hand_features(_synthetic_hand(),'Right')
        old,old_scores=encode_additions(measurements,PARAMETER_DEFAULTS,refs)
        new,new_scores=encode_additions(measurements,PARAMETER_DEFAULTS,modified)
        np.testing.assert_array_equal(old,new)
        self.assertGreaterEqual(new_scores[0],old_scores[0])
        np.testing.assert_array_equal(old_scores[1:],new_scores[1:])


class JointAngleInputTests(unittest.TestCase):
    def test_degree_columns_have_fixed_order_and_preserve_unknown(self):
        measurements=measure_hand_features(_synthetic_hand(),'Right')
        values=encode_angles(measurements)
        self.assertEqual(values.shape,(15,))
        self.assertAlmostEqual(values[ANGLE_COLUMNS.index('index_pip_angle_deg')],180)
        measurements['thumb_mcp_signed_deviation_deg']=None
        self.assertTrue(np.isnan(encode_angles(measurements)[-1]))

    def test_only_degree_inputs_are_scaled_using_training_statistics(self):
        base=np.arange(12,dtype=float).reshape(6,2)/10
        angles=np.tile(np.arange(15,dtype=float),(6,1))+np.arange(6)[:,None]
        angles[0,0]=np.nan
        X=np.c_[base,angles]
        model=angle_estimator(2,17).fit(X,['A','A','A','B','B','B'])
        transformer=model.steps[0][1]
        fitted=transformer.named_transformers_['angles']
        self.assertAlmostEqual(fitted.steps[0][1].statistics_[0],3)
        means=fitted.steps[1][1].mean_.copy()
        validation=X.copy();validation[:,2:]+=10000
        changed=transformer.transform(validation)
        np.testing.assert_array_equal(fitted.steps[1][1].mean_,means)
        np.testing.assert_array_equal(changed[:,:2],base)
        self.assertTrue(np.isfinite(changed).all())
