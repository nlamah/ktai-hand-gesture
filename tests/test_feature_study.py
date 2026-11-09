"""Factorial isolation, grouped shuffling and train-only angle preprocessing."""
import unittest

import numpy as np

from research.feature_study import VARIANTS,source_columns,matrix,category_columns,permutation_groups,permute_group,estimator
from research.inputs import LABELS
from research.references import load_reference_workbook
from research.studies.feature_factorial import selected_rows
from pathlib import Path


class FeatureStudyTests(unittest.TestCase):
    def test_eight_variants_exclude_workbook_scores_and_keep_exact_source_order(self):
        widths=[820,190,890,260,835,205,905,275]
        x=np.tile(np.arange(931,dtype=np.float32),(3,1))
        for variant,width in zip(VARIANTS,widths):
            columns=source_columns(variant)
            self.assertEqual(len(columns),width)
            self.assertEqual(len(set(columns)),width)
            self.assertFalse(set(columns)&set(range(890,916)))
            np.testing.assert_array_equal(matrix(x,variant)[0],columns)
            np.testing.assert_array_equal(columns[:190],np.arange(190))
        with self.assertRaises(ValueError):matrix(x,'E')

    def test_category_groups_move_whole_one_hot_vectors_and_leave_other_columns_fixed(self):
        refs=load_reference_workbook(Path('assets/asl_feature_references.xlsx').read_bytes())['A']
        source=category_columns(refs);self.assertEqual(len(source),19)
        x=np.arange(4*890,dtype=float).reshape(4,890)
        groups=permutation_groups('DSE',refs)
        key='check_flexion_index_pip';columns=groups[key]
        order=np.array([2,0,3,1]);altered=permute_group(x,columns,order)
        np.testing.assert_array_equal(altered[:,columns],x[order][:,columns])
        rest=[i for i in range(890) if i not in columns]
        np.testing.assert_array_equal(altered[:,rest],x[:,rest])
        self.assertEqual(len(columns),4)
        self.assertNotIn('block_J',groups)
        complete=permutation_groups('DSEJ',refs)
        self.assertIn('family_E_spread',complete)
        self.assertEqual(len(complete['family_J_index']),3)
        self.assertFalse(any(g.startswith('check_') for g in complete))
        with self.assertRaises(ValueError):permute_group(x,columns,[0,0,1,2])

    def test_shared_permutation_subset_is_reproducible_and_entirely_held_out(self):
        y=np.array(LABELS*8);ids=np.array([f'id{i}' for i in range(len(y))])
        tr=np.arange(4*26);te=np.arange(4*26,len(y));splits=[(1,tr,te)]
        a=selected_rows(y,ids,splits)[1];b=selected_rows(y,ids,splits)[1]
        np.testing.assert_array_equal(a,b)
        self.assertFalse(set(a)&set(tr));self.assertTrue(set(a)<=set(te))

    def test_only_world_angles_are_standardized_using_training_rows(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(52,905)).astype(np.float32)
        x[:,890:]+=150;y=np.array(LABELS*2)
        model=estimator('svm','DSEJ',52).fit(x,y)
        transformed=model[0].transform(x)
        np.testing.assert_array_equal(transformed[:,:890],x[:,:890])
        np.testing.assert_allclose(transformed[:,890:].mean(axis=0),0,atol=1e-5)
        scaler=model[0].named_transformers_['angles'].named_steps['standardscaler']
        mean=scaler.mean_.copy();model.predict(x+10000)
        np.testing.assert_array_equal(scaler.mean_,mean)
        self.assertEqual(estimator('svm','DE',52).__class__.__name__,'SVC')


if __name__=='__main__':unittest.main()
