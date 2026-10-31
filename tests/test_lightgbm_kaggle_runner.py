"""Permutation and ablation groups must respect the saved model schema."""
import unittest
from pathlib import Path
import numpy as np
from scripts.compare_kaggle_lightgbm import feature_groups,model_for
from gesturelab.features.asl_excel_review import load_reference_workbook


class LightGBMKaggleRunnerTests(unittest.TestCase):
    def test_groups_preserve_one_hot_blocks_and_cover_all_additions(self):
        root=Path(__file__).resolve().parents[1]
        refs=load_reference_workbook((root/'assets/asl_feature_references.xlsx').read_bytes())
        groups=feature_groups(refs)
        self.assertEqual(groups['all_extra'],list(range(820,931)))
        families=['thumb','finger_flexion','spread','crossing','contact']
        combined=[i for name in families for i in groups[name]]
        self.assertEqual(len(combined),len(set(combined)))
        self.assertEqual(set(combined),set(range(820,890))|set(range(916,931)))
        self.assertTrue(all(820<=i<931 for values in groups.values() for i in values))

    def test_both_classifiers_accept_fixed_combined_schema(self):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(30,931));y=np.array(['A','B']*15)
        X[0,930]=np.nan
        for name in ('svm','lightgbm'):
            model=model_for(name,True,len(y));model.fit(X,y)
            self.assertEqual(model.predict(X[:3]).shape,(3,))
