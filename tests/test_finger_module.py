"""Specification and arithmetic tests for the independently inspectable module."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gesturelab.features.finger_features import angle,classify_finger,compute_finger_features,validate_thresholds
from gesturelab.features.inspection import features_for_sample
from scripts.demo import skeleton

class FingerSpecificationTests(unittest.TestCase):
    def test_known_angles_and_degenerate_geometry(self):
        self.assertAlmostEqual(angle([-1,0,0],[0,0,0],[1,0,0]),180)
        self.assertAlmostEqual(angle([0,1,0],[0,0,0],[1,0,0]),90)
        with self.assertRaises(ValueError):angle([0,0,0],[0,0,0],[1,0,0])

    def test_exact_state_boundaries_and_ambiguous_middle(self):
        self.assertEqual(classify_finger(155,155,.8),(True,False))
        self.assertEqual(classify_finger(154.9,180,.95),(False,False))
        self.assertEqual(classify_finger(124.9,180,.95),(False,True))
        self.assertEqual(classify_finger(125,180,.95),(False,False))
        self.assertEqual(classify_finger(180,180,.64),(False,True))
        self.assertEqual(classify_finger(145,145,.8,'thumb'),(True,False))
        self.assertEqual(classify_finger(145,145,.8,'index'),(False,False))
        with self.assertRaises(ValueError):validate_thresholds({'bent_angle':160})
        with self.assertRaises(ValueError):classify_finger(float('nan'),180,1)

    def test_geometry_count_and_same_base_information(self):
        p=skeleton('open_hand')
        f=compute_finger_features(p)
        self.assertEqual(len(f),81)
        self.assertEqual(f['geom_index_extended'],1)
        sample={'image_landmarks':p*.2+.5,'world_landmarks':p,'width':320,'height':320}
        all_features=features_for_sample(sample)
        baseline={k:v for k,v in all_features.items() if k.startswith(('norm_','world_'))}
        augmented={k:v for k,v in all_features.items() if k.startswith(('norm_','world_','geom_'))}
        self.assertEqual(len(baseline),126);self.assertEqual(len(augmented),207)
        self.assertTrue(all(augmented[k]==v for k,v in baseline.items()))
        self.assertFalse(any(k.startswith('geom_') for k in baseline))

if __name__=='__main__':unittest.main()
