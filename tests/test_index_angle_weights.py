"""Targeted weights must affect only the intended standardized index angles."""
import unittest
import numpy as np
from scripts.tune_index_angles import weighted_input,JOINTS,WEIGHTS
from scripts.cross_validate_shin_workbook import ANGLE_COLUMNS


class IndexAngleWeightsTests(unittest.TestCase):
    def test_only_mcp_pip_dip_columns_receive_their_selected_weight(self):
        shin=np.arange(1640,dtype=float).reshape(2,820)
        angles=np.tile(np.arange(1,16,dtype=float),(2,1))
        original=angles.copy()
        actual=weighted_input(shin,angles,(.5,2.,1.))
        np.testing.assert_array_equal(actual[:,:820],shin)
        expected=angles.copy()
        expected[:,ANGLE_COLUMNS.index('index_mcp_angle_deg')]*=.5
        expected[:,ANGLE_COLUMNS.index('index_pip_angle_deg')]*=2
        np.testing.assert_array_equal(actual[:,820:],expected)
        np.testing.assert_array_equal(angles,original)
        self.assertEqual(actual.shape,(2,835))

    def test_identity_weights_exactly_preserve_inputs(self):
        shin=np.zeros((3,820));angles=np.ones((3,15))
        np.testing.assert_array_equal(weighted_input(shin,angles,(1.,1.,1.)),np.c_[shin,angles])
        self.assertEqual(WEIGHTS,(.5,1.,2.))
        self.assertEqual(len(JOINTS),3)
