"""Binary diagnostic live inputs must match training and clear failed detections."""
import unittest
from unittest.mock import Mock
import numpy as np
from research import live_gh
from research.geometry import measure_hand_features
from research.image_angles import projected_angles
from research.inputs import encode_angles
from tests.test_live_model import detection


class LiveGHTests(unittest.TestCase):
    def test_live_inputs_match_training_representations_and_middle_ablation(self):
        d=detection();d['image_size']=[640,480]
        w=encode_angles(measure_hand_features(d['world_landmarks'],d['handedness'])).astype(np.float32)
        p=projected_angles(d['image_landmarks'],d['image_size'])
        expected={'3d':w,'2d':p,'both':np.r_[w,p],'middle':np.array([w[6],p[6]],np.float32)}
        for variant,row in expected.items():np.testing.assert_array_equal(live_gh.inputs(d,variant)[0],row)
        model=Mock()
        self.assertIsNone(live_gh.predict({'estimator':model},{'status':'no_hand'},'middle'))
        model.predict.assert_not_called()
