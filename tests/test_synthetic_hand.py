"""Controls for the exploratory volume/landmark proxy; no webcam data needed."""
import unittest

import numpy as np

from research.gh_vertical import rotate_vertical
from research.live_model import detection_inputs
from research.studies.gh_occlusion import perturb, transformed
from research.synthetic_hand import render_hand
from tests.test_asl_categorical_features import synthetic_hand


class OcclusionProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points=synthetic_hand()
        # Overlapping index/middle fingers at different depths exchange occlusion on rotation.
        cls.points[9:13,0]=cls.points[5:9,0]+.02
        cls.points[9:13,2]=.2

    def test_noise_preserves_all_unselected_points_and_input(self):
        p=self.points.copy();before=p.copy();noise=np.ones((21,3));hidden=[7,8,11,12]
        turn=perturb(p,[],noise,0);noisy=perturb(p,hidden,noise,.05)
        visible=np.setdiff1d(np.arange(21),hidden)
        np.testing.assert_array_equal(noisy[visible],turn[visible])
        self.assertTrue(np.all(np.linalg.norm(noisy[hidden]-turn[hidden],axis=1)>0))
        np.testing.assert_array_equal(p,before)

    def test_proper_turn_and_intrinsic_measurements_preserved_without_noise(self):
        p=self.points
        source={'id':'unit-source','image_landmarks':p,'world_landmarks':p,'handedness':'Right','status':'ok'}
        expected=rotate_vertical(detection_inputs(source))
        actual=detection_inputs(transformed(source,[7,8],0))
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)

    def test_seed_is_repeatable_and_strengths_share_noise_direction(self):
        p=self.points;sample={'id':'test-source','handedness':'Right','image_landmarks':p,'world_landmarks':p}
        a=transformed(sample,[7,8],.02);b=transformed(sample,[7,8],.05);z=transformed(sample,[7,8],0)
        np.testing.assert_array_equal(a['image_landmarks'],transformed(sample,[7,8],.02)['image_landmarks'])
        np.testing.assert_allclose((a['image_landmarks']-z['image_landmarks'])*2.5,b['image_landmarks']-z['image_landmarks'],atol=1e-15)

    def test_depth_changes_visibility_without_removing_landmarks(self):
        p=self.points.copy();before=p.copy()
        front,a=render_hand(p,0);back,b=render_hand(p,180)
        self.assertEqual(front.size,(384,384));self.assertEqual(back.size,front.size)
        self.assertAlmostEqual(a['rotation_determinant'],1.)
        self.assertAlmostEqual(b['rotation_determinant'],1.)
        self.assertEqual(len(b['projected_landmarks']),21)
        self.assertNotEqual(a['component_occlusion'],b['component_occlusion'])
        self.assertTrue(set(b['hidden_finger_joints'])<=set((2,3,4,6,7,8,10,11,12,14,15,16,18,19,20)))
        np.testing.assert_array_equal(p,before)

    def test_reject_invalid_geometry(self):
        with self.assertRaises(ValueError):render_hand(np.full((21,3),np.nan))


if __name__=='__main__':unittest.main()
