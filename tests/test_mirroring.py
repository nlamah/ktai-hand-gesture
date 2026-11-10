"""Mirror semantics, source weighting, held-out assignment and train-only preprocessing."""
import unittest

import numpy as np
from PIL import Image

from research.live_model import detection_inputs
from research.mirroring import mirror_image, reflected_detection, weighted_training_rows
from research.studies.mirror_augmentation import assignments_for, balanced_views, fit_mirrored
from research.studies.compare_angles import classifier
from tests.test_live_model import detection


class MirroringTests(unittest.TestCase):
    def test_rgb_and_geometry_reflect_horizontally_without_swapping_finger_indices(self):
        pixels = np.arange(2*4*3, dtype=np.uint8).reshape(2,4,3)
        np.testing.assert_array_equal(np.asarray(mirror_image(Image.fromarray(pixels))), pixels[:,::-1])
        d = detection(); mirrored = reflected_detection(d)
        self.assertEqual(mirrored['handedness'], 'Left')
        np.testing.assert_allclose(np.array(mirrored['image_landmarks'])[:,0], 1-np.array(d['image_landmarks'])[:,0])
        np.testing.assert_allclose(np.array(mirrored['image_landmarks'])[:,1:], np.array(d['image_landmarks'])[:,1:])
        a, b = detection_inputs(d)[0], detection_inputs(mirrored)[0]
        np.testing.assert_allclose(a[:190], b[:190], atol=2e-6)
        expected = a[190:820].reshape(210,3).copy(); expected[:,0] *= -1
        np.testing.assert_allclose(b[190:820].reshape(210,3), expected, atol=2e-6)
        np.testing.assert_allclose(a[820:], b[820:], atol=2e-5)
        again = reflected_detection(mirrored)
        np.testing.assert_allclose(detection_inputs(again), detection_inputs(d), atol=2e-6)
        self.assertEqual(d['handedness'], 'Right')

    def test_failed_mirrors_do_not_remove_originals_or_change_source_weight(self):
        x=np.ones((3,835),np.float32); mirrored=2*x; mirrored[1]=np.nan
        rows,labels,weights=weighted_training_rows(x,mirrored,np.array(['A','B','C']),[True,False,True])
        self.assertEqual(labels.tolist(),['A','B','C','A','C'])
        np.testing.assert_array_equal(weights,[.5,1,.5,.5,.5])
        for letter in 'ABC':self.assertEqual(weights[labels==letter].sum(),1)
        self.assertTrue(np.isfinite(rows).all())

    def test_balanced_mix_is_reproducible_and_never_changes_source_folds(self):
        y=np.array(list('AAAAABBBBB')*2);groups=np.array(['g1']*10+['g2']*10)
        folds=[dict(fold=1,train_groups=['g2'],test_groups=['g1']),dict(fold=2,train_groups=['g1'],test_groups=['g2'])]
        assignments=assignments_for(groups,folds); views=balanced_views(y,assignments)
        np.testing.assert_array_equal(views,balanced_views(y,assignments))
        for fold in [1,2]:
            for letter in 'AB':self.assertEqual(views[(assignments==fold)&(y==letter)].sum(),2)
        with self.assertRaises(ValueError):assignments_for(groups,[dict(fold=1,train_groups=['g1','g2'],test_groups=['g1'])])

    def test_augmented_fit_preserves_original_scaling_gamma_and_class_weights(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(12,835)).astype(np.float32)
        y=np.array(['A']*8+['B']*4);mirrored=x.copy();mirrored[:,820:]+=1000
        baseline=classifier('svm','shin_angles',len(y)).fit(x,y)
        candidate,rows=fit_mirrored(baseline,x,mirrored,y,np.ones(12,bool))
        self.assertEqual(rows,24)
        np.testing.assert_allclose(candidate[0].transform(x),baseline[0].transform(x))
        self.assertEqual(candidate[-1]._gamma,baseline[-1]._gamma)
        np.testing.assert_array_equal(candidate[-1].class_weight_,baseline[-1].class_weight_)

    def test_live_mirror_comparison_buttons_do_not_overlap_round_or_clip_panel(self):
        from research import live_camera
        panel=live_camera.camera_panel(np.zeros((480,640,3),np.uint8),model_key='mirror')
        for model_key in ('mirror_base','mirror'):
            key,_,_,rect=next(b for b in live_camera.MODEL_BUTTONS if b[0]==model_key)
            state={'shown':object()}
            live_camera.snapshot_mouse(live_camera.cv2.EVENT_LBUTTONDOWN,rect[0]+5,rect[1]+5,0,state)
            self.assertEqual(state['model_request'],key)
            self.assertLess(rect[3],live_camera.ROUND_BUTTON[1])
        self.assertGreaterEqual(panel.shape[0],live_camera.ROUND_BUTTON[3]+30)
        self.assertEqual(panel.shape[1],640)


if __name__ == '__main__':unittest.main()
