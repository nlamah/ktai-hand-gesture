"""Aspect-correct 2D geometry, shared batch/live encoding and 849-input exports."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import sklearn

from research.image_angles import projected_angles,projected_angles_batch,PIXEL_ANGLE_COLUMNS,VERSION
from research.inputs import ANGLE_COLUMNS,LABELS
from research.live_model import IMAGE_ANGLE_BUNDLE_VERSION,SHIN_VERSION,load_bundle,model_inputs
from research.capture import save_capture
from research.studies.compare_image_angles import estimator
from tests.test_live_model import detection


class ImageAngleTests(unittest.TestCase):
    def test_pixel_aspect_and_projected_geometry_invariance(self):
        p=np.zeros((21,3));p[5,:2]=[.1,.2];p[6,:2]=[.2,.2];p[7,:2]=[.2,.4]
        self.assertAlmostEqual(float(projected_angles(p,[640,480])[3]),90,places=5)
        p[7,:2]=[.3,.3]
        # 64 horizontal pixels and (64,48) diagonal pixels: internal angle 143.13.
        value=projected_angles(p,[640,480])[3]
        self.assertAlmostEqual(float(value),143.1301,places=3)
        np.testing.assert_allclose(projected_angles(p,[1280,960]),projected_angles(p,[640,480]),equal_nan=True)
        reflected=p.copy();reflected[:,0]=1-p[:,0]
        np.testing.assert_allclose(projected_angles(reflected,[640,480]),projected_angles(p,[640,480]),atol=1e-5,equal_nan=True)
        p[:,2]=np.arange(21)*100
        self.assertAlmostEqual(float(projected_angles(p,[640,480])[3]),float(value),places=4)
        self.assertTrue(np.isnan(projected_angles(np.zeros((21,3)),[640,480])).all())
        with self.assertRaises(ValueError):projected_angles(p,[0,480])
        with self.assertRaises(ValueError):projected_angles(p,None)

    def test_batch_training_equals_live_inference_and_capture_schema(self):
        d=detection();d['image_size']=[640,480]
        pixels=projected_angles(d['image_landmarks'],d['image_size'])
        np.testing.assert_array_equal(projected_angles_batch([d['image_landmarks']],[d['image_size']])[0],pixels)
        rng=np.random.default_rng(64)
        model=estimator(52).fit(rng.normal(size=(52,849)).astype(np.float32),LABELS*2)
        bundle={'version':IMAGE_ANGLE_BUNDLE_VERSION,'shin_version':SHIN_VERSION,'angle_columns':list(ANGLE_COLUMNS),
                'pixel_angle_columns':list(PIXEL_ANGLE_COLUMNS),'pixel_angle_version':VERSION,
                'sklearn_version':sklearn.__version__,'estimator':model}
        row=model_inputs(bundle,d)
        self.assertEqual(row.shape,(1,849));np.testing.assert_array_equal(row[0,835:],pixels)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(bundle,path);load_bundle(path)
            frame=np.zeros((480,640,3),np.uint8)
            packet={'frame':frame,'model_frame':frame,'detection':d,'prediction':{'letter':'G'},'captured_at_utc':'test'}
            result=save_capture(packet,'G',bundle,path,'test',frame,Path(directory)/'captures')
            record=json.loads((result/'capture.json').read_text())
            self.assertEqual(len(record['raw_model_features']),849)
            self.assertEqual(len(record['feature_names']),849)
            self.assertEqual(len(record['joint_angles_degrees']),15)
            self.assertEqual(len(record['projected_angles_degrees']),14)
            self.assertEqual(record['input_variant'],'shin_angles_2d')
            bundle['pixel_angle_columns'].reverse();joblib.dump(bundle,path)
            with self.assertRaises(ValueError):load_bundle(path)
