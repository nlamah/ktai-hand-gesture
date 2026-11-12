"""Saved feature review must reuse capture geometry and never infer from the target."""
from pathlib import Path
from hashlib import sha256
import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import sklearn
from streamlit.testing.v1 import AppTest

from research.capture import save_capture
from research.feature_review import review_data
from research import live_model
from research.studies.compare_angles import classifier
from tests.test_live_model import detection

ROOT=Path(__file__).resolve().parents[1]
class FeatureReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=tempfile.TemporaryDirectory()
        cls.addClassCleanup(fixture.cleanup)
        directory=Path(fixture.name)
        rng=np.random.default_rng(42)
        estimator=classifier('svm','shin_angles',52).fit(
            rng.normal(size=(52,835)).astype(np.float32),live_model.LABELS*2)
        bundle={'version':live_model.BUNDLE_VERSION,'shin_version':live_model.SHIN_VERSION,
                'angle_columns':list(live_model.ANGLE_COLUMNS),
                'sklearn_version':sklearn.__version__,'estimator':estimator}
        cls.model=directory/'model.joblib'
        joblib.dump(bundle,cls.model)
        frame=np.arange(80*120*3,dtype=np.uint8).reshape(80,120,3)
        measured=detection()
        packet={'frame':frame,'model_frame':frame.copy(),'detection':measured,
                'prediction':live_model.predict_letter(bundle,measured),
                'captured_at_utc':'2026-11-12T00:00:00+00:00','camera':'test-fixture'}
        cls.capture=save_capture(packet,'H',bundle,cls.model,
                                sha256(cls.model.read_bytes()).hexdigest(),frame,
                                directory/'captures')

    def test_changing_reference_letter_changes_feature_comparison_but_not_prediction(self):
        args=(self.capture,ROOT/'assets/asl_feature_references.xlsx',ROOT/'config/thresholds.json',self.model)
        h=review_data(*args,label='H');g=review_data(*args,label='G')
        self.assertEqual(h['prediction'],g['prediction'])
        self.assertEqual(h['box'],g['box'])
        hh=next(r for r in h['rows'] if r['key']=='flexion_middle_pip')
        gg=next(r for r in g['rows'] if r['key']=='flexion_middle_pip')
        self.assertEqual(hh['actual'],gg['actual'])
        self.assertNotEqual(hh['expected'],gg['expected'])

    def test_app_review_branch_has_no_camera_toggle_and_uses_saved_measurements(self):
        args=['app.py','--capture',str(self.capture),'--model',str(self.model)]
        with patch('sys.argv',args):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertEqual(app.title[0].value,'Inspect finger features')
        self.assertFalse(any(t.label=='Camera on' for t in app.toggle))
        app.text_input[0].set_value('G')
        with patch('sys.argv',args):app.run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any('Compare against G' in s.value for s in app.subheader))


if __name__=='__main__':unittest.main()
