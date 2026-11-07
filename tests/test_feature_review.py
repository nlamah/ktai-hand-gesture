"""Saved feature review must reuse capture geometry and never infer from the target."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from research.feature_review import review_data
from research.live_model import DEFAULT_MODEL

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'captures/confusions/20261031T152116143793Z_11756e66'


@unittest.skipUnless((CAPTURE/'capture.json').exists() and DEFAULT_MODEL.exists(),'Local capture/model required')
class FeatureReviewTests(unittest.TestCase):
    def test_changing_reference_letter_changes_feature_comparison_but_not_prediction(self):
        args=(CAPTURE,ROOT/'assets/asl_feature_references.xlsx',ROOT/'config/thresholds.json',DEFAULT_MODEL)
        h=review_data(*args,label='H');g=review_data(*args,label='G')
        self.assertEqual(h['prediction'],g['prediction'])
        self.assertEqual(h['box'],g['box'])
        hh=next(r for r in h['rows'] if r['key']=='flexion_middle_pip')
        gg=next(r for r in g['rows'] if r['key']=='flexion_middle_pip')
        self.assertEqual(hh['actual'],gg['actual'])
        self.assertNotEqual(hh['expected'],gg['expected'])

    def test_app_review_branch_has_no_camera_toggle_and_uses_saved_measurements(self):
        with patch('sys.argv',['app.py','--capture',str(CAPTURE)]):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertEqual(app.title[0].value,'Vingerkenmerken controleren')
        self.assertFalse(any(t.label=='Webcam on' for t in app.toggle))
        app.text_input[0].set_value('G')
        with patch('sys.argv',['app.py','--capture',str(CAPTURE)]):app.run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any('Controle tegen G' in s.value for s in app.subheader))


if __name__=='__main__':unittest.main()
