"""Headless checks for the ASL webcam feature app."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class ASLLiveFeatureAppTests(unittest.TestCase):
    def test_app_renders_typed_letter_and_camera_toggle(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertTrue(any("inspect features" in item.value for item in app.title))
        self.assertTrue(any(
            item.label == "Intended letter" for item in app.text_input
        ))
        self.assertTrue(any(item.label == "Camera on" for item in app.toggle))

    def test_current_models_and_archive_preserve_selection(self):
        app=AppTest.from_file(str(APP),default_timeout=30).run()
        choices=app.selectbox(key='asl_live_model_choice')
        self.assertEqual(len(choices.options),2)
        choices.select('reference').run()
        self.assertEqual(app.selectbox(key='asl_live_model_choice').value,'reference')
        app.toggle(key='asl_model_archive').set_value(True).run()
        self.assertEqual(len(app.selectbox(key='asl_live_model_choice').options),5)
        app.selectbox(key='asl_live_model_choice').select('tilt').run()
        app.toggle(key='asl_model_archive').set_value(False).run()
        self.assertEqual(app.selectbox(key='asl_live_model_choice').value,'angles')
        self.assertFalse(app.exception)

    def test_browser_capture_saves_exact_analyzed_failed_attempt_with_selected_model(self):
        from research.model_choices import MODEL_CHOICES
        buffer=BytesIO();Image.new('RGB',(32,24),(190,80,20)).save(buffer,format='JPEG')
        payload={'id':'known-frame','data':base64.b64encode(buffer.getvalue()).decode()}
        detector=type('Fixture',(),{'analyze':lambda self,image:{'status':'no_hand'}})()
        with patch('research.data.Detector',return_value=detector), \
             patch('research.live_model.load_bundle',return_value={}), \
             patch('research.capture.save_capture',return_value=Path('/tmp/test-capture')) as save:
            app=AppTest.from_file(str(APP),default_timeout=30).run()
            app.selectbox(key='asl_live_model_choice').select('reference').run()
            app.text_input(key='asl_live_target_letter').set_value('G').run()
            app.toggle(key='asl_live_camera_active').set_value(True).run()
            app.session_state['asl_live_feature_camera']={'frame':payload}
            app.run()
            self.assertFalse(app.exception,str(app.exception))
            button=next(b for b in app.button if b.label=='Save capture')
            self.assertFalse(button.disabled)
            button.click().run()
            self.assertFalse(app.exception,str(app.exception))
            packet,label,bundle,path,*_=save.call_args.args
            self.assertEqual(label,'G')
            self.assertEqual(path,next(p for k,_,p in MODEL_CHOICES if k=='reference'))
            self.assertEqual(packet['detection']['status'],'no_hand')
            self.assertIsNone(packet['prediction'])
            self.assertEqual(packet['browser_frame_id'],'known-frame')
            expected=np.asarray(Image.open(BytesIO(buffer.getvalue())).convert('RGB'))[:,:,::-1]
            np.testing.assert_array_equal(packet['model_frame'],expected)
            app.toggle(key='asl_live_camera_active').set_value(False).run()
            self.assertTrue(next(b for b in app.button if b.label=='Save capture').disabled)

    def test_webcam_component_uses_browser_media_and_component_v2_only(self):
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_v1 = any(
            isinstance(node, ast.Attribute)
            and node.attr == "v1"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "components"
            for node in ast.walk(tree)
        )
        self.assertFalse(imported_v1)
        self.assertIn("navigator.mediaDevices.getUserMedia", source)
        self.assertIn('runtime.sendState("frame"', source)


if __name__ == "__main__":
    unittest.main()
