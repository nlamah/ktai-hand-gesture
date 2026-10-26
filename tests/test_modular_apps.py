"""User workflows in the separate apps, without access to physical hardware."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from streamlit.testing.v1 import AppTest
from scripts.demo import skeleton
from gesturelab.datasets.data import add_image,read_manifest,atomic_json
from gesturelab.features.audit import read_annotations,load_thresholds
from gesturelab.features.inspection import extract_features,FEATURE_VERSION

ROOT=Path(__file__).resolve().parents[1]

class ModularAppTests(unittest.TestCase):
    def assert_clean(self,app):
        self.assertEqual(len(app.exception),0,str(app.exception))

    def test_empty_apps_and_alphabet_capture_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=AppTest.from_file(str(ROOT/'apps/collect_app.py'),default_timeout=30)
            app.session_state['collect_project']=tmp
            app.run();self.assert_clean(app)
            for page in ['Guided capture','Upload photos','Check dataset','Export']:
                app.sidebar.radio[0].set_value(page).run();self.assert_clean(app)
            app.sidebar.radio[0].set_value('Guided capture').run()
            next(w for w in app.selectbox if w.label=='Capture plan').set_value('All 26 letters').run()
            self.assert_clean(app)
            start=next(w for w in app.button if w.label=='Open webcam and start new session')
            self.assertFalse(start.disabled)  # All references are ready; do not open real camera.
            audit=AppTest.from_file(str(ROOT/'apps/feature_check_app.py'),default_timeout=30)
            audit.session_state['audit_project']=tmp
            audit.run();self.assert_clean(audit)
            for page in ['Precision and recall','Code and definitions']:
                audit.radio[0].set_value(page).run();self.assert_clean(audit)

    def test_human_annotation_and_saved_threshold_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for i,role in enumerate(['development','test']):
                buffer=BytesIO();Image.new('RGB',(160,160),(40+i,70,90)).save(buffer,format='PNG')
                add_image(p,buffer.getvalue(),'sample.png','A',f'session{i}',role=role)
                m=read_manifest(p)
                world=skeleton('open_hand');raw=world*.2+[.5,.5,0]
                m['samples'][-1].update(status='ok',world_landmarks=world.tolist(),image_landmarks=raw.tolist(),
                    features=extract_features(raw,world,160,160),feature_version=FEATURE_VERSION)
                atomic_json(p/'manifest.json',m)
            app=AppTest.from_file(str(ROOT/'apps/feature_check_app.py'),default_timeout=30)
            app.session_state['audit_project']=str(p)
            app.run();self.assert_clean(app)
            photos=next(w for w in app.selectbox if w.label=='Photo')
            self.assertEqual(len(photos.options),1)  # Never tune against the held-out test image.
            for w in app.selectbox:
                if w.label in ['thumb','index finger','middle finger','ring finger','little finger']:
                    w.set_value('extended')
            app.run()
            next(w for w in app.button if w.label=='Save my assessment').click().run()
            self.assert_clean(app);self.assertEqual(len(read_annotations(p)),1)
            app.radio[0].set_value('Precision and recall').run();self.assert_clean(app)
            self.assertFalse(app.dataframe[0].value.precision.isna().all())
            next(w for w in app.number_input if w.label=='Extended: fingers (°)').set_value(160.0).run()
            next(w for w in app.button if w.label=='Save thresholds').click().run()
            self.assertEqual(load_thresholds(p)['extended_angle'],160.0)

if __name__=='__main__':unittest.main()
