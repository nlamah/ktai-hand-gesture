"""Camera workflow regressions without requiring webcam hardware."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.camera.runtime import CaptureSchedule,StableLabel,save_camera_photo,reference_for,save_reference,run_camera,new_session_name
from gesturelab.datasets.data import read_manifest,edit_metadata,export_project
from gesturelab.training.experiments import comparison_readiness,grouped_folds,train_preview,predict_bundle
from gesturelab.features.inspection import extract_features,FEATURE_VERSION
import joblib


def analysis(label="fist"):
    points=skeleton(label)
    raw=points*.15+.5
    return {"status":"ok","n_hands":1,"image_landmarks":raw.tolist(),"world_landmarks":points.tolist(),
            "features":extract_features(raw,points,100,80),"feature_version":FEATURE_VERSION}


def _capture_simulation():
    frame=np.zeros((80,100,3),dtype=np.uint8)
    camera=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None)
    state={"frame":0,"released":False,"closed":False,"clock":0}
    def read():
        state["frame"]+=1
        return True,frame+state["frame"]
    camera.read=read;camera.release=lambda:state.update(released=True)
    def key(_):
        n=state["frame"]
        return ord(' ') if n==1 else (ord('q') if n>=14 else -1)
    fake=SimpleNamespace(VideoCapture=lambda _:camera,CAP_PROP_FRAME_WIDTH=3,CAP_PROP_FRAME_HEIGHT=4,
         WINDOW_NORMAL=0,WND_PROP_VISIBLE=4,COLOR_BGR2RGB=1,COLOR_RGB2BGR=1,
         namedWindow=lambda *a:None,resizeWindow=lambda *a:None,imshow=lambda *a:None,
         cvtColor=lambda data,*a:data[:,:,::-1],waitKey=key,getWindowProperty=lambda *a:1,
         destroyAllWindows=lambda:None)
    class FakeDetector:
        def analyze(self,image):
            return {"status":"no_hand"} if state["frame"]<=3 else analysis()
        def close(self):
            state["closed"]=True
    def clock():
        state["clock"]+=.6
        return state["clock"]
    return fake, FakeDetector, clock, state


class CaptureTests(unittest.TestCase):
    def test_countdown_waits_for_hand_and_resets_after_pause(self):
        s=CaptureSchedule(["A","B"],2,interval=2,change_delay=4)
        self.assertFalse(s.tick(0,True)[0])
        s.toggle_pause()
        self.assertFalse(s.tick(0,True)[0])
        self.assertFalse(s.tick(3,False)[0])
        self.assertFalse(s.tick(4,True)[0])
        self.assertFalse(s.tick(7,True)[0])
        self.assertTrue(s.tick(8,True)[0]);s.commit()
        self.assertEqual((s.label,s.shot),("A",2))
        self.assertFalse(s.tick(9,True)[0]);self.assertTrue(s.tick(11,True)[0]);s.commit()
        self.assertEqual((s.label,s.shot),("B",1))
        s.toggle_pause();s.toggle_pause()
        self.assertFalse(s.tick(30,True)[0]);self.assertTrue(s.tick(34,True)[0])
        s.commit();s.commit()
        self.assertTrue(s.done);self.assertFalse(s.tick(100,True)[0])

    def test_missing_hand_clears_temporal_labels(self):
        vote=StableLabel(3)
        vote.update("A");vote.update("A")
        self.assertEqual(vote.update("B"),"A")
        self.assertIsNone(vote.update(None));self.assertEqual(vote.update("B"),"B")
        self.assertNotEqual(new_session_name(),new_session_name())

    def test_raw_pixels_and_landmarks_are_saved_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Image.new("RGB",(100,80),(20,30,50));image.putpixel((1,2),(255,0,0))
            s,created=save_camera_photo(tmp,image,analysis(),"fist","round1")
            self.assertTrue(created)
            saved=read_manifest(tmp)["samples"][0]
            self.assertEqual(saved["status"],"ok");self.assertEqual(saved["features"],analysis()["features"])
            np.testing.assert_array_equal(np.asarray(Image.open(Path(tmp)/s["image_path"])),np.asarray(image))
            self.assertFalse(saved["provenance"]["mirrored_input"])
            with self.assertRaises(ValueError):
                save_camera_photo(tmp,image,{"status":"no_hand"},"fist","round1")
            self.assertEqual(len(read_manifest(tmp)["samples"]),1)

    def test_reference_never_automatically_uses_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            im=Image.new("RGB",(100,80),(30,50,100))
            save_camera_photo(tmp,im,analysis(),"fist","test1",role="test")
            self.assertIsNone(reference_for(tmp,"fist")[0])
            buffer=BytesIO();im.save(buffer,format="PNG")
            selected=save_reference(tmp,"fist",buffer.getvalue())
            # macOS may expose the same temporary file via /var and /private/var.
            self.assertTrue(reference_for(tmp,"fist")[0].samefile(selected))
            self.assertEqual(len(read_manifest(tmp)["samples"]),1)
            from zipfile import ZipFile
            with ZipFile(export_project(tmp,Path(tmp)/"export.zip")) as z:
                self.assertIn("dataset/references/index.json",z.namelist())

    def test_live_prediction_uses_model_without_reading_fixed_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            read_manifest(tmp)
            (Path(tmp)/"rules.json").write_text("invalid old rules")
            camera=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,
                read=lambda:(True,np.zeros((80,100,3),dtype=np.uint8)),release=lambda:None)
            fake=SimpleNamespace(VideoCapture=lambda _:camera,CAP_PROP_FRAME_WIDTH=3,CAP_PROP_FRAME_HEIGHT=4,
                WINDOW_NORMAL=0,WND_PROP_VISIBLE=4,COLOR_BGR2RGB=1,COLOR_RGB2BGR=1,
                namedWindow=lambda *a:None,resizeWindow=lambda *a:None,imshow=lambda *a:None,
                cvtColor=lambda data,*a:data,waitKey=lambda _:ord('q'),getWindowProperty=lambda *a:1,
                destroyAllWindows=lambda:None)
            detector=SimpleNamespace(analyze=lambda image:analysis(),close=lambda:None)
            with patch.dict(sys.modules,{"cv2":fake}), \
                 patch("gesturelab.camera.runtime.Detector",return_value=detector), \
                 patch("gesturelab.camera.runtime.joblib.load",return_value={"feature_version":FEATURE_VERSION}), \
                 patch("gesturelab.camera.runtime.predict_bundle",return_value=["A"]), \
                 patch("gesturelab.camera.runtime.camera_canvas",return_value=Image.new("RGB",(100,80))) as canvas:
                run_camera({"mode":"predict","project":tmp,"model":"model.joblib"})
            self.assertEqual(canvas.call_args.args[5],"Prediction: A")
            self.assertEqual(canvas.call_args.args[6],["Prediction: A","Raw prediction: A"])

    def test_live_prediction_requires_model_before_opening_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("gesturelab.camera.runtime.Detector") as detector:
                with self.assertRaisesRegex(ValueError,"Select a trained model"):
                    run_camera({"mode":"predict","project":tmp})
                detector.assert_not_called()

    def test_camera_failures_publish_error_and_release_all_resources(self):
        for failure in ('read', 'analysis'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                fake, detector, clock, state = _capture_simulation()
                if failure == 'read':
                    fake.VideoCapture(0).read = lambda: (False, None)
                status = Path(tmp) / 'status.json'
                bundle = {'feature_version': FEATURE_VERSION}
                with patch.dict(sys.modules, {'cv2': fake}), \
                     patch('gesturelab.camera.runtime.Detector', detector), \
                     patch('gesturelab.camera.runtime.joblib.load', return_value=bundle), \
                     patch.object(detector, 'analyze', side_effect=RuntimeError('Simulated detector failure')):
                    with self.assertRaises(RuntimeError):
                        run_camera({'mode': 'predict', 'project': tmp, 'model': 'model.joblib'}, status)
                self.assertTrue(state['released'])
                self.assertTrue(state['closed'])
                self.assertEqual(json.loads(status.read_text())['state'], 'error')

    def test_record_loop_saves_sequence_and_releases_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake, FakeDetector, clock, state = _capture_simulation()
            buffer=BytesIO();Image.new("RGB",(100,80)).save(buffer,format="PNG")
            for label in ("A","B"):
                save_reference(tmp,label,buffer.getvalue())
            with patch.dict(sys.modules,{"cv2":fake}),patch("gesturelab.camera.runtime.Detector",FakeDetector),patch("gesturelab.camera.runtime.time.monotonic",clock):
                run_camera({"mode":"capture","project":tmp,"labels":["A","B"],"photos_per_label":1,
                            "interval":.5,"change_delay":.5,"session":"new_round"})
            samples=read_manifest(tmp)["samples"]
            self.assertEqual([s["label"] for s in samples],["A","B"])
            self.assertEqual({s["session"] for s in samples},{"new_round"})
            self.assertTrue(state["released"] and state["closed"])
            self.assertTrue(all(s["status"]=="ok" for s in samples))


class ReadinessTests(unittest.TestCase):
    def test_two_sessions_allow_two_folds_many_photos_alone_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            for session in ("round1","round2"):
                for label in ("fist","open_hand"):
                    for j in range(2):
                        value=20+len(read_manifest(tmp)["samples"])
                        save_camera_photo(tmp,Image.new("RGB",(100,80),(value,30,40)),analysis(label),label,session)
            ready=comparison_readiness(tmp)
            self.assertEqual(ready.groups.tolist(),[2,2]);self.assertEqual(ready.development_usable.tolist(),[4,4])
            df=pd.DataFrame(read_manifest(tmp)["samples"])
            folds,groups=grouped_folds(df,2)
            self.assertEqual(len(folds),2)
            for train,valid in folds:
                self.assertFalse(set(groups[train])&set(groups[valid]))
            df["session"]="same_name";edit_metadata(tmp,df)
            self.assertEqual(comparison_readiness(tmp).groups.tolist(),[1,1])
            with self.assertRaisesRegex(ValueError,"4 photos, 1 sessions"):
                grouped_folds(df,2)

    def test_preview_training_excludes_reserved_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label in ("fist","open_hand"):
                for i,role in enumerate(("development","test")):
                    value=40+len(read_manifest(tmp)["samples"])
                    save_camera_photo(tmp,Image.new("RGB",(100,80),(value,50,70)),analysis(label),label,"round"+str(i),role=role)
            path=train_preview(tmp)
            report=json.loads(path.with_name("preview.json").read_text())
            expected={s["id"] for s in read_manifest(tmp)["samples"] if s["role"]=="development"}
            self.assertEqual(set(report["train_ids"]),expected);self.assertNotIn("accuracy",report)
            result=predict_bundle(joblib.load(path),[analysis("fist")["features"],analysis("open_hand")["features"]])
            self.assertEqual(set(result),{"fist","open_hand"})


if __name__=="__main__":
    unittest.main()
