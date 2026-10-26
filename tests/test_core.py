import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gesturelab.features.inspection import extract_features,feature_columns
from gesturelab.training.rules import predict_rule,validate_rules,default_rules,UNKNOWN
from gesturelab.datasets.data import add_image,export_project,read_manifest,edit_metadata,project_lock
from gesturelab.training.experiments import grouped_folds,metrics
from scripts.demo import skeleton


class FeaturesTests(unittest.TestCase):
    def test_normalization_removes_translation_and_scale(self):
        points=skeleton("peace")
        raw=points*.2+.5
        first=extract_features(raw,points*.07)
        second=extract_features(raw*.7+.1,points*.18+[.2,.8,.4])
        names=feature_columns(first,"combined")
        np.testing.assert_allclose([first[n] for n in names],[second[n] for n in names],atol=1e-5)
        self.assertNotEqual(first["raw_0_x"],second["raw_0_x"])

    def test_geometry_rotation_invariant_but_normalized_coords_keep_orientation(self):
        points=skeleton("open_hand")
        t=.45
        rot=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        a=extract_features(points,points)
        b=extract_features(points@rot.T,points@rot.T)
        names=feature_columns(a,"geometry")
        np.testing.assert_allclose([a[n] for n in names],[b[n] for n in names],atol=1e-5)
        self.assertNotAlmostEqual(a["norm_8_x"],b["norm_8_x"])

    def test_bent_extended_and_thumb_features(self):
        fist=extract_features(skeleton("fist"),skeleton("fist"))
        palm=extract_features(skeleton("open_hand"),skeleton("open_hand"))
        thumb=extract_features(skeleton("thumbs_up"),skeleton("thumbs_up"))
        self.assertEqual(fist["geom_index_bent"],1)
        self.assertEqual(fist["geom_index_extended"],0)
        self.assertEqual(palm["geom_index_extended"],1)
        self.assertGreater(thumb["geom_thumb_index_mcp_distance"],fist["geom_thumb_index_mcp_distance"])
        self.assertEqual(len(feature_columns(fist,"raw")),63)
        self.assertEqual(len(feature_columns(fist,"normalized")),63)
        with self.assertRaises(ValueError):
            extract_features(np.zeros((21,3)),np.zeros((21,3)))

    def test_ambiguous_rules_abstain(self):
        config={"rules":[{"label":"a","conditions":{"x":{"min":0}}},
                         {"label":"b","conditions":{"x":{"max":1}}}]}
        validate_rules(config,["x"])
        self.assertEqual(predict_rule({"x":.5},config)[0],UNKNOWN)
        self.assertEqual(predict_rule({"x":2},config)[0],"a")
        with self.assertRaises(ValueError):
            validate_rules(config,["other"])
        score=metrics(["a","b"],[UNKNOWN,UNKNOWN],["a","b"])
        self.assertEqual(score["accuracy"],0)
        self.assertEqual(score["coverage"],0)

    def test_extended_thumb_is_not_a_fist_even_near_index_knuckle(self):
        points=skeleton("thumbs_up")
        features=extract_features(points,points)
        features["geom_thumb_index_mcp_distance"]=.65
        self.assertEqual(predict_rule(features,default_rules())[0],"thumbs_up")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project=Path(self.tmp.name)/"project"
        buffer=BytesIO()
        Image.new("RGB",(32,24),(60,100,120)).save(buffer,format="PNG")
        self.data=buffer.getvalue()

    def test_original_exact_input_and_deduplication(self):
        sample,created=add_image(self.project,self.data,"hand.png","fist","s1")
        self.assertTrue(created)
        self.assertEqual((self.project/sample["original_path"]).read_bytes(),self.data)
        im=Image.open(self.project/sample["image_path"])
        self.assertEqual(sample["pixel_sha256"],hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
        self.assertFalse(add_image(self.project,self.data,"copy.png","fist","s2")[1])
        with self.assertRaisesRegex(ValueError,"already exists"):
            add_image(self.project,self.data,"hand.png","open_hand","s2")

    def test_export_preserves_even_unprocessed_images(self):
        sample,_=add_image(self.project,self.data,"hand.png","fist","s1")
        path=export_project(self.project,Path(self.tmp.name)/"export.zip")
        with ZipFile(path) as archive:
            self.assertEqual(archive.read("dataset/"+sample["original_path"]),self.data)
            self.assertIn("dataset/"+sample["image_path"],archive.namelist())
            manifest=json.loads(archive.read("dataset/manifest.json"))
            self.assertEqual(manifest["samples"][0]["status"],"pending")

    def test_no_shared_session_between_development_and_test(self):
        add_image(self.project,self.data,"hand.png","fist","s1")
        buffer=BytesIO()
        Image.new("RGB",(32,24),(80,100,120)).save(buffer,format="PNG")
        with self.assertRaisesRegex(ValueError,"session"):
            add_image(self.project,buffer.getvalue(),"b.png","fist","s1",role="test")

    def test_project_lock_blocks_concurrent_write_and_releases(self):
        with project_lock(self.project):
            with self.assertRaises(ValueError):
                with project_lock(self.project):
                    pass
        with project_lock(self.project):
            pass


class SplitTests(unittest.TestCase):
    def test_each_validation_sample_once_and_groups_disjoint(self):
        rows=[{"label":label,"session":f"s{group}","participant":"p1"}
              for group in range(6) for label in ("a","b") for _ in range(3)]
        df=pd.DataFrame(rows)
        folds,groups=grouped_folds(df,3)
        validation=[]
        for train,test in folds:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            validation.extend(test.tolist())
        self.assertEqual(sorted(validation),list(range(len(df))))
        df["session"]="one_session"
        with self.assertRaisesRegex(ValueError,"different groups"):
            grouped_folds(df,3)


if __name__=="__main__":
    unittest.main()
