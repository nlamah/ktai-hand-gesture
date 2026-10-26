"""Regression checks for data isolation, reference migration and geometry."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo import skeleton
from gesturelab.datasets.alphabet import initialize_alphabet
from gesturelab.datasets.data import (
    add_image, analyze_project, atomic_json, edit_metadata, metadata_table,
    read_manifest,
)
from gesturelab.features.audit import evaluate_annotations, read_annotations, save_annotation
from gesturelab.features.inspection import extract_features, feature_columns
from gesturelab.features.finger_features import FINGERS, validate_thresholds


class TemporaryDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name)

    def add_photo(self, value=40, role="development", session="round1"):
        buffer = BytesIO()
        Image.new("RGB", (32, 24), (value, 60, 80)).save(buffer, format="PNG")
        return add_image(self.project, buffer.getvalue(), "photo.png", "A", session, role=role)[0]


class AnalysisRegressionTests(TemporaryDatasetTests):
    def test_already_analyzed_photos_do_not_initialize_detector(self):
        self.add_photo()
        manifest = read_manifest(self.project)
        manifest["samples"][0]["status"] = "ok"
        atomic_json(self.project / "manifest.json", manifest)
        with patch("gesturelab.datasets.data.Detector") as detector:
            self.assertEqual(analyze_project(self.project), 0)
            detector.assert_not_called()
        self.assertEqual(read_manifest(self.project), manifest)

    def test_only_pending_photos_are_analyzed_and_detector_is_closed(self):
        for value in (40, 80, 120):
            self.add_photo(value)
        manifest = read_manifest(self.project)
        manifest["samples"][0]["status"] = "ok"
        manifest["samples"][1]["status"] = "no_hand"
        atomic_json(self.project / "manifest.json", manifest)
        with patch("gesturelab.datasets.data.Detector") as detector:
            detector.return_value.analyze.return_value = {"status": "no_hand"}
            self.assertEqual(analyze_project(self.project), 1)
            detector.return_value.analyze.assert_called_once()
            detector.return_value.close.assert_called_once()
        self.assertEqual([s["status"] for s in read_manifest(self.project)["samples"]],
                         ["ok", "no_hand", "no_hand"])

    def test_detector_failure_closes_resources_without_losing_pending_photo(self):
        self.add_photo()
        before = read_manifest(self.project)
        with patch("gesturelab.datasets.data.Detector") as detector:
            detector.return_value.analyze.side_effect = RuntimeError("analysis failed")
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                analyze_project(self.project)
            detector.return_value.close.assert_called_once()
        self.assertEqual(read_manifest(self.project), before)

    def test_invalid_metadata_edit_is_not_partially_saved(self):
        self.add_photo()
        self.add_photo(80, session="round2")
        before = read_manifest(self.project)
        edited = metadata_table(self.project)
        edited.loc[0, "label"] = "B"
        edited.loc[1, "role"] = "invalid"
        with self.assertRaises(ValueError):
            edit_metadata(self.project, edited)
        self.assertEqual(read_manifest(self.project), before)


class AnnotationIsolationTests(TemporaryDatasetTests):
    def test_reserved_test_photos_cannot_receive_tuning_annotations(self):
        sample = self.add_photo(role="test")
        with self.assertRaisesRegex(ValueError, "development"):
            save_annotation(self.project, sample["id"], dict.fromkeys(FINGERS, "extended"))
        self.assertEqual(read_annotations(self.project), {})

    def test_incomplete_and_invalid_assessments_are_rejected(self):
        sample = self.add_photo()
        for states in ({"index": "extended"}, dict.fromkeys(FINGERS, "invalid")):
            with self.subTest(states=states), self.assertRaises(ValueError):
                save_annotation(self.project, sample["id"], states)
        self.assertEqual(read_annotations(self.project), {})

    def test_excluded_and_reassigned_test_photos_do_not_enter_audit(self):
        for value in (40, 80):
            sample = self.add_photo(value, session=f"round{value}")
            save_annotation(self.project, sample["id"], dict.fromkeys(FINGERS, "extended"))
        manifest = read_manifest(self.project)
        manifest["samples"][0]["included"] = False
        manifest["samples"][1]["role"] = "test"
        atomic_json(self.project / "manifest.json", manifest)
        with patch("gesturelab.features.audit.features_for_sample") as features:
            table, records, meta = evaluate_annotations(self.project)
            features.assert_not_called()
        self.assertTrue(records.empty)
        self.assertTrue(table.precision.isna().all())
        self.assertTrue(table.recall.isna().all())
        self.assertEqual(meta["evaluated_fingers"], 0)
        self.assertIsNone(meta["coverage"])


class ReferenceMigrationTests(TemporaryDatasetTests):
    def test_old_supplied_references_update_while_custom_references_survive(self):
        initialize_alphabet(self.project)
        custom = self.project / "references/custom.png"
        Image.new("RGB", (12, 12), "red").save(custom)
        original = custom.read_bytes()
        mapping = {"A": "references/custom.png", "B": "references/alphabet/B.png"}
        atomic_json(self.project / "references/index.json", mapping)
        initialize_alphabet(self.project)
        updated = json.loads((self.project / "references/index.json").read_text())
        self.assertEqual(updated["A"], "references/custom.png")
        self.assertEqual(custom.read_bytes(), original)
        self.assertEqual(updated["B"], "references/ngt_alphabet/B.png")
        self.assertEqual(len(updated), 26)
        self.assertEqual(read_manifest(self.project)["samples"], [])


class GeometryValidationTests(unittest.TestCase):
    def test_aspect_ratio_correction_preserves_same_pixel_geometry(self):
        world = skeleton("peace")
        pixels = world * 30 + [100, 100, 0]
        square = pixels / [320, 320, 320]
        wide = pixels / [640, 320, 640]
        first = extract_features(square, world, 320, 320)
        second = extract_features(wide, world, 640, 320)
        names = feature_columns(first, "normalized")
        np.testing.assert_allclose([first[n] for n in names], [second[n] for n in names], atol=1e-12)
        self.assertNotEqual(first["raw_0_x"], second["raw_0_x"])

    def test_nonfinite_or_malformed_landmarks_are_rejected(self):
        valid = skeleton("open_hand")
        invalid = [np.zeros((20, 3)), np.full((21, 3), np.nan), np.full((21, 3), np.inf)]
        for points in invalid:
            for raw, world in ((points, valid), (valid, points)):
                with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                    extract_features(raw, world)

    def test_invalid_image_dimensions_are_rejected(self):
        points = skeleton("open_hand")
        for width, height in ((0, 100), (100, -1), (float("inf"), 100), (100, float("nan"))):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                extract_features(points, points, width, height)

    def test_invalid_thresholds_cannot_silently_change_classification(self):
        for config in ({"unknown": 1}, {"extended_angle": True}, {"bent_angle": float("nan")},
                       {"extended_straightness": 1.1}, {"bent_straightness": .9}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_thresholds(config)


if __name__ == "__main__":
    unittest.main()
