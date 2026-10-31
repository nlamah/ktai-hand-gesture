"""Tests for development-only MediaPipe threshold comparisons."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from gesturelab.datasets.data import Detector, analyze_project, read_manifest
from scripts.tune_detector_thresholds import (
    parse_confidences, select_development_samples, threshold_pairs, tune_project,
)


class DetectorTuningTests(unittest.TestCase):
    def test_confidence_parser_validates_values_and_removes_duplicates(self):
        self.assertEqual(parse_confidences("0.1, 0.5,0.1"), (0.1, 0.5))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_confidences("1.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_confidences("not-a-number")

    def test_detector_rejects_confidences_outside_unit_interval(self):
        for detection, presence in ((-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)):
            with self.subTest(detection=detection, presence=presence):
                with self.assertRaises(ValueError):
                    Detector(detection, presence)

    def test_sample_selection_is_stable_and_excludes_test_or_excluded_images(self):
        samples = [
            {"id": "z", "label": "A", "role": "development", "included": True},
            {"id": "a", "label": "A", "role": "development", "included": True},
            {"id": "test", "label": "A", "role": "test", "included": True},
            {"id": "excluded", "label": "A", "role": "development", "included": False},
        ]
        selected = select_development_samples(samples, 1)
        self.assertEqual(selected, select_development_samples(samples, 1))
        self.assertEqual(len(selected), 1)
        self.assertNotIn(selected[0]["id"], {"test", "excluded"})

    def test_baseline_is_first_and_present_once(self):
        pairs = threshold_pairs((0.2, 0.5), (0.2, 0.5))
        self.assertEqual(pairs[0], (0.5, 0.5))
        self.assertEqual(pairs.count((0.5, 0.5)), 1)

    def test_sweep_writes_results_without_mutating_manifest_or_using_test_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            image_path = project / "images" / "development.png"
            image_path.parent.mkdir()
            Image.new("RGB", (4, 4)).save(image_path)
            manifest = {
                "schema": 2,
                "samples": [
                    {"id": "dev", "label": "A", "role": "development", "included": True,
                     "image_path": "images/development.png", "status": "no_hand"},
                    {"id": "test", "label": "A", "role": "test", "included": True,
                     "image_path": "images/test.png", "status": "no_hand"},
                ],
            }
            manifest_path = project / "manifest.json"
            original_manifest = json.dumps(manifest)
            manifest_path.write_text(original_manifest, encoding="utf-8")
            output = Path(temporary) / "tuning"

            class FakeDetector:
                def __init__(self, detection, presence):
                    self.thresholds = detection, presence

                def analyze(self, image):
                    image.load()
                    status = "ok" if min(self.thresholds) < 0.5 else "no_hand"
                    return {"status": status, "n_hands": 1 if status == "ok" else 0}

                def close(self):
                    pass

            with patch("scripts.tune_detector_thresholds.Detector", FakeDetector):
                summary = tune_project(project, output, (0.2,), (0.2,), per_label=0)

            self.assertEqual(len(summary), 2)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)
            with (output / "sample_results.csv").open(encoding="utf-8", newline="") as results_file:
                results = list(csv.DictReader(results_file))
            self.assertEqual(len(results), 2)
            self.assertTrue(all(result["sample_id"] == "dev" for result in results))
            self.assertEqual(results[1]["recovered_vs_baseline"], "True")
            with (output / "summary.csv").open(encoding="utf-8", newline="") as summary_file:
                summaries = list(csv.DictReader(summary_file))
            self.assertEqual(len(summaries), 2)
            self.assertEqual(summaries[0]["usable"], "0")
            self.assertEqual(summaries[1]["usable"], "1")
            with self.assertRaises(FileExistsError):
                tune_project(project, output, (0.2,), (0.2,), per_label=0)

    def test_reprocess_applies_and_records_selected_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            image_path = project / "images" / "already-analyzed.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(image_path)
            project.mkdir(exist_ok=True)
            manifest = {
                "schema": 2,
                "samples": [{
                    "id": "sample", "label": "A", "role": "development", "included": True,
                    "image_path": "images/already-analyzed.png", "status": "ok",
                }],
            }
            (project / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            class FakeDetector:
                def __init__(self, detection, presence):
                    self.thresholds = {
                        "min_hand_detection_confidence": detection,
                        "min_hand_presence_confidence": presence,
                    }

                def analyze(self, image):
                    image.load()
                    return {"status": "no_hand", "n_hands": 0, "detector_thresholds": self.thresholds}

                def close(self):
                    pass

            with patch("gesturelab.datasets.data.Detector", side_effect=FakeDetector) as detector:
                self.assertEqual(analyze_project(project), 0)
                self.assertEqual(analyze_project(
                    project, min_hand_detection_confidence=.2,
                    min_hand_presence_confidence=.3, reprocess=True), 1)

            detector.assert_called_once_with(.2, .3)
            updated = read_manifest(project)["samples"][0]
            self.assertEqual(updated["status"], "no_hand")
            self.assertEqual(updated["detector_thresholds"], {
                "min_hand_detection_confidence": .2,
                "min_hand_presence_confidence": .3,
            })


if __name__ == "__main__":
    unittest.main()
