"""Synthetic tests for human ASL feature annotation and agreement."""
import tempfile
import unittest
from pathlib import Path

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_annotations import (
    NOT_ASSESSABLE, agreement_report, empty_annotation_file,
    load_annotations, save_annotation, validate_annotation_file,
)


class ASLAnnotationTests(unittest.TestCase):
    def _project(self, directory, role="development"):
        sample = {
            "id": "sample-001", "role": role, "status": "ok", "included": True,
            "original_sha256": "abc123", "image_path": "images/sample.png",
        }
        atomic_json(Path(directory) / "manifest.json", {"schema": 2, "samples": [sample]})
        return sample

    def test_saves_and_reloads_one_independent_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = self._project(temporary)
            saved = save_annotation(
                temporary, sample, "reviewer-a",
                {"flexion_index_pip": "extended", "spread_index_middle": NOT_ASSESSABLE},
            )
            data = load_annotations(temporary)
            self.assertEqual(len(data["annotations"]), 1)
            self.assertEqual(data["annotations"][0]["features"], saved["features"])
            self.assertNotIn("label", saved)

    def test_test_samples_and_automatic_unknown_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = self._project(temporary, role="test")
            with self.assertRaisesRegex(ValueError, "Only included development"):
                save_annotation(temporary, sample, "reviewer-a", {})
        data = empty_annotation_file()
        data["annotations"] = [{
            "sample_id": "sample-001", "annotator": "reviewer-a",
            "image_sha256": "abc123", "annotated_at": "now",
            "features": {"flexion_index_pip": "unknown"},
        }]
        with self.assertRaisesRegex(ValueError, "Invalid human category"):
            validate_annotation_file(data)

    def test_agreement_excludes_not_assessable_and_unpaired_values(self):
        data = empty_annotation_file()
        data["annotations"] = [
            {"sample_id": "one", "annotator": "a", "features": {
                "flexion_index_pip": "extended",
                "spread_index_middle": NOT_ASSESSABLE,
            }},
            {"sample_id": "one", "annotator": "b", "features": {
                "flexion_index_pip": "extended",
                "spread_index_middle": "spread",
            }},
            {"sample_id": "two", "annotator": "a", "features": {
                "flexion_index_pip": "bent",
            }},
        ]
        result = {row["feature"]: row for row in agreement_report(data)}
        self.assertEqual(result["flexion_index_pip"]["exact_agreements"], 1)
        self.assertEqual(result["flexion_index_pip"]["exact_agreement"], 1.0)
        self.assertEqual(result["spread_index_middle"]["double_annotated_samples"], 0)
        self.assertIsNone(result["spread_index_middle"]["exact_agreement"])


if __name__ == "__main__":
    unittest.main()
