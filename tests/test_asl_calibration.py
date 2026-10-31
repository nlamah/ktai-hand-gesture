"""Synthetic-only tests for calibration role and threshold-selection safeguards."""
import json
from pathlib import Path
import tempfile
import unittest

from gesturelab.datasets.data import atomic_json, fingerprint
from gesturelab.features.asl_calibration import calibrate_asl_thresholds
from gesturelab.features.asl_annotations import ANNOTATION_FILENAME, empty_annotation_file
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS


def _synthetic_landmarks():
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [[-.25, .1, 0], [-.4, .25, 0], [-.55, .4, 0], [-.7, .55, 0]]
    for base, x in ((5, -.45), (9, -.15), (13, .15), (17, .45)):
        points[base] = [x, .5, 0]
        points[base + 1] = [x, .8, 0]
        points[base + 2] = [x, 1.1, 0]
        points[base + 3] = [x, 1.4, 0]
    image = [[point[0] * .2 + .5, point[1] * .2 + .5, 0] for point in points]
    return points, image


class ASLCalibrationTests(unittest.TestCase):
    def test_selects_on_calibration_only_and_reports_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            world, image = _synthetic_landmarks()
            samples = []
            roles = {}
            groups = {}
            annotations = empty_annotation_file()
            for index in range(5):
                sample_id = f"synthetic-{index}"
                partition = "calibration_validation" if index >= 3 else "calibration_train"
                group = f"group-{index}"
                samples.append({
                    "id": sample_id, "label": "A", "participant": "p",
                    "session": group, "role": "development", "included": True,
                    "status": "ok", "width": 320, "height": 320,
                    "image_landmarks": image, "world_landmarks": world,
                })
                roles[sample_id] = partition
                groups[sample_id] = json.dumps(["p", group])
                annotations["annotations"].append({
                    "sample_id": sample_id, "annotator": "reviewer-a",
                    "annotated_at": "synthetic", "image_sha256": "",
                    "features": {"flexion_index_pip": "extended"},
                })
            samples.append({
                "id": "test-only", "label": "A", "role": "test",
                "included": True, "status": "ok",
            })
            atomic_json(project / "manifest.json", {"schema": 2, "samples": samples})
            role_map_path = root / "roles.json"
            atomic_json(role_map_path, {
                "schema_version": 1, "dataset_fingerprint": fingerprint(project),
                "grouping": "session", "roles": roles, "groups": groups,
            })
            annotation_path = project / "annotations" / ANNOTATION_FILENAME
            atomic_json(annotation_path, annotations)
            candidate_path = root / "candidates.json"
            atomic_json(candidate_path, {
                "candidates": [
                    PARAMETER_DEFAULTS,
                    {"flexion": {"extended_min_deg": 179, "bent_max_deg": 115}},
                ],
            })

            output = calibrate_asl_thresholds(
                project, role_map_path, candidate_path, root / "calibration")
            report = json.loads((output / "calibration.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["baseline_validation"]["n_samples"], 2)
            self.assertEqual(report["selected_validation"]["n_samples"], 2)
            self.assertIn("final_test samples were not read", report["limitations"][-1])
            self.assertFalse((output / "model.joblib").exists())

    def test_rejects_group_leakage_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            world, image = _synthetic_landmarks()
            samples = []
            roles = {}
            for index in range(2):
                sample_id = f"sample-{index}"
                samples.append({
                    "id": sample_id, "label": "A", "participant": "p",
                    "session": "same", "role": "development", "included": True,
                    "status": "ok", "width": 320, "height": 320,
                    "image_landmarks": image, "world_landmarks": world,
                })
                roles[sample_id] = "calibration_train" if index == 0 else "calibration_validation"
            atomic_json(project / "manifest.json", {"schema": 2, "samples": samples})
            role_map = root / "roles.json"
            atomic_json(role_map, {
                "schema_version": 1, "dataset_fingerprint": fingerprint(project),
                "grouping": "session",
                "roles": roles, "groups": {"sample-0": "leak", "sample-1": "leak"},
            })
            annotations = empty_annotation_file()
            annotations["annotations"] = [{
                "sample_id": sample_id, "annotator": "reviewer",
                "annotated_at": "synthetic", "image_sha256": "",
                "features": {"flexion_index_pip": "extended"},
            } for sample_id in roles]
            atomic_json(project / "annotations" / ANNOTATION_FILENAME, annotations)
            candidate_path = root / "candidates.json"
            atomic_json(candidate_path, {"candidates": [PARAMETER_DEFAULTS]})
            with self.assertRaisesRegex(ValueError, "separate, nonempty source groups"):
                calibrate_asl_thresholds(project, role_map, candidate_path, root / "out")


if __name__ == "__main__":
    unittest.main()
