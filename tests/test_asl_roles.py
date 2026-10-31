"""Synthetic grouped-role assignment checks; no experiment is executed."""
import json
from pathlib import Path
import tempfile
import unittest

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_roles import create_asl_role_map


class ASLRoleTests(unittest.TestCase):
    def test_roles_hold_out_entire_source_groups_and_reserve_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            samples = []
            for block in range(5):
                group = f"block-{block}"
                for label in ("A", "B"):
                    samples.append({
                        "id": f"{label}-{block}", "label": label,
                        "participant": "unknown", "session": group,
                        "role": "development", "included": True,
                        "status": "ok",
                        "provenance": {"source_group": group},
                    })
            samples.append({
                "id": "test-a", "label": "A", "participant": "unknown",
                "session": "test", "role": "test", "included": True,
                "status": "pending", "provenance": {"source_group": "test"},
            })
            atomic_json(project / "manifest.json", {"schema": 2, "samples": samples})

            path = create_asl_role_map(project, root / "roles.json", folds=5)
            role_map = json.loads(path.read_text())
            train = {
                sample_id for sample_id, role in role_map["roles"].items()
                if role == "calibration_train"
            }
            validation = {
                sample_id for sample_id, role in role_map["roles"].items()
                if role == "calibration_validation"
            }
            self.assertTrue(train)
            self.assertTrue(validation)
            self.assertEqual(role_map["roles"]["test-a"], "final_test")
            self.assertFalse({
                role_map["groups"][sample_id] for sample_id in train
            } & {
                role_map["groups"][sample_id] for sample_id in validation
            })
            self.assertIn("No feature calibration", role_map["note"])


if __name__ == "__main__":
    unittest.main()
