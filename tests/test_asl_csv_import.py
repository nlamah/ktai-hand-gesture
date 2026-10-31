"""Synthetic import tests for the generic external ASL CSV adapter."""
import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from gesturelab.datasets.asl_csv import CSV_COLUMNS, import_asl_csv
from gesturelab.datasets.data import read_manifest


class ASLCSVImportTests(unittest.TestCase):
    def _fixture(self, root, entries):
        image_root = root / "images"
        image_root.mkdir()
        for index, entry in enumerate(entries):
            image = Image.new("RGB", (8, 8), (index * 50, 20, 10))
            image.save(image_root / entry["image_path"])
        csv_path = root / "dataset.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(entries)
        return csv_path, image_root

    @staticmethod
    def _entry(path, label, group, role="development"):
        return {
            "image_path": path, "label": label, "participant": "p1",
            "session": group, "condition": "baseline", "role": role,
            "source_group": group,
        }

    def test_import_preserves_provenance_and_leaves_images_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = [
                self._entry("a.png", "A", "group-a"),
                self._entry("b.png", "B", "group-b", role="test"),
            ]
            csv_path, image_root = self._fixture(root, entries)
            project = root / "imported"
            result = import_asl_csv(csv_path, image_root, project, "user-supplied-set")
            samples = read_manifest(project)["samples"]
            self.assertEqual(result["added"], 2)
            self.assertEqual({sample["status"] for sample in samples}, {"pending"})
            self.assertEqual(samples[0]["provenance"]["source_name"], "user-supplied-set")
            self.assertEqual(samples[1]["role"], "test")
            self.assertNotIn("world_landmarks", samples[0])

    def test_duplicate_pixels_cannot_cross_source_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = [
                self._entry("a.png", "A", "group-a"),
                self._entry("b.png", "A", "group-b"),
            ]
            csv_path, image_root = self._fixture(root, entries)
            Image.open(image_root / "a.png").save(image_root / "b.png")
            with self.assertRaisesRegex(ValueError, "duplicates image pixels"):
                import_asl_csv(csv_path, image_root, root / "imported", "synthetic")

    def test_paths_must_remain_under_the_declared_image_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(outside)
            csv_path = root / "dataset.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(self._entry("../outside.png", "A", "group-a"))
            image_root = root / "images"
            image_root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside --images-root"):
                import_asl_csv(csv_path, image_root, root / "imported", "synthetic")


if __name__ == "__main__":
    unittest.main()
