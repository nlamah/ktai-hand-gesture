"""Tests for importing the ASL feature workbook and scoring landmark results."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook
import numpy as np
from PIL import Image

from gesturelab.features.asl_excel_review import (
    _expected_values,
    compare_sample,
    load_reference_workbook,
    read_coco_split,
)


HEADERS = (
    "Letter", "Duim MCP", "Duim IP", "Wijsvinger MCP", "Wijs PIP", "wijs DIP",
    "middel MCP", "middel PIP", "middel DIP", "Ring MCP", "Ring PIP", "Ring DIP",
    "Pink MCP", "Pink PIP", "Pink Dip", "Spreiding wijs - middel",
    "spreiding middel - ring", "kruising wijs - middel", "duim contact wijs",
    "duim contact middel",
)


def workbook_bytes(contact="nee", thumb_ip="gestrekt/gebogen"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    values = [
        "neutraal", thumb_ip, "gestrekt", "licht gebogen", "gebogen",
        "gestrekt", "gestrekt", "gestrekt", "gebogen", "gebogen", "gebogen",
        "gestrekt", "gestrekt", "gestrekt", "nee", "ja", "nee", contact, "nee",
    ]
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        sheet.append([label, *values])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def synthetic_world_landmarks():
    points = np.zeros((21, 3), dtype=float)
    points[0] = (0, 0, 0)
    points[1:5] = [(-.25, .1, 0), (-.4, .25, 0), (-.55, .4, 0), (-.7, .55, 0)]
    for base, x in ((5, -.45), (9, -.15), (13, .15), (17, .45)):
        points[base] = (x, .5, 0)
        points[base + 1] = (x, .8, 0)
        points[base + 2] = (x, 1.1, 0)
        points[base + 3] = (x, 1.4, 0)
    return points


class ASLExcelReviewTests(unittest.TestCase):
    def test_workbook_maps_all_letters_and_alternative_values(self):
        references = load_reference_workbook(workbook_bytes())
        self.assertEqual(len(references), 26)
        self.assertEqual(len(references["A"]), 19)
        self.assertEqual(
            references["A"]["flexion_thumb_ip"]["allowed"],
            ["bent", "extended"],
        )
        self.assertEqual(
            references["A"]["flexion_index_pip"]["allowed"],
            ["partly_bent"],
        )
        self.assertEqual(
            references["A"]["spread_index_middle"]["allowed"],
            ["not_spread"],
        )
        self.assertEqual(
            references["A"]["thumb_mcp_shape"]["allowed"],
            ["not_B-vormig"],
        )

    def test_project_workbook_has_updated_letter_references(self):
        path = Path(__file__).resolve().parents[1] / "assets/asl_feature_references.xlsx"
        references = load_reference_workbook(path.read_bytes())
        self.assertEqual(references["U"]["spread_index_middle"]["allowed"], ["not_spread"])
        for letter in ("V", "K"):
            self.assertEqual(references[letter]["spread_index_middle"]["allowed"], ["spread"])
        self.assertEqual(references["K"]["flexion_middle_mcp"]["allowed"], ["bent"])
        self.assertEqual(references["C"]["flexion_index_mcp"]["allowed"], ["extended", "partly_bent"])

    def test_slash_flexion_values_accept_only_listed_states(self):
        self.assertEqual(_expected_values("flexion_index_pip", "gebogen/gestrekt"),
                         ["bent", "extended"])
        self.assertEqual(_expected_values("flexion_index_pip", "licht gebogen/gestrekt"),
                         ["extended", "partly_bent"])
        self.assertEqual(_expected_values("flexion_index_pip", "gebogen"), ["bent"])
        self.assertEqual(_expected_values("flexion_index_pip", "licht gebogen"), ["partly_bent"])

    def test_optional_workbook_values_are_not_scored(self):
        references = load_reference_workbook(workbook_bytes(contact="kan"))
        sample = {
            "id": "sample", "filename": "sample.jpg", "label": "A",
            "image_path": "sample.jpg", "status": "no_hand", "n_hands": 0,
        }
        result = compare_sample(sample, references)
        self.assertEqual(result["image_status"], "needs_detection_review")
        self.assertEqual(
            result["comparisons"]["thumb_tip_index_proximity"]["status"],
            "not_scored",
        )
        self.assertEqual(result["comparisons"]["flexion_thumb_ip"]["status"], "unknown")

    def test_no_contact_expectation_accepts_near_and_far_but_not_contact(self):
        references = load_reference_workbook(workbook_bytes(contact="nee"))
        sample = {
            "id": "sample", "filename": "sample.jpg", "label": "A",
            "image_path": "sample.jpg", "status": "ok", "n_hands": 1,
            "handedness": "Right",
            "world_landmarks": synthetic_world_landmarks().tolist(),
        }
        result = compare_sample(sample, references)
        self.assertEqual(
            result["comparisons"]["thumb_tip_index_proximity"]["status"],
            "match",
        )

        sample["world_landmarks"][4] = sample["world_landmarks"][8]
        result = compare_sample(sample, references)
        self.assertEqual(
            result["comparisons"]["thumb_tip_index_proximity"]["status"],
            "mismatch",
        )

    def test_coco_split_checks_single_letter_labels_and_existing_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = Path(temporary)
            Image.new("RGB", (4, 4)).save(split / "sample.jpg")
            document = {
                "categories": [{"id": 1, "name": "A"}],
                "images": [{"id": 1, "file_name": "sample.jpg"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            }
            (split / "_annotations.coco.json").write_text(json.dumps(document))
            samples = read_coco_split(split)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["label"], "A")
            self.assertEqual(samples[0]["filename"], "sample.jpg")

            document["images"][0]["file_name"] = "../outside.jpg"
            (split / "_annotations.coco.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "escapes the split"):
                read_coco_split(split)


if __name__ == "__main__":
    unittest.main()
