"""Tests for webcam frame validation and ASL live feature comparisons."""
import base64
from io import BytesIO
import unittest

from openpyxl import Workbook
import numpy as np
from PIL import Image

from gesturelab.features.asl_excel_review import load_reference_workbook
from gesturelab.features.asl_live import decode_live_frame, live_feature_rows
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS


_HEADERS = (
    "Letter", "Duim MCP", "Duim IP", "Wijsvinger MCP", "Wijs PIP", "wijs DIP",
    "middel MCP", "middel PIP", "middel DIP", "Ring MCP", "Ring PIP", "Ring DIP",
    "Pink MCP", "Pink PIP", "Pink Dip", "Spreiding wijs - middel",
    "spreiding middel - ring", "kruising wijs - middel", "duim contact wijs",
    "duim contact middel",
)


def _references(contact="nee"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(_HEADERS)
    values = [
        "neutraal", "gestrekt", "gestrekt", "licht gebogen", "gebogen",
        "gestrekt", "gestrekt", "gestrekt", "gebogen", "gebogen", "gebogen",
        "gestrekt", "gestrekt", "gestrekt", "nee", "ja", "nee", contact, "nee",
    ]
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        sheet.append([label, *values])
    output = BytesIO()
    workbook.save(output)
    return load_reference_workbook(output.getvalue())


def _world_landmarks():
    points = np.zeros((21, 3), dtype=float)
    points[0] = (0, 0, 0)
    points[1:5] = [(-.25, .1, 0), (-.4, .25, 0), (-.55, .4, 0), (-.7, .55, 0)]
    for base, x in ((5, -.45), (9, -.15), (13, .15), (17, .45)):
        points[base] = (x, .5, 0)
        points[base + 1] = (x, .8, 0)
        points[base + 2] = (x, 1.1, 0)
        points[base + 3] = (x, 1.4, 0)
    return points.tolist()


class ASLLiveFeatureTests(unittest.TestCase):
    def test_decodes_bounded_jpeg_webcam_frame(self):
        image = Image.new("RGB", (32, 24), (10, 30, 50))
        payload = BytesIO()
        image.save(payload, format="JPEG")
        decoded = decode_live_frame({
            "id": "frame-1",
            "data": base64.b64encode(payload.getvalue()).decode("ascii"),
        })
        self.assertEqual(decoded.size, (32, 24))
        self.assertEqual(decoded.mode, "RGB")

    def test_rejects_invalid_and_oversized_frames(self):
        with self.assertRaisesRegex(ValueError, "ongeldig"):
            decode_live_frame({"data": "not-base64"})
        with self.assertRaisesRegex(ValueError, "groter"):
            decode_live_frame({"data": "A" * 1_400_000})

    def test_live_feature_rows_report_matches_and_mismatches(self):
        detection = {
            "status": "ok",
            "n_hands": 1,
            "handedness": "Right",
            "world_landmarks": _world_landmarks(),
        }
        rows = live_feature_rows(detection, "A", _references(), PARAMETER_DEFAULTS)
        self.assertEqual(len(rows), 19)
        statuses = {row["key"]: row["status"] for row in rows}
        self.assertEqual(statuses["flexion_index_pip"], "mismatch")
        self.assertEqual(
            next(row for row in rows if row["key"] == "flexion_index_pip")["color"],
            "red",
        )
        self.assertIn("Duim", next(row for row in rows if row["key"] == "thumb_mcp_shape")["title"])
        self.assertEqual(
            next(row for row in rows if row["key"] == "thumb_tip_index_proximity")["color"],
            "green",
        )

    def test_no_hand_uses_red_and_not_scored_uses_muted_color(self):
        rows = live_feature_rows(
            {"status": "no_hand", "n_hands": 0},
            "A",
            _references(contact="kan"),
            PARAMETER_DEFAULTS,
        )
        self.assertTrue(all(row["color"] == "red" for row in rows if row["status"] != "not_scored"))
        self.assertTrue(all(row["color"] == "muted" for row in rows if row["status"] == "not_scored"))


if __name__ == "__main__":
    unittest.main()
