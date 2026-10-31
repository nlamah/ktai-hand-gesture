"""Reference matrix persistence, export, and distinction checks."""
import json
from pathlib import Path
import tempfile
import unittest

from gesturelab.features.asl_references import (
    REFERENCE_PATH, default_reference_matrix, export_matrix_csv, export_matrix_html,
    export_reference_bundle, load_reference_matrix, quality_report, save_reference_matrix,
)
from gesturelab.features.asl_schema import FEATURE_SCHEMA


class ASLReferenceTests(unittest.TestCase):
    def test_checked_in_chart_draft_has_proposals_for_every_letter_and_feature(self):
        matrix = load_reference_matrix(REFERENCE_PATH)
        self.assertEqual(matrix["labels"], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        expected_features = {feature["key"] for feature in FEATURE_SCHEMA}
        for entry in matrix["letters"].values():
            pattern = entry["patterns"][0]
            self.assertEqual(pattern["status"], "concept")
            self.assertEqual(set(pattern["features"]), expected_features)
            self.assertTrue(all(
                cell["state"] == "specified" and cell["allowed"]
                for cell in pattern["features"].values()
            ))
            self.assertIn("assets/asl_alphabet.jpg", pattern["source"])
        a = matrix["letters"]["A"]["patterns"][0]["features"]
        self.assertEqual(a["thumb_mcp_position"]["allowed"], ["neutral"])
        self.assertEqual(a["flexion_thumb_ip"]["allowed"], ["extended"])
        x = matrix["letters"]["X"]["patterns"][0]["features"]
        self.assertEqual(x["flexion_index_pip"]["allowed"], ["extended", "partly_bent"])
        self.assertEqual(x["flexion_index_dip"]["allowed"], ["bent"])
        self.assertIn("flexion_index_mcp", x)
        for label in "AXLUV":
            pattern = matrix["letters"][label]["patterns"][0]
            self.assertEqual(pattern["status"], "concept")
            self.assertEqual(set(pattern["features"]), expected_features, label)
        for label in "ACIJ":
            self.assertEqual(
                matrix["letters"][label]["patterns"][0]["features"]["thumb_mcp_position"]["allowed"],
                ["neutral"],
            )
        self.assertEqual(
            matrix["letters"]["B"]["patterns"][0]["features"]["thumb_mcp_position"]["allowed"],
            ["toward_palm"],
        )
        self.assertEqual(
            matrix["letters"]["L"]["patterns"][0]["features"]["thumb_mcp_position"]["allowed"],
            ["away_from_palm"],
        )
        self.assertEqual(
            matrix["letters"]["B"]["patterns"][0]["features"]["flexion_pinky_mcp"]["allowed"],
            ["extended"],
        )
        self.assertEqual(
            matrix["letters"]["A"]["patterns"][0]["features"]["flexion_index_mcp"]["allowed"],
            ["bent"],
        )
        self.assertEqual(
            matrix["letters"]["O"]["patterns"][0]["features"]["thumb_tip_index_proximity"]["allowed"],
            ["contact"],
        )
        self.assertEqual(
            matrix["letters"]["R"]["patterns"][0]["features"]["crossing_index_middle"]["allowed"],
            ["suspected_crossed"],
        )
        self.assertEqual(
            matrix["letters"]["K"]["patterns"][0]["features"]["spread_index_middle"]["allowed"],
            ["spread"],
        )
        self.assertEqual(
            matrix["letters"]["P"]["patterns"][0]["features"]["spread_index_middle"]["allowed"],
            ["spread"],
        )
        self.assertEqual(
            matrix["letters"]["F"]["patterns"][0]["features"]["spread_ring_pinky"]["allowed"],
            ["not_spread", "spread"],
        )
        self.assertIn(
            "variatie",
            matrix["letters"]["F"]["patterns"][0]["features"]["spread_ring_pinky"]["note"].lower(),
        )
        self.assertEqual(
            matrix["letters"]["F"]["patterns"][0]["features"]["thumb_tip_index_proximity"]["allowed"],
            ["contact"],
        )
        self.assertEqual(
            matrix["letters"]["O"]["patterns"][0]["features"]["thumb_tip_index_proximity"]["allowed"],
            ["contact"],
        )
        self.assertEqual(
            matrix["letters"]["Q"]["patterns"][0]["features"]["thumb_tip_index_proximity"]["allowed"],
            ["near", "far"],
        )

    def test_default_matrix_is_unknown_not_inferred_from_letter_labels(self):
        matrix = default_reference_matrix(("A", "B"))
        self.assertEqual(len(matrix["letters"]), 2)
        self.assertEqual(matrix["letters"]["A"]["patterns"][0]["features"], {})
        report = quality_report(matrix)
        self.assertEqual(len(report["issues"]), 6)
        self.assertEqual(len(report["insufficiently_distinguished_pairs"]), 1)

    def test_cell_edit_validation_and_json_roundtrip(self):
        matrix = default_reference_matrix(("A", "B"))
        pattern = matrix["letters"]["A"]["patterns"][0]
        pattern["features"]["flexion_index_pip"] = {
            "state": "specified", "allowed": ["extended"], "note": "Gecontroleerde stilstaande referentie.",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "references.json"
            save_reference_matrix(matrix, path)
            self.assertEqual(load_reference_matrix(path), matrix)
        pattern["status"] = "verified"
        with self.assertRaisesRegex(ValueError, "source, reviewer, and note"):
            save_reference_matrix(matrix, Path("not-written.json"))

    def test_multiple_allowed_categories_and_patterns_are_preserved(self):
        matrix = default_reference_matrix(("A",))
        matrix["letters"]["A"]["patterns"].append({
            "name": "variant", "status": "concept", "source": "", "reviewer": "",
            "note": "", "features": {
                "spread_index_middle": {
                    "state": "specified", "allowed": ["spread", "not_spread"], "note": "",
                },
            },
        })
        self.assertEqual(len(matrix["letters"]["A"]["patterns"]), 2)
        self.assertIn("gespreid / niet gespreid", export_matrix_csv(matrix))

    def test_only_disjoint_checked_features_count_as_distinguishing(self):
        matrix = default_reference_matrix(("A", "B"))
        key = "flexion_thumb_ip"
        matrix["letters"]["A"]["patterns"][0]["features"][key] = {
            "state": "specified", "allowed": ["extended"], "note": "",
        }
        matrix["letters"]["B"]["patterns"][0]["features"][key] = {
            "state": "specified", "allowed": ["bent"], "note": "",
        }
        result = quality_report(matrix)
        self.assertTrue(result["letter_pairs"][0]["distinguishable"])
        self.assertEqual(result["letter_pairs"][0]["distinguishing_features"], [key])
        matrix["letters"]["B"]["patterns"][0]["features"][key]["allowed"] = ["extended"]
        self.assertFalse(quality_report(matrix)["letter_pairs"][0]["distinguishable"])

    def test_every_allowed_pattern_pair_must_be_distinguished(self):
        matrix = default_reference_matrix(("A", "B"))
        first = "flexion_thumb_ip"
        second = "flexion_index_mcp"
        matrix["letters"]["A"]["patterns"].append({
            "name": "variant", "status": "concept", "source": "", "reviewer": "",
            "note": "", "features": {
                first: {"state": "specified", "allowed": ["bent"], "note": ""},
                second: {"state": "specified", "allowed": ["extended"], "note": ""},
            },
        })
        matrix["letters"]["A"]["patterns"][0]["features"][first] = {
            "state": "specified", "allowed": ["extended"], "note": "",
        }
        matrix["letters"]["B"]["patterns"][0]["features"][first] = {
            "state": "specified", "allowed": ["extended"], "note": "",
        }
        result = quality_report(matrix)["letter_pairs"][0]
        self.assertIn(first, result["distinguishing_features"])
        self.assertFalse(result["distinguishable"])
        self.assertEqual(len(result["pattern_pairs"]), 2)

    def test_json_bundle_contains_reference_and_versioned_schema(self):
        bundle = export_reference_bundle(default_reference_matrix(("A",)))
        self.assertEqual(bundle["schema"]["version"], "asl-categorical-v4")
        self.assertEqual(bundle["references"]["labels"], ["A"])

    def test_html_export_marks_unreviewed_values_and_escapes_text(self):
        matrix = default_reference_matrix(("<A>",))
        matrix["letters"]["<A>"]["example"] = None
        html = export_matrix_html(matrix)
        self.assertIn("Oningevulde referentiecellen", html)
        self.assertIn("&lt;A&gt;", html)
        self.assertNotIn("<td><A></td>", html)
        self.assertIn("feature_schema_version", json.dumps(matrix))


if __name__ == "__main__":
    unittest.main()
