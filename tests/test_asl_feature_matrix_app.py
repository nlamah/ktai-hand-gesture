"""Headless checks for the reference editor; app startup must not run ML."""
import ast
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "apps" / "asl_feature_matrix_app.py"


class ASLFeatureMatrixAppTests(unittest.TestCase):
    def test_app_opens_with_unknown_matrix_and_does_not_write_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "references.json"
            app = AppTest.from_file(str(APP), default_timeout=30)
            app.run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertTrue(any("ASL-letterfeaturematrix" in item.value for item in app.title))
            self.assertFalse(matrix_path.exists())
            self.assertTrue(any("geen extractie, tuning of training" in item.value for item in app.markdown))

    def test_overview_displays_proposals_without_question_marks(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        matrix = app.dataframe[0].value
        self.assertEqual(len(matrix), 26)
        self.assertFalse((matrix.iloc[:, 3:] == "?").any().any())

    def test_app_source_does_not_import_training_or_calibration_modules(self):
        tree = ast.parse(APP.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.startswith("gesturelab.training") for name in imports))
        self.assertFalse(any("calibration" in name for name in imports))

    def test_feature_value_dropdown_lists_all_categories_for_selected_feature(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        value_dropdown = next(
            widget for widget in app.multiselect
            if widget.label == "Suggestie / toegestane waarde(n)"
        )
        self.assertEqual(
            value_dropdown.options,
            ["gebogen naar handpalm", "neutraal", "gebogen van hand weg"],
        )
        feature_picker = next(
            widget for widget in app.selectbox
            if widget.label == "Feature om te bewerken"
        )
        feature_picker.select("spread_thumb_index").run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        value_dropdown = next(
            widget for widget in app.multiselect
            if widget.label == "Suggestie / toegestane waarde(n)"
        )
        self.assertEqual(
            value_dropdown.options, ["niet gespreid", "gespreid"])
        feature_picker = next(
            widget for widget in app.selectbox
            if widget.label == "Feature om te bewerken"
        )
        feature_picker.select("flexion_index_mcp").run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        value_dropdown = next(
            widget for widget in app.multiselect
            if widget.label == "Suggestie / toegestane waarde(n)"
        )
        self.assertEqual(value_dropdown.options, ["gestrekt", "licht gebogen", "gebogen"])


if __name__ == "__main__":
    unittest.main()
