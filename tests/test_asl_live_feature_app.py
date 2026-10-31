"""Headless checks for the ASL webcam feature app."""
import ast
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "apps" / "asl_live_feature_check_app.py"


class ASLLiveFeatureAppTests(unittest.TestCase):
    def test_app_renders_typed_letter_and_camera_toggle(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertTrue(any("ASL-features live controleren" in item.value for item in app.title))
        self.assertTrue(any(
            item.label == "Letter die je gaat maken" for item in app.text_input
        ))
        self.assertTrue(any(item.label == "Webcam aan" for item in app.toggle))

    def test_webcam_component_uses_browser_media_and_component_v2_only(self):
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_v1 = any(
            isinstance(node, ast.Attribute)
            and node.attr == "v1"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "components"
            for node in ast.walk(tree)
        )
        self.assertFalse(imported_v1)
        self.assertIn("navigator.mediaDevices.getUserMedia", source)
        self.assertIn('runtime.sendState("frame"', source)


if __name__ == "__main__":
    unittest.main()
