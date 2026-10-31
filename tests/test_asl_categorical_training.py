"""Synthetic-only tests for ASL categorical classifier training and inference."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib

from gesturelab.datasets.data import atomic_json
from gesturelab.features.asl_models import (
    INPUT_MODE, input_columns, model_features_for_asl_bundle,
)
from gesturelab.features.asl_roles import create_asl_role_map
from gesturelab.features.asl_schema import FEATURE_SCHEMA
from gesturelab.features.training_inputs import validate_model_input
from gesturelab.research.models import ModelHandle, register_model
from gesturelab.training.asl_feature_analysis import feature_ablation, permutation_importance
from gesturelab.training.asl_categorical import _classifier, train_asl_categorical
from gesturelab.training.asl_comparison import compare_asl_runs
from gesturelab.training.experiments import predict_bundle
from gesturelab.training.neural_network import BalancedMLPClassifier
from sklearn.svm import SVC


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


class ASLCategoricalTrainingTests(unittest.TestCase):
    def test_classifier_routes_have_fixed_distinct_estimators(self):
        self.assertIsInstance(_classifier("svm", 8, 42), SVC)
        self.assertIsInstance(_classifier("mlp", 8, 42), BalancedMLPClassifier)
        with self.assertRaisesRegex(ValueError, "No trainable classifier"):
            _classifier("rules", 8, 42)

    def test_synthetic_svm_run_saves_valid_webcam_input_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            world, image = _synthetic_landmarks()
            samples = []
            for block in range(5):
                group = f"synthetic-group-{block}"
                for label in ("A", "B"):
                    samples.append({
                        "id": f"{label}-{block}", "label": label,
                        "participant": "synthetic", "session": group,
                        "condition": "synthetic", "role": "development",
                        "included": True, "status": "ok",
                        "width": 320, "height": 320,
                        "image_landmarks": image, "world_landmarks": world,
                        "provenance": {
                            "type": "synthetic", "source_group": group,
                        },
                    })
            atomic_json(project / "manifest.json", {"schema": 2, "samples": samples})
            role_map = create_asl_role_map(
                project, root / "roles.json", grouping="source", folds=5, seed=42)
            output = train_asl_categorical(
                project, role_map, root / "trained", model="svm", seed=42)
            combined_output = train_asl_categorical(
                project, role_map, root / "trained-combined", model="svm", seed=42,
                input_variant="categorical_plus_shin2021")
            comparison = compare_asl_runs(
                {"categorical": output, "categorical+shin": combined_output},
                root / "comparison")
            summary_lines = (comparison / "summary.csv").read_text().splitlines()
            self.assertEqual(len(summary_lines), 3)
            self.assertTrue((comparison / "paired_validation_predictions.csv").is_file())
            self.assertTrue((comparison / "report.html").is_file())
            report = json.loads((output / "training.json").read_text())
            bundle = joblib.load(output / "model.joblib")
            validate_model_input(bundle)
            values = model_features_for_asl_bundle(bundle, samples[0])
            prediction = predict_bundle(bundle, [values])[0]
            self.assertIn(prediction, ("A", "B"))
            registration = register_model(
                output / "model.joblib", root / "registered.json",
                experiment=output, alphabet="ASL",
            )
            handle = ModelHandle.load(registration)
            self.assertIn(handle.predict(samples[0], 320, 320), ("A", "B"))
            self.assertFalse(report["test_evaluated"])
            self.assertTrue(report["validation"]["confusion_matrix"])
            self.assertTrue((output / "validation_predictions.csv").is_file())
            importance = permutation_importance(
                output, root / "importance", repeats=2, seed=7)
            ablation = feature_ablation(output, root / "ablation", seed=7)
            self.assertTrue((importance / "permutation_importance.csv").is_file())
            ablation_rows = (ablation / "feature_ablation.csv").read_text().splitlines()
            self.assertEqual(len(ablation_rows), 2 + len(FEATURE_SCHEMA))

    def test_fixed_shin_feature_counts_and_composition(self):
        categories = input_columns("categorical")
        shin = input_columns("shin2021_combined")
        combined = input_columns("categorical_plus_shin2021")
        self.assertEqual(len(shin), 820)
        self.assertEqual(len(combined), len(categories) + 820)
        self.assertEqual(combined[:len(categories)], categories)
        self.assertEqual(combined[len(categories):], shin)
        self.assertEqual(INPUT_MODE, "asl_onehot_and_shin2021_v4")
        world, image = _synthetic_landmarks()
        sample = {"world_landmarks": world, "image_landmarks": image, "width": 320, "height": 320}
        bundle = {
            "input_mode": INPUT_MODE,
            "config": {"input_variant": "shin2021_combined"},
        }
        extracted = model_features_for_asl_bundle(bundle, sample)
        self.assertEqual(tuple(extracted), shin)
        self.assertEqual(len(extracted), 820)


if __name__ == "__main__":
    unittest.main()
