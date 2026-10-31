"""Synthetic categorical encoding and reference-rule checks."""
import unittest

from gesturelab.features.asl_models import (
    ASLReferenceRuleClassifier, FEATURE_COLUMNS, assert_one_hot_rows,
    encode_categories, predict_reference_rule,
)
from gesturelab.features.asl_references import default_reference_matrix
from gesturelab.features.asl_schema import FEATURE_SCHEMA, UNKNOWN


class ASLModelInputTests(unittest.TestCase):
    def test_encoding_is_fixed_width_and_one_hot_per_categorical_feature(self):
        categories = {
            feature["key"]: feature["categories"][0]["id"]
            for feature in FEATURE_SCHEMA
        }
        encoded = encode_categories(categories)
        assert_one_hot_rows([encoded])
        self.assertEqual(len(encoded), len(FEATURE_COLUMNS))
        with self.assertRaisesRegex(ValueError, "does not match"):
            encode_categories({})

    def test_rule_baseline_requires_verified_and_unambiguous_references(self):
        matrix = default_reference_matrix(("A", "B"))
        categories = {
            feature["key"]: feature["categories"][0]["id"]
            for feature in FEATURE_SCHEMA
        }
        selected = FEATURE_SCHEMA[:4]
        pattern = matrix["letters"]["A"]["patterns"][0]
        for feature in selected:
            pattern["features"][feature["key"]] = {
                "state": "specified",
                "allowed": [categories[feature["key"]]],
                "note": "",
            }
        prediction, details = predict_reference_rule(categories, matrix)
        self.assertEqual(prediction, UNKNOWN)
        self.assertEqual(details["reason"], "no_matching_pattern")

        pattern.update(status="verified", source="synthetic", reviewer="reviewer", note="fixture")
        prediction, details = predict_reference_rule(categories, matrix)
        self.assertEqual(prediction, "A")
        self.assertEqual(details["reason"], "matched")

        other = matrix["letters"]["B"]["patterns"][0]
        other.update(status="verified", source="synthetic", reviewer="reviewer", note="fixture")
        other["features"] = {
            key: dict(value) for key, value in pattern["features"].items()
        }
        self.assertEqual(predict_reference_rule(categories, matrix)[0], UNKNOWN)
        other["features"][FEATURE_SCHEMA[0]["key"]]["allowed"] = [
            FEATURE_SCHEMA[0]["categories"][1]["id"]
        ]
        rule_model = ASLReferenceRuleClassifier(matrix)
        encoded = encode_categories(categories)
        row = [[encoded[column] for column in FEATURE_COLUMNS]]
        self.assertEqual(rule_model.predict(row)[0], "A")

    def test_rule_baseline_abstains_when_too_few_features_are_observable(self):
        matrix = default_reference_matrix(("A",))
        pattern = matrix["letters"]["A"]["patterns"][0]
        pattern.update(status="verified", source="synthetic", reviewer="reviewer", note="fixture")
        categories = {}
        for feature in FEATURE_SCHEMA[:5]:
            expected = feature["categories"][0]["id"]
            categories[feature["key"]] = expected
            pattern["features"][feature["key"]] = {
                "state": "specified", "allowed": [expected], "note": "",
            }
        for feature in FEATURE_SCHEMA[1:]:
            categories.setdefault(feature["key"], UNKNOWN)
        prediction, details = predict_reference_rule(categories, matrix)
        self.assertEqual(prediction, "A")
        categories[FEATURE_SCHEMA[0]["key"]] = UNKNOWN
        categories[FEATURE_SCHEMA[1]["key"]] = UNKNOWN
        self.assertEqual(predict_reference_rule(categories, matrix)[0], UNKNOWN)


if __name__ == "__main__":
    unittest.main()
