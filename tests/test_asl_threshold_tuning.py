"""Tests for grouped, train-only workbook threshold tuning."""
import unittest

import numpy as np

from gesturelab.features.asl_excel_review import categorize_workbook_features
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS
from gesturelab.features.asl_threshold_tuning import (
    _PARAMETER_GRIDS,
    _grouped_split,
    _score,
)
from gesturelab.features.categorical_hand import (
    extract_categorical_features,
    measure_hand_features,
    validate_parameters,
)


def _synthetic_hand():
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[0] = [0.0, 0.0, 0.0]
    points[1:5] = [[-.25, .1, 0], [-.4, .25, 0], [-.55, .4, 0], [-.7, .55, 0]]
    for base, x in ((5, -.45), (9, -.15), (13, .15), (17, .45)):
        points[base] = [x, .5, 0]
        points[base + 1] = [x, .8, 0]
        points[base + 2] = [x, 1.1, 0]
        points[base + 3] = [x, 1.4, 0]
    return points


class ASLThresholdTuningTests(unittest.TestCase):
    def test_every_search_candidate_is_a_valid_threshold_configuration(self):
        for group, candidates in _PARAMETER_GRIDS.items():
            for candidate in candidates:
                parameters = {
                    key: dict(value) for key, value in PARAMETER_DEFAULTS.items()
                }
                parameters[group].update(candidate)
                validate_parameters(parameters)

    def test_reference_category_mapping_matches_review_extractor(self):
        world = _synthetic_hand()
        measurements = measure_hand_features(world, "Right")
        categories = categorize_workbook_features(measurements)
        expected = extract_categorical_features(world, handedness="Right")
        self.assertEqual(
            {key: value for key, value in categories.items() if not key.startswith(("flexion_", "spread_")) and key != "thumb_mcp_shape"},
            {key: value for key, value in expected.items() if not key.startswith(("flexion_", "spread_"))},
        )
        self.assertIn(categories["thumb_mcp_shape"], {
            "B-vormig", "not_B-vormig", "unknown",
        })

    def test_three_finger_states_preserve_thumb_and_unknowns(self):
        measurements = measure_hand_features(_synthetic_hand(), "Right")
        parameters = {"finger_flexion": {"extended_min_deg": 150, "bent_max_deg": 110}}
        for angle, expected in ((110, "bent"), (110.1, "partly_bent"), (149.9, "partly_bent"),
                                (150, "extended"), (None, "unknown")):
            measurements["index_pip_angle_deg"] = angle
            measurements["thumb_ip_angle_deg"] = 130
            categories = categorize_workbook_features(measurements, parameters)
            self.assertEqual(categories["flexion_index_pip"], expected)
            self.assertEqual(categories["flexion_thumb_ip"], "partly_bent")

    def test_spread_requires_gap_shared_plane_and_unfolded_fingers(self):
        measurements = measure_hand_features(_synthetic_hand(), "Right")
        parameters = {"workbook_spread": {"minimum_gap_ratio": 0.2,
                                           "maximum_depth_ratio": 0.4}}
        prefix = "spread_index_middle"
        measurements.update({prefix + "_gap_ratio": 0.3,
                             prefix + "_depth_ratio": 0.1,
                             prefix + "_direction_cosine": 0.8})
        self.assertEqual(categorize_workbook_features(measurements, parameters)[prefix], "spread")
        for key, value in ((prefix + "_gap_ratio", 0.05),
                           (prefix + "_gap_ratio", -0.3),
                           (prefix + "_depth_ratio", 0.6),
                           (prefix + "_direction_cosine", -0.1),
                           ("middle_pip_angle_deg", 90),
                           ("index_dip_angle_deg", 100)):
            altered = {**measurements, key: value}
            self.assertEqual(categorize_workbook_features(altered, parameters)[prefix], "not_spread")
        altered = {**measurements, prefix + "_gap_ratio": None}
        self.assertEqual(categorize_workbook_features(altered, parameters)[prefix], "unknown")

    def test_crossing_at_dip_is_found_when_tips_return_to_original_order(self):
        world = np.array(_synthetic_hand())
        world[7, 0] = 0.05
        world[11, 0] = -0.35
        world[8, 0] = -0.2
        parameters = {"workbook_crossing": {"minimum_swap_ratio": 0.02,
                                              "minimum_joint_angle_deg": 0}}
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for transformed in (world, world * [-1, 1, 1], world @ rotation * 3 + 2):
            measurements = measure_hand_features(transformed, "Right")
            original = extract_categorical_features(transformed, handedness="Right")
            self.assertEqual(original["crossing_index_middle"], "not_crossed")
            self.assertEqual(categorize_workbook_features(measurements, parameters)["crossing_index_middle"],
                             "suspected_crossed")

    def test_crossing_rejects_folded_fingers_and_depth_separation(self):
        measurements = measure_hand_features(_synthetic_hand(), "Right")
        measurements.update(crossing_profile_gap_ratio=-0.1,
                            crossing_profile_depth_ratio=0.1)
        self.assertEqual(categorize_workbook_features(measurements)["crossing_index_middle"],
                         "suspected_crossed")
        for key, value in (("index_pip_angle_deg", 80),
                           ("crossing_profile_depth_ratio", 1),
                           ("crossing_profile_gap_ratio", 0)):
            self.assertEqual(categorize_workbook_features({**measurements, key: value})["crossing_index_middle"],
                             "not_crossed")
        self.assertEqual(categorize_workbook_features({**measurements, "crossing_profile_gap_ratio": None})["crossing_index_middle"],
                         "unknown")

    def test_binary_thumb_uses_its_own_trained_threshold(self):
        measurements = measure_hand_features(_synthetic_hand(), "Right")
        for deviation, expected in ((21, "B-vormig"), (20, "not_B-vormig"),
                                    (-40, "not_B-vormig"), (None, "unknown")):
            measurements["thumb_mcp_signed_deviation_deg"] = deviation
            self.assertEqual(categorize_workbook_features(measurements)["thumb_mcp_shape"], expected)

    def test_grouped_split_keeps_augmentations_together(self):
        samples = [
            {
                "filename": f"letter-{group}.rf.aug-{augmentation}.jpg",
                "label": "A",
            }
            for group in range(15)
            for augmentation in range(3)
        ]
        training, validation, total_groups, holdout_groups, training_groups = _grouped_split(
            samples,
        )
        train_names = {
            sample["filename"].split(".rf.", 1)[0] for sample in training
        }
        validation_names = {
            sample["filename"].split(".rf.", 1)[0] for sample in validation
        }
        self.assertEqual(total_groups, 15)
        self.assertEqual(len(train_names), training_groups)
        self.assertEqual(len(validation_names), holdout_groups)
        self.assertFalse(train_names & validation_names)
        self.assertEqual(len(training) + len(validation), len(samples))

    def test_no_contact_reference_accepts_near_or_far_category(self):
        measurements = measure_hand_features(_synthetic_hand(), "Right")
        score = _score([(
            {"id": "sample", "label": "A"},
            {"thumb_tip_index_proximity": {"allowed": ["no_contact"]}},
            measurements,
        )], {})
        self.assertEqual(score["exact_match_images"], 1)
        self.assertEqual(score["scored_feature_values"], 1)
        self.assertEqual(
            score["feature_macro_f1"]["thumb_tip_index_proximity"], 1.0,
        )


if __name__ == "__main__":
    unittest.main()
