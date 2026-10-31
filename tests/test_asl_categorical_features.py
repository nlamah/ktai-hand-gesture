"""Synthetic arithmetic and schema checks for categorical ASL hand features."""
import unittest

import numpy as np

from gesturelab.features.asl_schema import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, LETTERS
from gesturelab.features.categorical_hand import (
    extract_categorical_features, measure_hand_features, validate_parameters,
)


def synthetic_hand():
    points = np.zeros((21, 3), dtype=float)
    points[0] = (0, 0, 0)
    points[1:5] = [(-.25, .1, 0), (-.4, .25, 0), (-.55, .4, 0), (-.7, .55, 0)]
    for base, x in ((5, -.45), (9, -.15), (13, .15), (17, .45)):
        points[base] = (x, .5, 0)
        points[base + 1] = (x, .8, 0)
        points[base + 2] = (x, 1.1, 0)
        points[base + 3] = (x, 1.4, 0)
    return points


class CategoricalHandTests(unittest.TestCase):
    def test_schema_has_unique_feature_keys_and_full_label_set(self):
        keys = [feature["key"] for feature in FEATURE_SCHEMA]
        self.assertEqual(len(keys), 23)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(any(key.startswith("orientation_") for key in keys))
        spread = next(feature for feature in FEATURE_SCHEMA if feature["key"] == "spread_thumb_index")
        self.assertEqual(
            [category["id"] for category in spread["categories"]],
            ["not_spread", "spread", "unknown"],
        )
        self.assertEqual(LETTERS, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for finger in ("index", "middle", "ring", "pinky"):
            self.assertIn(f"flexion_{finger}_mcp", keys)
        self.assertEqual(
            sum(feature["group"] == "finger_flexion" for feature in FEATURE_SCHEMA),
            13,
        )
        thumb_mcp = next(feature for feature in FEATURE_SCHEMA if feature["key"] == "thumb_mcp_position")
        self.assertEqual(
            [category["id"] for category in thumb_mcp["categories"]],
            ["toward_palm", "neutral", "away_from_palm", "unknown"],
        )
        self.assertEqual(FEATURE_SCHEMA_VERSION, "asl-categorical-v4")

    def test_thumb_mcp_position_uses_bend_direction_and_handedness(self):
        def bent_thumb(degrees):
            world = synthetic_hand()
            base = world[2] - world[1]
            angle = np.radians(-degrees)
            rotation = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ])
            world[3] = world[2] + rotation @ base
            return world

        toward = bent_thumb(35)
        measurements = measure_hand_features(toward, "Right")
        self.assertAlmostEqual(measurements["thumb_mcp_signed_deviation_deg"], 35)
        self.assertEqual(
            extract_categorical_features(toward, handedness="Right")["thumb_mcp_position"],
            "toward_palm",
        )
        self.assertEqual(
            extract_categorical_features(bent_thumb(-35), handedness="Right")["thumb_mcp_position"],
            "away_from_palm",
        )
        self.assertEqual(
            extract_categorical_features(bent_thumb(10), handedness="Right")["thumb_mcp_position"],
            "neutral",
        )

        mirrored = toward.copy()
        mirrored[:, 0] *= -1
        self.assertEqual(
            extract_categorical_features(mirrored, handedness="Left")["thumb_mcp_position"],
            "toward_palm",
        )
        self.assertEqual(
            extract_categorical_features(toward)["thumb_mcp_position"],
            "unknown",
        )

    def test_straight_and_bent_joint_measurements_are_separate(self):
        world = synthetic_hand()
        world[5] = (0, 0.5, 0)
        world[6] = (0, 0.8, 0)
        world[7] = (0, 1.1, 0)
        world[8] = (0, 1.4, 0)
        measurements = measure_hand_features(world)
        self.assertAlmostEqual(measurements["index_mcp_angle_deg"], 180.0)
        self.assertAlmostEqual(measurements["index_pip_angle_deg"], 180.0)
        self.assertAlmostEqual(measurements["index_dip_angle_deg"], 180.0)
        categories = extract_categorical_features(world)
        self.assertEqual(categories["flexion_index_mcp"], "extended")
        self.assertEqual(categories["flexion_index_pip"], "extended")
        self.assertEqual(categories["flexion_index_dip"], "extended")

        world[8] = world[7] + np.array([.3, 0, 0])
        measurements = measure_hand_features(world)
        self.assertAlmostEqual(measurements["index_dip_angle_deg"], 90.0)
        categories = extract_categorical_features(world)
        self.assertEqual(categories["flexion_index_pip"], "extended")
        self.assertEqual(categories["flexion_index_dip"], "bent")

        world = synthetic_hand()
        world[5] = (0, 0.5, 0)
        world[6] = (0.3, 0.5, 0)
        categories = extract_categorical_features(world)
        self.assertEqual(categories["flexion_index_mcp"], "bent")

    def test_measurements_are_scale_and_translation_invariant(self):
        world = synthetic_hand()
        base = measure_hand_features(world)
        changed = measure_hand_features(world * 3.2 + 7)
        for key in ("index_pip_angle_deg", "spread_index_middle_ratio",
                    "thumb_tip_index_distance_ratio", "crossing_base_lateral_ratio"):
            self.assertAlmostEqual(base[key], changed[key])
        self.assertEqual(
            extract_categorical_features(world),
            extract_categorical_features(world * 3.2 + 7),
        )

    def test_spread_is_binary_and_uses_configured_threshold(self):
        world = synthetic_hand()
        measurements = measure_hand_features(world)
        ratio = measurements["spread_index_middle_ratio"]
        self.assertEqual(
            extract_categorical_features(world, {"spread": {"minimum_spread_ratio": ratio}})
            ["spread_index_middle"],
            "spread",
        )
        self.assertEqual(
            extract_categorical_features(
                world, {"spread": {"minimum_spread_ratio": ratio + .01}}
            )["spread_index_middle"],
            "not_spread",
        )

    def test_invalid_geometry_and_thresholds_are_rejected(self):
        world = synthetic_hand()
        invalid = world.copy()
        invalid[0] = invalid[9]
        with self.assertRaises(ValueError):
            extract_categorical_features(invalid)
        with self.assertRaises(ValueError):
            measure_hand_features(world[:, :2])
        with self.assertRaises(ValueError):
            validate_parameters({"flexion": {"bent_max_deg": 170}})
        with self.assertRaises(ValueError):
            validate_parameters({"spread": {"minimum_spread_ratio": -0.1}})

    def test_undefined_local_measurements_become_unknown_per_feature(self):
        world = synthetic_hand()
        world[8] = world[7]
        categories = extract_categorical_features(world)
        self.assertEqual(categories["flexion_index_dip"], "unknown")
        self.assertNotEqual(categories["flexion_index_pip"], "unknown")

        world = synthetic_hand()
        world[5] = world[17]
        categories = extract_categorical_features(world)
        self.assertEqual(categories["crossing_index_middle"], "unknown")
        self.assertNotEqual(categories["spread_index_middle"], "unknown")


if __name__ == "__main__":
    unittest.main()
