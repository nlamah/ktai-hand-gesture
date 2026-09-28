import unittest
from dataclasses import dataclass

from gesture_classifier import (
    classify_finger_states,
    classify_landmarks,
    detect_extended_fingers,
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def make_hand(*extended: str) -> list[Point]:
    """Build a simple upright synthetic hand with selected straight fingers."""

    landmarks = [Point(0.5, 0.9) for _ in range(21)]

    if "thumb" in extended:
        landmarks[1:5] = [
            Point(0.42, 0.72),
            Point(0.32, 0.62),
            Point(0.22, 0.52),
            Point(0.10, 0.42),
        ]
    else:
        landmarks[1:5] = [
            Point(0.42, 0.72),
            Point(0.37, 0.68),
            Point(0.42, 0.64),
            Point(0.47, 0.69),
        ]

    finger_landmarks = {
        "index": (5, 0.38),
        "middle": (9, 0.48),
        "ring": (13, 0.58),
        "pinky": (17, 0.68),
    }
    for finger, (start, x_coordinate) in finger_landmarks.items():
        if finger in extended:
            landmarks[start:start + 4] = [
                Point(x_coordinate, 0.65),
                Point(x_coordinate, 0.48),
                Point(x_coordinate, 0.31),
                Point(x_coordinate, 0.14),
            ]
        else:
            landmarks[start:start + 4] = [
                Point(x_coordinate, 0.65),
                Point(x_coordinate, 0.50),
                Point(x_coordinate + 0.06, 0.58),
                Point(x_coordinate + 0.02, 0.66),
            ]

    return landmarks


class GestureClassifierTests(unittest.TestCase):
    def classify(self, *extended: str) -> str:
        states = {
            finger: finger in extended
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        }
        return classify_finger_states(states)

    def test_open_palm(self):
        self.assertEqual(
            self.classify("thumb", "index", "middle", "ring", "pinky"),
            "Open palm",
        )

    def test_fist(self):
        self.assertEqual(self.classify(), "Fist")

    def test_pointing(self):
        self.assertEqual(self.classify("index"), "Pointing")

    def test_victory(self):
        self.assertEqual(self.classify("index", "middle"), "Victory")

    def test_thumbs_up(self):
        self.assertEqual(self.classify("thumb"), "Thumbs up")

    def test_unknown_combination(self):
        self.assertEqual(self.classify("thumb", "pinky"), "Unknown")

    def test_missing_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing finger states"):
            classify_finger_states({"index": True})

    def test_landmark_geometry_detects_open_hand(self):
        landmarks = make_hand("thumb", "index", "middle", "ring", "pinky")

        self.assertEqual(
            detect_extended_fingers(landmarks),
            {
                "thumb": True,
                "index": True,
                "middle": True,
                "ring": True,
                "pinky": True,
            },
        )
        self.assertEqual(classify_landmarks(landmarks).label, "Open palm")

    def test_landmark_geometry_detects_victory(self):
        prediction = classify_landmarks(make_hand("index", "middle"))

        self.assertEqual(prediction.label, "Victory")
        self.assertEqual(
            [
                finger
                for finger, is_extended in prediction.extended_fingers.items()
                if is_extended
            ],
            ["index", "middle"],
        )

    def test_wrong_landmark_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 21 hand landmarks"):
            detect_extended_fingers(make_hand()[:-1])


if __name__ == "__main__":
    unittest.main()
