"""Rule-based classification of a few static hand gestures.

The classifier deliberately has no MediaPipe dependency. It accepts any
sequence of 21 objects with ``x`` and ``y`` attributes, including MediaPipe
NormalizedLandmark objects. Keeping this module independent makes the gesture
logic easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot
from typing import Mapping, Protocol, Sequence


class Landmark(Protocol):
    """Minimal interface used from a MediaPipe hand landmark."""

    x: float
    y: float


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class GesturePrediction:
    """A gesture label together with the detected finger state."""

    label: str
    extended_fingers: Mapping[str, bool]


def _distance(first: Landmark, second: Landmark) -> float:
    return hypot(first.x - second.x, first.y - second.y)


def _joint_angle(first: Landmark, middle: Landmark, last: Landmark) -> float:
    """Return the angle at ``middle`` in degrees."""

    first_vector = (first.x - middle.x, first.y - middle.y)
    last_vector = (last.x - middle.x, last.y - middle.y)
    first_length = hypot(*first_vector)
    last_length = hypot(*last_vector)

    if first_length == 0 or last_length == 0:
        return 0.0

    cosine = (
        first_vector[0] * last_vector[0]
        + first_vector[1] * last_vector[1]
    ) / (first_length * last_length)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def _finger_is_extended(
    landmarks: Sequence[Landmark],
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
) -> bool:
    """Estimate whether a non-thumb finger is straight and away from the wrist."""

    wrist = landmarks[0]
    pip_angle = _joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
    dip_angle = _joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])
    points_away_from_wrist = (
        _distance(wrist, landmarks[tip])
        > 1.15 * _distance(wrist, landmarks[pip])
    )

    return pip_angle >= 150.0 and dip_angle >= 150.0 and points_away_from_wrist


def _thumb_is_extended(landmarks: Sequence[Landmark]) -> bool:
    """Estimate whether the thumb is straight and away from the wrist."""

    wrist = landmarks[0]
    mcp_angle = _joint_angle(landmarks[1], landmarks[2], landmarks[3])
    ip_angle = _joint_angle(landmarks[2], landmarks[3], landmarks[4])
    points_away_from_wrist = (
        _distance(wrist, landmarks[4])
        > 1.05 * _distance(wrist, landmarks[3])
    )

    return mcp_angle >= 140.0 and ip_angle >= 140.0 and points_away_from_wrist


def detect_extended_fingers(landmarks: Sequence[Landmark]) -> dict[str, bool]:
    """Convert 21 hand landmarks into five boolean finger states."""

    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}.")

    return {
        "thumb": _thumb_is_extended(landmarks),
        "index": _finger_is_extended(landmarks, 5, 6, 7, 8),
        "middle": _finger_is_extended(landmarks, 9, 10, 11, 12),
        "ring": _finger_is_extended(landmarks, 13, 14, 15, 16),
        "pinky": _finger_is_extended(landmarks, 17, 18, 19, 20),
    }


def classify_finger_states(extended_fingers: Mapping[str, bool]) -> str:
    """Map finger states to one of five gestures or ``Unknown``."""

    missing = set(FINGER_NAMES) - set(extended_fingers)
    if missing:
        raise ValueError(f"Missing finger states: {', '.join(sorted(missing))}.")

    thumb = extended_fingers["thumb"]
    index = extended_fingers["index"]
    middle = extended_fingers["middle"]
    ring = extended_fingers["ring"]
    pinky = extended_fingers["pinky"]
    non_thumb_count = sum((index, middle, ring, pinky))

    if non_thumb_count == 4:
        return "Open palm"
    if not any((thumb, index, middle, ring, pinky)):
        return "Fist"
    if index and not any((middle, ring, pinky)):
        return "Pointing"
    if index and middle and not ring and not pinky:
        return "Victory"
    if thumb and non_thumb_count == 0:
        return "Thumbs up"

    return "Unknown"


def classify_landmarks(landmarks: Sequence[Landmark]) -> GesturePrediction:
    """Classify a MediaPipe-compatible collection of 21 landmarks."""

    extended_fingers = detect_extended_fingers(landmarks)
    return GesturePrediction(
        label=classify_finger_states(extended_fingers),
        extended_fingers=extended_fingers,
    )

