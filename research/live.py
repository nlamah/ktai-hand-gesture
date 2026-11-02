"""Prepare live webcam frames and compare them with ASL workbook features."""
import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from research.workbook import compare_sample
from research.schema import FEATURE_SCHEMA

MAX_FRAME_BYTES = 1_000_000
MAX_FRAME_DIMENSION = 640

_FEATURE_TITLES = {
    feature["key"]: feature["title"] for feature in FEATURE_SCHEMA
}
_FEATURE_TITLES["thumb_mcp_shape"] = "Thumb: MCP shape"
_CATEGORY_LABELS = {
    feature["key"]: {
        category["id"]: category["label"]
        for category in feature["categories"]
    }
    for feature in FEATURE_SCHEMA
}
_CATEGORY_LABELS["thumb_mcp_shape"] = {
    "B-vormig": "B-shaped",
    "not_B-vormig": "Not B-shaped",
    "C-vormig": "C-shaped",
    "L-vormig": "L-shaped",
    "neutraal": "Neutral",
    "unknown": "Unknown",
}
_CATEGORY_LABELS["thumb_tip_index_proximity"].update({
    "contact": "contact",
    "near": "Near",
    "far": "Far",
})
_CATEGORY_LABELS["thumb_tip_middle_proximity"].update({
    "contact": "contact",
    "near": "Near",
    "far": "Far",
})


def decode_live_frame(frame):
    """Decode a bounded base64 JPEG frame sent by the browser component."""
    if not isinstance(frame, dict) or not isinstance(frame.get("data"), str):
        raise ValueError("The webcam has not sent a usable frame yet.")
    encoded = frame["data"]
    if not encoded or len(encoded) > MAX_FRAME_BYTES * 4 // 3 + 8:
        raise ValueError("The webcam frame is empty or exceeds the size limit.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError("The received webcam frame is invalid.") from error
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("The webcam frame is empty or exceeds the size limit.")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format != "JPEG":
                raise ValueError("The browser must send webcam frames as JPEG.")
            if max(source.size) > MAX_FRAME_DIMENSION:
                raise ValueError("The webcam frame resolution is too high.")
            return source.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("The received webcam frame cannot be read.") from error


def live_feature_rows(detection, label, references, parameters):
    """Compare one current detection to the chosen letter's expected features."""
    if label not in references:
        raise ValueError(f"No feature reference found for letter {label!r}.")
    sample = {
        **detection,
        "id": "live-webcam",
        "filename": "live-webcam.jpg",
        "label": label,
        "image_path": "",
    }
    comparison = compare_sample(sample, references, parameters)
    rows = []
    for feature, result in comparison["comparisons"].items():
        status = result["status"]
        actual = result["actual"]
        rows.append({
            "key": feature,
            "title": _FEATURE_TITLES.get(feature, feature),
            "expected": result["expected"],
            "actual": _CATEGORY_LABELS.get(feature, {}).get(actual, actual),
            "status": status,
            "color": (
                "green" if status == "match"
                else "muted" if status == "not_scored"
                else "red"
            ),
        })
    return rows
