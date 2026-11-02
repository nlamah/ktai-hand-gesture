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
_FEATURE_TITLES["thumb_mcp_shape"] = "Duim: MCP-vorm"
_CATEGORY_LABELS = {
    feature["key"]: {
        category["id"]: category["label"]
        for category in feature["categories"]
    }
    for feature in FEATURE_SCHEMA
}
_CATEGORY_LABELS["thumb_mcp_shape"] = {
    "B-vormig": "B-vormig",
    "not_B-vormig": "niet B-vormig",
    "C-vormig": "C-vormig",
    "L-vormig": "L-vormig",
    "neutraal": "neutraal",
    "unknown": "onbekend",
}
_CATEGORY_LABELS["thumb_tip_index_proximity"].update({
    "contact": "contact",
    "near": "nabij",
    "far": "ver weg",
})
_CATEGORY_LABELS["thumb_tip_middle_proximity"].update({
    "contact": "contact",
    "near": "nabij",
    "far": "ver weg",
})


def decode_live_frame(frame):
    """Decode a bounded base64 JPEG frame sent by the browser component."""
    if not isinstance(frame, dict) or not isinstance(frame.get("data"), str):
        raise ValueError("De webcam heeft nog geen bruikbaar beeld gestuurd.")
    encoded = frame["data"]
    if not encoded or len(encoded) > MAX_FRAME_BYTES * 4 // 3 + 8:
        raise ValueError("Het webcambeeld is leeg of groter dan toegestaan.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError("Het ontvangen webcambeeld is ongeldig.") from error
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Het webcambeeld is leeg of groter dan toegestaan.")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format != "JPEG":
                raise ValueError("De browser moet webcambeelden als JPEG versturen.")
            if max(source.size) > MAX_FRAME_DIMENSION:
                raise ValueError("Het webcambeeld heeft een te hoge resolutie.")
            return source.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("Het ontvangen webcambeeld kan niet worden gelezen.") from error


def live_feature_rows(detection, label, references, parameters):
    """Compare one current detection to the chosen letter's expected features."""
    if label not in references:
        raise ValueError(f"Geen feature-referentie gevonden voor letter {label!r}.")
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
