"""Recognize webcam ASL letters and compare optional target-letter features."""
import argparse
import json
from pathlib import Path
import threading

import streamlit as st

from research.data import Detector
from research.workbook import load_reference_workbook
from research.live import decode_live_frame, live_feature_rows
from research.live_model import DEFAULT_MODEL, load_bundle, predict_letter
from research.schema import PARAMETER_DEFAULTS
from research.geometry import validate_parameters


_ROOT = Path(__file__).resolve().parent
_COMPONENT_KEY = "asl_live_feature_camera"
_PROCESSED_FRAME_KEY = "asl_live_feature_processed_frame"
_DETECTION_KEY = "asl_live_feature_last_detection"
_DETECTOR_ERROR_KEY = "asl_live_feature_detector_error"

_COMPONENT_HTML = """
<main class="camera-app">
  <section class="camera-card">
    <div class="camera-stage">
      <video id="camera-video" autoplay playsinline muted></video>
      <div id="camera-placeholder" class="placeholder">
        Turn on the camera to begin.
      </div>
    </div>
    <div id="camera-status" class="camera-status">Camera is off.</div>
  </section>
  <section class="features-card">
    <div class="feature-heading">
      <div>
        <div class="eyebrow">Expected hand shape</div>
        <h2 id="target-letter">Letter —</h2>
      </div>
      <div id="match-count" class="match-count">Waiting for frame</div>
    </div>
    <div id="feature-list" class="feature-list"></div>
  </section>
</main>
"""

_COMPONENT_CSS = """
:host {
  display: block;
  color: var(--st-text-color, #202124);
  font-family: var(--st-font, sans-serif);
}
.camera-app { display: grid; gap: 1rem; }
.camera-card, .features-card {
  border: 1px solid var(--st-border-color, #d7dbe0);
  border-radius: 0.75rem;
  background: var(--st-background-color, #fff);
  overflow: hidden;
}
.camera-stage {
  position: relative;
  min-height: 16rem;
  background: #10151c;
  display: grid;
  place-items: center;
}
#camera-video {
  width: 100%;
  max-height: 32rem;
  object-fit: contain;
  transform: scaleX(-1);
}
.placeholder {
  position: absolute;
  color: #f5f7fa;
  text-align: center;
  padding: 1rem;
}
.camera-status { padding: 0.7rem 1rem; font-size: 0.92rem; }
.camera-status[data-tone="error"] { color: #b42318; }
.camera-status[data-tone="success"] { color: #157347; }
.features-card { padding: 1rem; }
.feature-heading {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.8rem;
}
.eyebrow { color: var(--st-secondary-text-color, #667085); font-size: 0.78rem; }
.feature-heading h2 { margin: 0.15rem 0 0; font-size: 1.35rem; }
.match-count { font-weight: 650; }
.feature-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.5rem; }
.feature {
  border: 1px solid var(--st-border-color, #d7dbe0);
  border-left: 0.35rem solid #98a2b3;
  border-radius: 0.45rem;
  padding: 0.6rem 0.7rem;
  min-width: 0;
}
.feature[data-status="match"] { border-left-color: #198754; background: color-mix(in srgb, #198754 8%, transparent); }
.feature[data-status="mismatch"], .feature[data-status="unknown"] { border-left-color: #dc3545; background: color-mix(in srgb, #dc3545 8%, transparent); }
.feature[data-status="not_scored"] { border-left-color: #98a2b3; opacity: 0.75; }
.feature-name { font-weight: 600; margin-bottom: 0.2rem; }
.feature-values { font-size: 0.84rem; line-height: 1.4; overflow-wrap: anywhere; }
"""

_COMPONENT_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const video = parentElement.querySelector("#camera-video")
  const placeholder = parentElement.querySelector("#camera-placeholder")
  const cameraStatus = parentElement.querySelector("#camera-status")
  const targetLetter = parentElement.querySelector("#target-letter")
  const matchCount = parentElement.querySelector("#match-count")
  const featureList = parentElement.querySelector("#feature-list")
  if (!video || !placeholder || !cameraStatus || !targetLetter || !matchCount || !featureList) return

  const runtime = parentElement._aslLiveRuntime || {
    stream: null,
    starting: false,
    failed: false,
    timer: null,
    lastSent: 0,
    frameNumber: 0,
    canvas: document.createElement("canvas"),
  }
  parentElement._aslLiveRuntime = runtime
  runtime.sendState = setStateValue
  runtime.active = Boolean(data?.active)

  const stopCamera = () => {
    if (runtime.timer !== null) {
      clearInterval(runtime.timer)
      runtime.timer = null
    }
    if (runtime.stream) {
      runtime.stream.getTracks().forEach(track => track.stop())
      runtime.stream = null
    }
    video.srcObject = null
    runtime.starting = false
    runtime.failed = false
    placeholder.hidden = false
  }

  const startCamera = async () => {
    if (runtime.starting || runtime.stream || runtime.failed) return
    if (!navigator.mediaDevices?.getUserMedia) {
      runtime.failed = true
      setStateValue("camera_error", "This browser does not support webcam access.")
      return
    }
    runtime.starting = true
    try {
      runtime.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      if (!runtime.active) {
        stopCamera()
        return
      }
      video.srcObject = runtime.stream
      await video.play()
      placeholder.hidden = true
      setStateValue("camera_error", "")
      if (runtime.timer === null) {
        runtime.timer = setInterval(() => {
          if (!runtime.active || video.readyState < 2 || !video.videoWidth) return
          const now = performance.now()
          if (now - runtime.lastSent < 450) return
          runtime.lastSent = now
          const width = Math.min(480, video.videoWidth)
          const height = Math.round(width * video.videoHeight / video.videoWidth)
          runtime.canvas.width = width
          runtime.canvas.height = height
          runtime.canvas.getContext("2d").drawImage(video, 0, 0, width, height)
          const encoded = runtime.canvas.toDataURL("image/jpeg", 0.65).split(",", 2)[1]
          runtime.frameNumber += 1
          runtime.sendState("frame", {
            id: `${Date.now()}-${runtime.frameNumber}`,
            data: encoded,
          })
        }, 100)
      }
    } catch (error) {
      stopCamera()
      runtime.failed = true
      setStateValue(
        "camera_error",
        error?.name === "NotAllowedError"
          ? "Camera permission denied. Allow camera access in browser settings."
          : `Failed to start webcam: ${error?.message || "unknown error"}`,
      )
    } finally {
      runtime.starting = false
    }
  }

  cameraStatus.textContent = data?.message || (runtime.active ? "Starting webcam…" : "Camera is off.")
  cameraStatus.dataset.tone = data?.tone || "neutral"
  targetLetter.textContent = data?.label ? `Letter ${data.label}` : "Letter —"
  matchCount.textContent = data?.summary || "Waiting for frame"
  placeholder.hidden = Boolean(runtime.active && runtime.stream)
  featureList.replaceChildren()

  for (const item of (data?.features || [])) {
    const row = document.createElement("article")
    row.className = "feature"
    row.dataset.status = item.status
    const name = document.createElement("div")
    name.className = "feature-name"
    name.textContent = item.title
    const values = document.createElement("div")
    values.className = "feature-values"
    values.textContent = `Expected: ${item.expected} · Observed: ${item.actual}`
    row.append(name, values)
    featureList.append(row)
  }

  if (!runtime.active) {
    stopCamera()
  } else {
    startCamera()
  }
}
"""


_LIVE_COMPONENT = st.components.v2.component(
    "asl_live_feature_check",
    html=_COMPONENT_HTML,
    css=_COMPONENT_CSS,
    js=_COMPONENT_JS,
)


class _SharedDetector:
    def __init__(self):
        self.detector = Detector()
        self.lock = threading.Lock()

    def analyze(self, image):
        with self.lock:
            return self.detector.analyze(image)


@st.cache_resource(max_entries=1)
def _shared_detector():
    return _SharedDetector()


@st.cache_resource(max_entries=1)
def _prediction_model(path, modified_ns):
    return load_bundle(path)


def _load_parameters(path):
    parameters = json.loads(path.read_text(encoding="utf-8"))
    return validate_parameters(parameters)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--reference",
    default=str(_ROOT / "assets" / "asl_feature_references.xlsx"),
)
parser.add_argument(
    "--thresholds",
    default=str(_ROOT / "config" / "thresholds.json"),
)
parser.add_argument("--model", default=str(DEFAULT_MODEL))
args, _ = parser.parse_known_args()

st.set_page_config(
    page_title="ASL live letter recognition",
    page_icon=":material/back_hand:",
    layout="wide",
)
st.title("Check ASL features and recognize letters live")
st.write(
    "Turn on the webcam to recognize letters. Optionally enter the letter you "
    "intend to sign to compare the features with your Excel reference."
)
st.caption(
    "Green = matching feature; red = mismatch or not detected; gray = "
    "not scored. Model prediction uses Shin features and joint angles; "
    "the entered letter does not affect that prediction."
)

st.caption(
    "Spread: space between adjacent fingers; fanning out is allowed. "
    "Touching, crossed or strongly folded fingers do not count."
)

reference_path = Path(args.reference).expanduser()
uploaded_reference = st.file_uploader("Excel feature table (.xlsx)", type=["xlsx"])
if uploaded_reference is not None:
    reference_contents = uploaded_reference.getvalue()
elif reference_path.is_file():
    reference_contents = reference_path.read_bytes()
    st.caption(f"Reference file: {reference_path}")
else:
    st.info("Upload your Excel feature table to start the live check.")
    st.stop()

try:
    references = load_reference_workbook(reference_contents)
    parameters = _load_parameters(Path(args.thresholds).expanduser())
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Could not load settings: {error}")
    st.stop()

st.caption(
    f"Finger flexion: bent ≤ {parameters['finger_flexion']['bent_max_deg']:g}°; "
    f"extended ≥ {parameters['finger_flexion']['extended_min_deg']:g}°; "
    "partly bent in between. Slash values accept each listed category."
)

recognition_enabled = st.toggle("Recognize letters using trained model", value=True)
model_path = Path(args.model).expanduser()
model_available = model_path.is_file()
if recognition_enabled and not model_available:
    st.info("Live model missing. Train it once with: python -m research.live_model")
st.caption("Recognition is experimental. For a valid hand the model always chooses a letter; alternatives are not confidence percentages.")

input_col, camera_col = st.columns([1, 1], vertical_alignment="bottom")
with input_col:
    typed_label = st.text_input(
        "Intended letter",
        max_chars=1,
        placeholder="For example A",
        key="asl_live_target_letter",
    )
label = typed_label.strip().upper()
valid_label = len(label) == 1 and label in references
with camera_col:
    camera_active = st.toggle(
        "Webcam on",
        value=False,
        disabled=not (valid_label or (recognition_enabled and model_available)),
        key="asl_live_camera_active",
    )

camera_active = bool(camera_active and (valid_label or (recognition_enabled and model_available)))

if not valid_label and not (recognition_enabled and model_available):
    st.warning("Enter one letter A-Z for feature checking, or load a model for recognition.")

component_state = st.session_state.get(_COMPONENT_KEY, {})
frame = component_state.get("frame") if isinstance(component_state, dict) else None
frame_id = frame.get("id") if isinstance(frame, dict) else None
detector_error = None

if not camera_active:
    st.session_state.pop(_DETECTION_KEY, None)
    st.session_state.pop(_DETECTOR_ERROR_KEY, None)
    st.session_state[_PROCESSED_FRAME_KEY] = frame_id
elif frame_id and frame_id != st.session_state.get(_PROCESSED_FRAME_KEY):
    try:
        image = decode_live_frame(frame)
        detection = _shared_detector().analyze(image)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        st.session_state.pop(_DETECTION_KEY, None)
        detector_error = str(error)
        st.session_state[_DETECTOR_ERROR_KEY] = detector_error
    else:
        st.session_state[_DETECTION_KEY] = detection
        st.session_state.pop(_DETECTOR_ERROR_KEY, None)
    st.session_state[_PROCESSED_FRAME_KEY] = frame_id

detector_error = st.session_state.get(_DETECTOR_ERROR_KEY)
detection = st.session_state.get(_DETECTION_KEY)
prediction = None
prediction_error = None
if camera_active and recognition_enabled and model_available and detection is not None and detection.get("status") == "ok" and not detector_error:
    try:
        prediction = predict_letter(_prediction_model(str(model_path.resolve()), model_path.stat().st_mtime_ns), detection)
    except (OSError, ValueError, RuntimeError, ImportError, EOFError, TypeError, KeyError) as error:
        prediction_error = str(error)
if recognition_enabled:
    st.metric("Recognized letter", prediction["letter"] if prediction else "—")
    if prediction:
        st.caption("Model alternatives: " + ", ".join(prediction["alternatives"]))
    if prediction_error:
        st.error(f"Letter recognition stopped: {prediction_error}")
comparison_label = label if valid_label else prediction["letter"] if prediction else ""
features = (
    live_feature_rows(detection, comparison_label, references, parameters)
    if camera_active and comparison_label and detection is not None and not detector_error
    else []
)
if prediction and not valid_label:
    st.caption("The feature table below compares against the predicted letter. Enter an intended letter for an independent comparison.")

if detector_error:
    message = f"Feature check stopped: {detector_error}"
    tone = "error"
    summary = "Check stopped"
elif not camera_active:
    message = "Turn on the webcam to recognize letters or check features."
    tone = "neutral"
    summary = "Camera is off"
elif detection is None:
    message = "Waiting for the first webcam frames…"
    tone = "neutral"
    summary = "Waiting for frame"
elif detection.get("status") != "ok":
    if detection.get("status") == "no_hand":
        message = "No hand found. Keep one hand in frame."
    else:
        message = "Exactly one hand must be visible."
    tone = "error"
    summary = "Hand not detected"
elif prediction_error:
    message = f"Letter recognition stopped: {prediction_error}"
    tone = "error"
    summary = "No prediction"
else:
    matches = sum(row["status"] == "match" for row in features)
    scored = sum(row["status"] != "not_scored" for row in features)
    all_match = matches == scored and scored > 0
    message = (
        f"All scored features of {comparison_label} match."
        if all_match
        else f"{matches} of {scored} scored features of {comparison_label} match."
    )
    if prediction:
        message = f"Model recognizes {prediction['letter']}. " + message
    tone = "success" if all_match else "error"
    summary = f"{matches}/{scored} match"

_LIVE_COMPONENT(
    key=_COMPONENT_KEY,
    data={
        "active": bool(camera_active and not detector_error),
        "label": comparison_label,
        "message": message,
        "tone": tone,
        "summary": summary,
        "features": features,
    },
    on_frame_change=lambda: None,
    on_camera_error_change=lambda: None,
)

camera_error = component_state.get("camera_error") if isinstance(component_state, dict) else None
if camera_error:
    st.error(camera_error)

st.caption(
    "Thumb MCP shapes and Excel values are provisional references; "
    "also inspect mismatches visually. Browser camera access is required."
)
