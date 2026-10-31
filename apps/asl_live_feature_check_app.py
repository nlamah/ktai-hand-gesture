"""Compare webcam hand features with the typed ASL letter in real time."""
import argparse
import json
from pathlib import Path
import threading

import streamlit as st

from gesturelab.datasets.data import Detector
from gesturelab.features.asl_excel_review import load_reference_workbook
from gesturelab.features.asl_live import decode_live_frame, live_feature_rows
from gesturelab.features.asl_schema import PARAMETER_DEFAULTS
from gesturelab.features.categorical_hand import validate_parameters


_ROOT = Path(__file__).resolve().parents[1]
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
        Zet de camera aan om te beginnen.
      </div>
    </div>
    <div id="camera-status" class="camera-status">Camera staat uit.</div>
  </section>
  <section class="features-card">
    <div class="feature-heading">
      <div>
        <div class="eyebrow">Verwachte handvorm</div>
        <h2 id="target-letter">Letter —</h2>
      </div>
      <div id="match-count" class="match-count">Wacht op beeld</div>
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
      setStateValue("camera_error", "Deze browser ondersteunt geen webcamtoegang.")
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
          ? "Geen webcamtoestemming. Sta cameratoegang toe in de browserinstellingen."
          : `Webcam starten is mislukt: ${error?.message || "onbekende fout"}`,
      )
    } finally {
      runtime.starting = false
    }
  }

  cameraStatus.textContent = data?.message || (runtime.active ? "Webcam starten…" : "Camera staat uit.")
  cameraStatus.dataset.tone = data?.tone || "neutral"
  targetLetter.textContent = data?.label ? `Letter ${data.label}` : "Letter —"
  matchCount.textContent = data?.summary || "Wacht op beeld"
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
    values.textContent = `Verwacht: ${item.expected} · Gezien: ${item.actual}`
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
    default=str(_ROOT / "config" / "asl_landmark_thresholds_tuned.json"),
)
args, _ = parser.parse_known_args()

st.set_page_config(
    page_title="ASL live features",
    page_icon=":material/back_hand:",
    layout="wide",
)
st.title("ASL-features live controleren")
st.write(
    "Typ de letter die je gaat maken. De webcam toont je hand en vergelijkt "
    "de herkende features live met de referentie voor die letter."
)
st.caption(
    "Groen = feature komt overeen; rood = afwijkend of niet herkend; grijs = "
    "niet gescoord. Dit controleert features, niet automatisch de letter."
)

st.caption(
    "Spreiding: ruimte tussen vingers die naast elkaar liggen; uitwaaieren mag. "
    "Tegen elkaar liggende, gekruiste of sterk gevouwen vingers tellen niet mee."
)

reference_path = Path(args.reference).expanduser()
uploaded_reference = st.file_uploader("Excel-featuretabel (.xlsx)", type=["xlsx"])
if uploaded_reference is not None:
    reference_contents = uploaded_reference.getvalue()
elif reference_path.is_file():
    reference_contents = reference_path.read_bytes()
    st.caption(f"Gebruikte referenties: {reference_path}")
else:
    st.info("Upload je Excel-featuretabel om de live controle te starten.")
    st.stop()

try:
    references = load_reference_workbook(reference_contents)
    parameters = _load_parameters(Path(args.thresholds).expanduser())
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"Instellingen konden niet worden geladen: {error}")
    st.stop()

st.caption(
    f"Vingerbuiging: gebogen ≤ {parameters['finger_flexion']['bent_max_deg']:g}°; "
    f"gestrekt ≥ {parameters['finger_flexion']['extended_min_deg']:g}°; "
    "daartussen licht gebogen. Slashwaarden accepteren de genoemde categorieën."
)

input_col, camera_col = st.columns([1, 1], vertical_alignment="bottom")
with input_col:
    typed_label = st.text_input(
        "Letter die je gaat maken",
        max_chars=1,
        placeholder="Bijvoorbeeld A",
        key="asl_live_target_letter",
    )
label = typed_label.strip().upper()
valid_label = len(label) == 1 and label in references
with camera_col:
    camera_active = st.toggle(
        "Webcam aan",
        value=False,
        disabled=not valid_label,
        key="asl_live_camera_active",
    )

if not valid_label:
    st.warning("Vul één letter A–Z in voordat je de webcam start.")

component_state = st.session_state.get(_COMPONENT_KEY, {})
frame = component_state.get("frame") if isinstance(component_state, dict) else None
frame_id = frame.get("id") if isinstance(frame, dict) else None
detector_error = None

if not camera_active:
    st.session_state.pop(_DETECTION_KEY, None)
    st.session_state.pop(_DETECTOR_ERROR_KEY, None)
    st.session_state.pop(_PROCESSED_FRAME_KEY, None)
elif valid_label and frame_id and frame_id != st.session_state.get(_PROCESSED_FRAME_KEY):
    try:
        image = decode_live_frame(frame)
        detection = _shared_detector().analyze(image)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        detector_error = str(error)
        st.session_state[_DETECTOR_ERROR_KEY] = detector_error
    else:
        st.session_state[_DETECTION_KEY] = detection
        st.session_state.pop(_DETECTOR_ERROR_KEY, None)
    st.session_state[_PROCESSED_FRAME_KEY] = frame_id

detector_error = st.session_state.get(_DETECTOR_ERROR_KEY)
detection = st.session_state.get(_DETECTION_KEY)
features = (
    live_feature_rows(detection, label, references, parameters)
    if camera_active and valid_label and detection is not None
    else []
)

if detector_error:
    message = f"Featurecontrole gestopt: {detector_error}"
    tone = "error"
    summary = "Controle gestopt"
elif not camera_active:
    message = "Vul een letter in en zet daarna de webcam aan."
    tone = "neutral"
    summary = "Camera staat uit"
elif not valid_label:
    message = "Vul één letter A–Z in om features te vergelijken."
    tone = "error"
    summary = "Geen geldige letter"
elif detection is None:
    message = "Wacht op de eerste webcambeelden…"
    tone = "neutral"
    summary = "Wacht op beeld"
elif detection.get("status") != "ok":
    if detection.get("status") == "no_hand":
        message = "Geen hand gevonden. Houd één hand in beeld."
    else:
        message = "Er moet precies één hand zichtbaar zijn."
    tone = "error"
    summary = "Hand niet herkend"
else:
    matches = sum(row["status"] == "match" for row in features)
    scored = sum(row["status"] != "not_scored" for row in features)
    all_match = matches == scored and scored > 0
    message = (
        f"Alle gescoorde features van {label} komen overeen."
        if all_match
        else f"{matches} van {scored} gescoorde features van {label} komen overeen."
    )
    tone = "success" if all_match else "error"
    summary = f"{matches}/{scored} match"

_LIVE_COMPONENT(
    key=_COMPONENT_KEY,
    data={
        "active": bool(camera_active and valid_label and not detector_error),
        "label": label if valid_label else "",
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
    "De duim-MCP-vormen en de Excel-waardes zijn voorlopige referenties; "
    "controleer afwijkingen ook visueel. Browsercameratoegang is nodig."
)
