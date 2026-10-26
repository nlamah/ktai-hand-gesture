"""Streamlit controls for a user-started local camera subprocess."""
from pathlib import Path
import json
import subprocess
import sys
import uuid

import streamlit as st
from PIL import Image

from gesturelab.camera.runtime import new_session_name, reference_for, save_reference,load_reference
from gesturelab.datasets.data import atomic_json


def camera_running():
    job=st.session_state.get("camera_job")
    return bool(job and job["process"].poll() is None)


def start_camera(project,config):
    if camera_running():
        raise ValueError("Close the current camera window first.")
    project=Path(project).resolve()
    folder=project/".camera_runs"/uuid.uuid4().hex
    folder.mkdir(parents=True)
    config={**config,"project":str(project)}
    if config.get("model"):
        config["model"]=str(Path(config["model"]).resolve())
    atomic_json(folder/"config.json",config)
    command=[sys.executable,"-m","gesturelab.camera.runtime","--config",str(folder/"config.json"),
             "--status",str(folder/"status.json"),"--stop",str(folder/"stop")]
    with (folder/"output.log").open("wb") as log:
        proc=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=subprocess.STDOUT)
    st.session_state["camera_job"]={"process":proc,"folder":str(folder),"project":str(project),"mode":config["mode"]}


@st.fragment(run_every=1)
def camera_status():
    job=st.session_state.get("camera_job")
    if not job:
        return
    folder=Path(job["folder"])
    running=job["process"].poll() is None
    status={}
    if (folder/"status.json").exists():
        try:
            status=json.loads((folder/"status.json").read_text(encoding="utf-8"))
        except (OSError,ValueError):
            pass
    message=status.get("message","Starting camera window...")
    if status.get("state")=="error" or (not running and job["process"].returncode):
        st.error(message if status.get("state")=="error" else "Could not start the camera window. See the error details below.")
        with st.expander("Camera error details"):
            if (folder/"output.log").exists():
                st.code((folder/"output.log").read_text(encoding="utf-8",errors="replace")[-7000:])
    else:
        st.info(message+f" · saved: {status.get('saved',0)} · project: {Path(job['project']).name}")
    c1,c2=st.columns(2)
    if c1.button("Close camera window",disabled=not running,key="stop_camera_job"):
        (folder/"stop").touch()
    if c2.button("Refresh dataset and screen",key="refresh_camera_data"):
        st.rerun()


def common_camera_options(prefix):
    c1,c2=st.columns(2)
    index=c1.number_input("Camera",min_value=0,max_value=10,value=0,step=1,key=prefix+"_camera")
    mirror=c2.checkbox("Mirror the display",value=True,key=prefix+"_mirror")
    st.caption("The camera opens in a separate window on this computer. Grant Terminal/Python camera access if needed. "
               "Only the display is mirrored; stored images and model input remain unmirrored.")
    return int(index),mirror


def _capture_labels(project, samples, default_labels, allowed_labels):
    available = sorted({s['label'] for s in samples if s['included'] and s['role'] == 'development'})
    default = ', '.join(default_labels or available or ['fist', 'open_hand', 'peace', 'pointing', 'thumbs_up'])
    text = st.text_input('Gestures in order (comma-separated)', default,
                         key='capture_labels_' + str(project) + str(default_labels))
    labels = [label.strip() for label in text.split(',') if label.strip()]
    duplicates = len(labels) != len(set(labels))
    invalid = [label for label in labels if allowed_labels is not None and label not in allowed_labels]
    if invalid:
        st.error('Unknown letter: ' + ', '.join(invalid) + '. Use uppercase letters A–Z.')
    if duplicates:
        st.error('Each label may appear only once; set the photo count below.')
    return labels, bool(invalid) or duplicates


def _reference_picker(project, labels):
    with st.expander('View or replace references', expanded=True):
        st.caption('An existing development photo is used automatically. Check that it shows the gesture clearly. '
                   'You can upload your own reference; it is a guide and is not automatically used as training data. '
                   'Test photos are never selected automatically as references.')
        if labels:
            chosen = st.selectbox('Reference for', list(dict.fromkeys(labels)), key='guide_label')
            ref, source = reference_for(project, chosen)
            if ref:
                st.image(load_reference(project, chosen), width=280, caption=source)
            upload = st.file_uploader('Custom reference photo', type=['jpg', 'jpeg', 'png', 'webp'], key='guide_upload_' + chosen)
            if st.button('Save reference', disabled=upload is None or camera_running()):
                try:
                    save_reference(project, chosen, upload.getvalue())
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(str(error))
    missing = [label for label in labels if reference_for(project, label)[0] is None]
    if missing:
        st.warning('Choose a reference first for: ' + ', '.join(missing) + '. Upload a photo for each label above, or add and analyze development photos.')
    return missing


def _capture_options():
    c1, c2, c3 = st.columns(3)
    participant = c1.text_input('Participant', 'participant1', key='guided_person')
    condition = c2.text_input('Condition', 'baseline', key='guided_condition')
    role = c3.selectbox('Usage', ['development', 'test'], key='guided_role',
                        format_func=lambda value: 'Development' if value == 'development' else 'Separate final test')
    c1, c2, c3 = st.columns(3)
    count = c1.number_input('Photos per gesture', min_value=1, max_value=100, value=3, step=1)
    change = c2.slider('Countdown for next gesture (s)', 1.0, 20.0, 5.0, .5)
    interval = c3.slider('Time between photos (s)', .5, 15.0, 3.0, .5)
    index, mirror = common_camera_options('guided')
    st.info('Each start creates a new session. All photos from this round stay together. '
            'Reset your hand pose between photos. Change your position and camera angle between sessions.')
    st.caption('In the camera window: SPACE start/pause, + longer interval, - shorter interval, R restart countdown, Q close.')
    return {'photos_per_label': int(count), 'change_delay': float(change), 'interval': float(interval),
            'participant': participant.strip(), 'condition': condition.strip(), 'role': role,
            'camera_index': index, 'mirror': mirror}


def render_capture(project, samples, default_labels=None, allowed_labels=None):
    """Collect capture inputs; start hardware only after the explicit button press."""
    st.subheader('Guided capture with reference and countdown')
    st.write('The reference gesture appears on the left and your webcam on the right. After the countdown, the photo, label, '
             'landmarks and features are saved. The app then continues automatically. '
             'Without one usable hand, the countdown waits and then restarts.')
    labels, invalid = _capture_labels(project, samples, default_labels, allowed_labels)
    missing = _reference_picker(project, labels)
    options = _capture_options()
    disabled = camera_running() or not labels or bool(missing) or invalid or not options['participant']
    if st.button('Open webcam and start new session', type='primary', disabled=disabled):
        try:
            start_camera(project, {'mode': 'capture', 'labels': labels, 'session': new_session_name(), **options})
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
