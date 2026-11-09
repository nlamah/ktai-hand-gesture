"""Read-only replay of the live feature comparison on an existing capture."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from research.geometry import measure_hand_features, validate_parameters
from research.live import live_feature_rows
from research.live_model import load_bundle, predict_letter
from research.workbook import load_reference_workbook

CONNECTIONS=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
             (15,16),(13,17),(0,17),(17,18),(18,19),(19,20))
KEYS=('thumb_mcp_shape','flexion_index_pip','flexion_middle_pip','flexion_ring_pip',
      'flexion_pinky_pip','spread_index_middle','crossing_index_middle','thumb_tip_index_proximity')
NAMES={'thumb_mcp_shape':'Thumb shape (MCP)','flexion_index_pip':'Index finger (PIP)',
       'flexion_middle_pip':'Middle finger (PIP)','flexion_ring_pip':'Ring finger (PIP)',
       'flexion_pinky_pip':'Little finger (PIP)','spread_index_middle':'Index/middle spreading',
       'crossing_index_middle':'Index/middle crossing','thumb_tip_index_proximity':'Thumb tip near index'}
STATES={'match':'Match','mismatch':'Mismatch','unknown':'Unknown','not_scored':'Not scored'}


def english(value):
    return ' / '.join(part.strip() for part in str(value).split('/'))


def hand_crop(image,landmarks,overlay=True):
    """Crop to the measured hand; drawing does not alter model input or landmarks."""
    im=image.convert('RGB').copy();w,h=im.size
    points=np.asarray(landmarks,dtype=float)
    if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('Invalid landmark geometry.')
    xy=points[:,:2]*[w,h]
    lo=xy.min(axis=0);hi=xy.max(axis=0);margin=max(hi-lo)*.16+8
    box=(max(0,int(lo[0]-margin)),max(0,int(lo[1]-margin)),
         min(w,int(np.ceil(hi[0]+margin))),min(h,int(np.ceil(hi[1]+margin))))
    if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Hand lies outside the image.')
    if overlay:
        draw=ImageDraw.Draw(im);width=max(2,round(min(w,h)/200))
        for a,b in CONNECTIONS:draw.line([tuple(xy[a]),tuple(xy[b])],fill='#00d8ae',width=width)
        for j,(x,y) in enumerate(xy):
            r=width+1;draw.ellipse((x-r,y-r,x+r,y+r),fill='#ffcc33')
    return im.crop(box),box


def review_data(capture_path,reference_path,threshold_path,model_path,label=None):
    folder=Path(capture_path);path=folder if folder.is_file() else folder/'capture.json'
    record=json.loads(path.read_text());folder=path.parent
    image_path=folder/'model_input.png'
    if sha256(image_path.read_bytes()).hexdigest()!=record['files']['model_input.png']['sha256']:
        raise ValueError('Capture image checksum differs from its record.')
    detection=record['detection']
    if detection['status']!='ok':raise ValueError('The saved capture has no accepted hand.')
    parameters=validate_parameters(json.loads(Path(threshold_path).read_text()))
    references=load_reference_workbook(Path(reference_path).read_bytes())
    target=label or record['expected_letter']
    if target not in references:raise ValueError('Choose one letter A-Z.')
    rows=live_feature_rows(detection,target,references,parameters)
    measurements=measure_hand_features(detection['world_landmarks'],detection.get('handedness'))
    with Image.open(image_path) as image:crop,box=hand_crop(image,detection['image_landmarks'])
    prediction=predict_letter(load_bundle(Path(model_path)),detection)
    return {'record':record,'target':target,'rows':rows,'measurements':measurements,'crop':crop,'box':box,
            'prediction':prediction,'parameters':parameters,'model_sha256':sha256(Path(model_path).read_bytes()).hexdigest()}


def show_review(capture_path,reference_path,threshold_path,model_path):
    import streamlit as st
    path=Path(capture_path);path=path if path.is_file() else path/'capture.json'
    try:
        record=json.loads(path.read_text())
        st.title('Inspect finger features')
        st.caption('Saved capture: the same feature comparison as the live tool. No camera or model training is started.')
        target=st.text_input('Compare with letter',value=record['expected_letter'],max_chars=1).strip().upper()
        data=review_data(capture_path,reference_path,threshold_path,model_path,target)
    except (OSError,ValueError,KeyError) as error:
        st.error(f'Example unavailable: {error}');return
    left,right=st.columns([1,1.7],gap='large')
    with left:
        st.image(data['crop'],width='stretch',alt='Saved hand crop with original landmarks')
        st.caption(f"Intended letter: {record['expected_letter']} · model prediction: {data['prediction']['letter']}")
        st.caption('Landmark overlay for display; prediction uses the saved full-frame geometry.')
    with right:
        st.subheader(f'Compare against {target}')
        selected=[row for row in data['rows'] if row['key'] in KEYS]
        table=[{'Feature':NAMES[row['key']],'Expected':english(row['expected']),
                'Observed':english(row['actual']),'Check':STATES[row['status']]} for row in selected]
        st.table(table,hide_index=True,alt='Expected and observed core features with match/mismatch')
        p=data['parameters']['finger_flexion']
        st.caption(f"Finger flexion: bent ≤ {p['bent_max_deg']:g}°, extended ≥ {p['extended_min_deg']:g}°; partly bent between the thresholds.")
        st.caption('PIP is the middle finger joint. A match checks the selected reference, not anatomical landmark accuracy.')
        angles=data['measurements']
        st.write(f"Index PIP: {angles['index_pip_angle_deg']:.1f}° · Middle PIP: {angles['middle_pip_angle_deg']:.1f}°")
    with st.expander('All features'):
        st.table([{'Feature':r['title'],'Expected':english(r['expected']),'Observed':english(r['actual']),
                   'Check':STATES[r['status']]} for r in data['rows']],hide_index=True,alt='All reference features for the saved hand')
