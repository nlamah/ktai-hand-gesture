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
NAMES={'thumb_mcp_shape':'Duimvorm (MCP)','flexion_index_pip':'Wijsvinger (PIP)',
       'flexion_middle_pip':'Middelvinger (PIP)','flexion_ring_pip':'Ringvinger (PIP)',
       'flexion_pinky_pip':'Pink (PIP)','spread_index_middle':'Spreiding wijs/middel',
       'crossing_index_middle':'Kruising wijs/middel','thumb_tip_index_proximity':'Duimtop bij wijsvinger'}
WORDS={'Partly bent':'Deels gebogen','Extended':'Gestrekt','Bent':'Gebogen',
       'Not B-shaped':'Niet B-vormig','B-shaped':'B-vormig','Not spread':'Niet gespreid',
       'Spread':'Gespreid','Not crossed':'Niet gekruist','Crossed':'Gekruist',
       'Near':'Nabij','Far':'Veraf','No':'Nee','Yes':'Ja','contact':'Contact',
       'Unknown':'Onbekend'}
STATES={'match':'Komt overeen','mismatch':'Afwijking','unknown':'Onbekend','not_scored':'Niet beoordeeld'}


def dutch(value):
    return ' / '.join(WORDS.get(part.strip(),part.strip()) for part in str(value).split('/'))


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
        st.title('Vingerkenmerken controleren')
        st.caption('Opgeslagen beeld: dezelfde featurecontrole als in de live tool. Er wordt geen camera gestart of model getraind.')
        target=st.text_input('Vergelijk met letter',value=record['expected_letter'],max_chars=1).strip().upper()
        data=review_data(capture_path,reference_path,threshold_path,model_path,target)
    except (OSError,ValueError,KeyError) as error:
        st.error(f'Voorbeeld niet beschikbaar: {error}');return
    left,right=st.columns([1,1.7],gap='large')
    with left:
        st.image(data['crop'],width='stretch',alt='Uitsnede van de opgeslagen hand met de oorspronkelijke handpunten')
        st.caption(f"Opgegeven letter: {record['expected_letter']} · modelvoorspelling: {data['prediction']['letter']}")
        st.caption('Uitsnede met handpunten; de voorspelling gebruikt de opgeslagen geometrie van het volledige beeld.')
    with right:
        st.subheader(f'Controle tegen {target}')
        selected=[row for row in data['rows'] if row['key'] in KEYS]
        table=[{'Kenmerk':NAMES[row['key']],'Verwacht':dutch(row['expected']),
                'Gemeten':dutch(row['actual']),'Controle':STATES[row['status']]} for row in selected]
        st.table(table,hide_index=True,alt='Verwachte en gemeten kernkenmerken met overeenkomst of afwijking')
        p=data['parameters']['finger_flexion']
        st.caption(f"Vingerbuiging: gebogen ≤ {p['bent_max_deg']:g}°, gestrekt ≥ {p['extended_min_deg']:g}°; daartussen deels gebogen.")
        st.caption('PIP is het middelste vingergewricht. Een overeenkomst toetst de gekozen referentie, niet de anatomische juistheid van de handpunten.')
        angles=data['measurements']
        st.write(f"Wijsvinger-PIP: {angles['index_pip_angle_deg']:.1f}° · Middelvinger-PIP: {angles['middle_pip_angle_deg']:.1f}°")
    with st.expander('Alle kenmerken'):
        st.table([{'Kenmerk':r['title'],'Verwacht':dutch(r['expected']),'Gemeten':dutch(r['actual']),
                   'Controle':STATES[r['status']]} for r in data['rows']],hide_index=True,alt='Alle referentiekenmerken van de opgeslagen hand')
