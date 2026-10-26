"""Inspect finger feature code without training a letter classifier."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inspect
import pandas as pd
from PIL import Image,ImageDraw
import streamlit as st
from gesturelab.datasets.data import ensure_project,read_manifest,overlay,atomic_json
from gesturelab.features.inspection import features_for_sample,finger_table
from gesturelab.features import finger_features
from gesturelab.features.finger_features import FINGERS,FINGER_NAMES,validate_thresholds
from gesturelab.features.audit import load_thresholds,read_annotations,save_annotation,evaluate_annotations,STATES

p=argparse.ArgumentParser(add_help=False);p.add_argument('--project',default='data/ngt');a,_=p.parse_known_args()
st.set_page_config(page_title='NGT · Check finger features',page_icon='🔎',layout='wide')
st.title('2 · Check finger features')
st.session_state.setdefault('audit_project',str(Path(a.project).resolve()))
project=ensure_project(Path(st.sidebar.text_input('Project directory',key='audit_project')).expanduser())
saved=load_thresholds(project)
st.sidebar.subheader('Thresholds for this check')
config={}
for key,label in [('extended_angle','Extended: fingers (°)'),('thumb_extended_angle','Extended: thumb (°)'),('bent_angle','Bent: angle below (°)')]:
    config[key]=st.sidebar.number_input(label,0.0,180.0,float(saved[key]),1.0,key=str(project)+key)
for key,label in [('extended_straightness','Extended: minimum straightness'),('bent_straightness','Bent: straightness below')]:
    config[key]=st.sidebar.number_input(label,0.0,1.0,float(saved[key]),.01,key=str(project)+key)
try:
    validate_thresholds(config)
except ValueError as error:
    st.error(str(error));st.stop()
if st.sidebar.button('Save thresholds'):
    atomic_json(project/'finger_thresholds.json',config)
    st.sidebar.success('Saved. New experiments will use these settings.')
st.sidebar.caption('Existing experiments retain their frozen settings. Previously cached features in the manifest are not overwritten.')
page=st.radio('Section',['Visual inspection','Precision and recall','Code and definitions'],horizontal=True)
samples=[s for s in read_manifest(project)['samples'] if s['included'] and s['role']=='development']
if page=='Visual inspection':
    st.write('Assess the photo yourself first, then reveal the computed measurements. The letter label does not determine which fingers are extended.')
    if not samples:
        st.info('First collect development photos with collect_app.py. Reserved test photos are not used here.');st.stop()
    labels=st.multiselect('Letters',sorted({s['label'] for s in samples}),default=sorted({s['label'] for s in samples}))
    choices=[s for s in samples if s['label'] in labels]
    if not choices:
        st.stop()
    sample=st.selectbox('Photo',choices,format_func=lambda s:f"{s['label']} · {s['session']} · {s['id'][:8]}")
    st.image(str(project/sample['image_path']),width=450,caption='Assess the visible pose; choose unscorable for occluded fingers.')
    existing=read_annotations(project).get(sample['id'],{})
    states={}
    cols=st.columns(5)
    for col,finger in zip(cols,FINGERS):
        options=['']+list(STATES)
        default=existing.get('states',{}).get(finger,'')
        states[finger]=col.selectbox(FINGER_NAMES[finger],options,index=options.index(default),
            format_func=lambda x:STATES.get(x,'Choose...'),key=sample['id']+finger)
    note=st.text_input('Note',existing.get('note',''),key='note_'+sample['id'])
    if st.button('Save my assessment',disabled=not all(states.values())):
        save_annotation(project,sample['id'],states,note);st.success('Manual assessment saved.')
    with st.expander('Show landmarks, computed states and measurements'):
        if sample['status']!='ok':
            st.warning('No usable hand detection: '+sample['status'])
        else:
            image=Image.open(project/sample['image_path']).convert('RGB')
            marked=overlay(image,sample);draw=ImageDraw.Draw(marked)
            for i,point in enumerate(sample['image_landmarks']):
                draw.text((point[0]*image.width+4,point[1]*image.height),str(i),fill='red',stroke_width=1,stroke_fill='white')
            c1,c2=st.columns(2);c1.image(marked)
            features=features_for_sample(sample,config)
            c2.dataframe(pd.DataFrame(finger_table(features)),hide_index=True)
            st.dataframe(pd.DataFrame([{'feature':k,'value':v} for k,v in features.items() if k.startswith('geom_')]),hide_index=True)
elif page=='Precision and recall':
    table,records,meta=evaluate_annotations(project,config)
    st.write('Extension and bending are each evaluated as a binary task for every finger. An intermediate state is negative for both tasks.')
    st.latex(r'\mathrm{precision}=\frac{TP}{TP+FP}\qquad \mathrm{recall}=\frac{TP}{TP+FN}')
    st.caption('Precision: what fraction of predicted extensions are correct? Recall: what fraction of actual extensions were detected?')
    if records.empty:
        st.info('First save your own assessments in Visual inspection.')
    else:
        st.dataframe(table,hide_index=True)
        graph=table.copy();graph['finger / target']=graph.finger.map(FINGER_NAMES)+' / '+graph.target.map(STATES)
        st.bar_chart(graph.set_index('finger / target')[['precision','recall']])
        st.json(meta)
        st.dataframe(records[records.truth!=records.prediction],hide_index=True)
        st.download_button('Download precision/recall',table.to_csv(index=False).encode(),'finger_precision_recall.csv','text/csv')
    st.info('This is a development diagnostic. A blank score means the denominator is zero. Unscorable assessments are excluded and counted. Missing hand detection counts as a false negative when the human assessment is positive.')
else:
    st.write('The code below is also used by capture, feature inspection and training scripts. There is no separate test implementation.')
    st.code(Path(finger_features.__file__).read_text(),language='python')
    st.code('python -m unittest discover -s tests -p test_finger_module.py -v',language='bash')
    st.markdown('**Extended:** both interior angles and straightness meet their thresholds. **Bent:** at least one angle or straightness is below its bending threshold. Otherwise: **intermediate**. The thumb has its own extension angle threshold.')
    st.write('81 geometric features: 50 per-finger values, 10 fingertip distances, 12 thumb-to-knuckle distances and 9 spread/crossing/depth values. The precision table checks binary finger states; it does not validate all 81 measurements.')
