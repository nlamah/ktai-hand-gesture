"""Collect data only. Start: streamlit run apps/collect_app.py -- --project data/ngt"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io import BytesIO
import tempfile
import pandas as pd
from PIL import Image
import streamlit as st
from gesturelab.datasets.alphabet import initialize_alphabet,LETTERS
from gesturelab.camera.runtime import reference_for,new_session_name
from gesturelab.camera.ui import render_capture,camera_status,camera_running
from gesturelab.datasets.data import (read_manifest,add_image,analyze_project,metadata_table,edit_metadata,
                             export_project,fingerprint)

parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--project',default='data/ngt')
args,_=parser.parse_known_args()
st.set_page_config(page_title='NGT · Training data',page_icon='📸',layout='wide')
st.title('1 · Collect training data')
st.caption('Alphabet references → your webcam photos → labels and sessions → dataset export')
st.session_state.setdefault('collect_project',str(Path(args.project).resolve()))
project=Path(st.sidebar.text_input('Project directory',key='collect_project')).expanduser()
if any(not (project/'references/ngt_alphabet'/(letter+'.png')).exists() for letter in LETTERS):
    initialize_alphabet(project)
samples=read_manifest(project)['samples']
camera_status()
c1,c2,c3=st.columns(3)
c1.metric('Your images',len(samples))
c2.metric('Usable hand detections',sum(s['status']=='ok' and s['included'] for s in samples))
c3.metric('Recording sessions',len({(s['participant'],s['session']) for s in samples if s['included']}))
page=st.sidebar.radio('Section',['Alphabet','Guided capture','Upload photos','Check dataset','Export'])
if page=='Alphabet':
    st.write('These are crops of the Dutch manual alphabet (source: Geocachen.be; see assets/SOURCES.md). Use them as pose references; the drawings are not training images.')
    for start in range(0,len(LETTERS),6):
        cols=st.columns(6)
        for col,letter in zip(cols,LETTERS[start:start+6]):
            path,_=reference_for(project,letter)
            col.image(str(path),caption=letter,width=130)
elif page=='Guided capture':
    preset=st.selectbox('Capture plan',['Start with 8 letters','All 26 letters'])
    defaults=['A','B','C','D','E','F','K','L'] if preset.startswith('Start') else LETTERS
    render_capture(project,samples,default_labels=defaults,allowed_labels=LETTERS)
elif page=='Upload photos':
    st.write('Use this option for existing photos. Keep images from the same recording session together.')
    c1,c2,c3=st.columns(3)
    label=c1.selectbox('Letter',LETTERS)
    participant=c2.text_input('Participant','participant1')
    session=c3.text_input('Actual recording session','session1')
    condition=st.text_input('Condition','baseline')
    role=st.selectbox('Usage',['development','test'],format_func=lambda r:'Development' if r=='development' else 'Reserve for final test')
    uploads=st.file_uploader('Photos of this letter',type=['jpg','jpeg','png','webp'],accept_multiple_files=True)
    if st.button('Save photos',disabled=not uploads or camera_running()):
        try:
            n=sum(add_image(project,u.getvalue(),u.name,label,session,participant,condition,role)[1] for u in uploads)
            st.success(f'{n} new images saved; identical images skipped.')
        except (ValueError,OSError) as error:
            st.error(str(error))
    if st.button('Analyze pending photos',disabled=camera_running()):
        try:
            with st.spinner('Computing hand landmarks...'):
                count=analyze_project(project)
            st.success(f'{count} photos analyzed.')
        except Exception as error:
            st.error(str(error))
elif page=='Check dataset':
    rows=[]
    for letter in LETTERS:
        part=[s for s in samples if s['label']==letter and s['included']]
        valid=[s for s in part if s['role']=='development' and s['status']=='ok']
        rows.append({'Letter':letter,'Photos':len(part),'Usable development images':len(valid),
                     'Development sessions':len({(s['participant'],s['session']) for s in valid}),
                     'Test photos':sum(s['role']=='test' for s in part)})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.caption('A session is one actual recording round, not one photo. Correct incorrect session names below. Do not create arbitrary groups.')
    table=metadata_table(project)
    if not table.empty:
        edited=st.data_editor(table,hide_index=True,key='metadata_'+fingerprint(project),
            disabled=['id','filename','created_at','status'],
            column_config={'role':st.column_config.SelectboxColumn(options=['development','test'])})
        if st.button('Save corrections',disabled=camera_running()):
            try:
                edit_metadata(project,edited);st.rerun()
            except ValueError as error:
                st.error(str(error))
        s=st.selectbox('View image',samples,format_func=lambda s:f"{s['label']} · {s['session']} · {s['id'][:8]}")
        st.image(str(project/s['image_path']),width=400)
else:
    st.write('Export the entire dataset, including unchanged source images. Results in the separate experiments directory are not automatically included.')
    if st.button('Create dataset ZIP',disabled=not samples or camera_running()):
        with tempfile.TemporaryDirectory() as tmp:
            output=export_project(project,Path(tmp)/'dataset.zip')
            st.session_state['export_data']=output.read_bytes()
    if st.session_state.get('export_data'):
        st.download_button('Download dataset',st.session_state['export_data'],'ngt_dataset.zip','application/zip')
