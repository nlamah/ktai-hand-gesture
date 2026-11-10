"""Export real saved-hand examples and the tool's shared feature comparisons."""
from hashlib import sha256
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from research.data import atomic_json
from research.feature_review import review_data, hand_crop, KEYS, NAMES, STATES, english
from research.live_model import DEFAULT_MODEL, load_bundle, predict_letter
from research.mirroring import mirror_image, pixel_hash

OUT=Path('results/appendix_visuals_01')
FIG=Path('paper/figures')
CALIBRATION_FREE=Path('models/asl_shin_angles_GH_y180.joblib')
CAPTURES=('20261031T152108010733Z_9233886f','20261031T152116143793Z_11756e66',
          '20261031T181839806179Z_9ef071cc','20261031T212827197875Z_759e34da')


def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


def feature_panel(data,name):
    fig=plt.figure(figsize=(9.3,3.6),layout='constrained')
    grid=fig.add_gridspec(1,2,width_ratios=[1,2.4])
    photo=fig.add_subplot(grid[0,0]);table=fig.add_subplot(grid[0,1]);photo.axis('off');table.axis('off')
    photo.imshow(data['crop'])
    photo.set_title(f"Saved H capture\nModel prediction: {data['prediction']['letter']}",fontsize=10)
    values=data['measurements']
    photo.text(.5,-.10,f"Index PIP {values['index_pip_angle_deg']:.1f}°\nMiddle PIP {values['middle_pip_angle_deg']:.1f}°",
               ha='center',va='top',transform=photo.transAxes,fontsize=9)
    rows=[row for row in data['rows'] if row['key'] in KEYS]
    cells=[[NAMES[row['key']],english(row['expected']),english(row['actual']),STATES[row['status']]] for row in rows]
    cells=[['\n'.join(textwrap.wrap(value,width)) for value,width in zip(row,[19,15,15,15])] for row in cells]
    t=table.table(cellText=cells,colLabels=['Feature','Expected','Observed','Check'],
                  colWidths=[.31,.23,.23,.23],cellLoc='left',colLoc='left',bbox=[0,0,1,.96])
    t.auto_set_font_size(False);t.set_fontsize(10.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#cbd5dc');cell.set_linewidth(.5)
        if r==0:cell.set_facecolor('#e9eef2');cell.get_text().set_weight('bold')
        elif c==3:
            status=rows[r-1]['status'];cell.set_facecolor('#dbf0e0' if status=='match' else '#fbe0dd' if status=='mismatch' else '#eeeeee')
    fig.suptitle(f"Feature inspection: compare against {data['target']}",fontsize=12,weight='bold')
    save(fig,name)


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    references=Path('assets/asl_feature_references.xlsx');thresholds=Path('config/thresholds.json')
    records=[];camera={}
    alternative=load_bundle(CALIBRATION_FREE)
    for id_ in CAPTURES:
        folder=Path('captures/confusions')/id_
        data=review_data(folder,references,thresholds,DEFAULT_MODEL)
        data['rotated']=predict_letter(alternative,data['record']['detection'])['letter'];camera[id_]=data
        records.append({'kind':'selected_camera_development','id':id_,'intended':data['record']['expected_letter'],
                        'reference':data['prediction']['letter'],'gh_y180':data['rotated'],'crop_box':data['box'],
                        'capture_sha256':sha256((folder/'capture.json').read_bytes()).hexdigest(),
                        'image_sha256':sha256((folder/'model_input.png').read_bytes()).hexdigest(),
                        'role':'Previously inspected; not fitting data for these two classifiers; reused in other calibration experiments'})
    h=camera[CAPTURES[1]]
    g_reference=review_data(Path('captures/confusions')/CAPTURES[1],references,thresholds,DEFAULT_MODEL,'G')
    assert h['prediction']==g_reference['prediction']
    feature_panel(h,'feature_review_H');feature_panel(g_reference,'feature_review_G')
    atomic_json(OUT/'feature_comparisons.json',{
        'capture':CAPTURES[1],'model_prediction':h['prediction']['letter'],'rows_H':h['rows'],'rows_G':g_reference['rows'],
        'measurements':h['measurements'],'thresholds':h['parameters'],
        'note':'Report rendering of shared live_feature_rows/review_data, not a live-webcam screenshot'})
    # First matching ID is a reproducible, explicitly selected illustration, not a prevalence estimate.
    raw=pd.read_csv('results/mirror_01/predictions.csv',keep_default_na=False)
    raw=raw[raw.model=='svm'];selected=[]
    for letter in ('A','H'):
        o=raw[(raw.label==letter)&(raw.view=='original')].set_index('id')
        m=raw[(raw.label==letter)&(raw.view=='mirrored_rgb')].set_index('id')
        pair=o.join(m,lsuffix='_o',rsuffix='_m')
        mask=(pair.status_o=='ok')&(pair.status_m=='ok')&(pair.original_training_o==letter)&\
             (pair.original_training_m!=letter)&(pair.mirror_training_m==letter)&\
             (pair.detected_handedness_o=='Right')&(pair.detected_handedness_m=='Left')
        row=pair[mask].sort_index().iloc[0]
        assert row.fold_o==row.fold_m and row.group_o==row.group_m
        selected.append({'id':row.name,'label':letter,'fold':int(row.fold_o),
                         'original':row.original_training_o,'reflected_original_model':row.original_training_m,
                         'reflected_mirror_model':row.mirror_training_m,
                         'reflected_original_model_original_view':row.mirror_training_o})
    g=raw[(raw.label=='G')&(raw.view=='original')&(raw.status=='ok')&
          (raw.original_training=='G')&(raw.detected_handedness=='Right')].sort_values('id').iloc[0]
    sample_ids={row['id'] for row in selected}|{g.id}
    cached={}
    for chunk in sorted(Path('runs/mirror_01/detections').glob('chunk_*.npz')):
        with np.load(chunk) as data:
            for i,id_ in enumerate(data['ids']):
                if str(id_) in sample_ids:
                    cached[str(id_)]={k:data[k][i] for k in ('image_landmarks','pixel_sha256','status','handedness')}
    assert set(cached)==sample_ids
    manifest=json.loads(Path('data/asl_kaggle/manifest.json').read_text())
    samples={s['id']:s for s in manifest['samples'] if s['id'] in sample_ids}
    images={}
    for id_,sample in samples.items():
        path=Path('data/asl_kaggle')/sample['image_path']
        with Image.open(path) as image:images[id_]=image.convert('RGB')
        assert pixel_hash(images[id_])==str(cached[id_]['pixel_sha256'][0])
        assert pixel_hash(mirror_image(images[id_]))==str(cached[id_]['pixel_sha256'][1])
        records.append({'kind':'dataset_illustration','id':id_,'label':sample['label'],
                        'filename':sample['filename'],'path':str(path),'pixel_sha256':pixel_hash(images[id_])})
    fig,axes=plt.subplots(3,2,figsize=(8,8),layout='constrained')
    ref_ids=[g.id,next(row['id'] for row in selected if row['label']=='H')]
    for j,(id_,letter) in enumerate(zip(ref_ids,('G','H'))):
        crop,_=hand_crop(images[id_],cached[id_]['image_landmarks'][0])
        axes[0,j].imshow(crop);axes[0,j].axis('off')
        axes[0,j].set_title(f"Dataset label {letter} · {samples[id_]['filename']}\nIllustrative crop; not a camera test",fontsize=10)
    for i,id_ in enumerate(CAPTURES):
        data=camera[id_];ax=axes[1+i//2,i%2];ax.imshow(data['crop']);ax.axis('off')
        view='palm control' if i==3 else 'hand back, development example'
        ax.set_title(f"Intended {data['record']['expected_letter']} · {view}\nReference [2]: {data['prediction']['letter']} | G/H y180 [9]: {data['rotated']}",fontsize=10)
    save(fig,'gh_webcam_examples')
    fig,axes=plt.subplots(2,2,figsize=(8,6.2),layout='constrained')
    for i,row in enumerate(selected):
        image=images[row['id']];data=cached[row['id']]
        for view,im in enumerate((image,mirror_image(image))):
            crop,_=hand_crop(im,data['image_landmarks'][view]);ax=axes[i,view]
            ax.imshow(crop);ax.axis('off')
            title=f"{samples[row['id']]['filename']} · " + ('original' if view==0 else 'synthetically mirrored')
            if view==0:subtitle=f"Detector: Right | reference: {row['original']}"
            else:subtitle=f"Detector: Left | reference: {row['reflected_original_model']} | mirror training: {row['reflected_mirror_model']}"
            ax.set_title(title+'\n'+subtitle,fontsize=10)
    save(fig,'mirror_examples')
    atomic_json(OUT/'mirror_selection.json',{'selection':'Lexicographically first ID per A/H satisfying accepted Right/Left, original reference correct, mirror reference wrong, mirror-trained prediction correct',
                'role':'Selected illustrative held-out source predictions, not independent real left-handed signers',
                'examples':selected})
    atomic_json(OUT/'provenance.json',{'sources':records,
        'models':{str(p):sha256(p.read_bytes()).hexdigest() for p in (DEFAULT_MODEL,CALIBRATION_FREE)},
        'reference_sha256':sha256(references.read_bytes()).hexdigest(),'threshold_sha256':sha256(thresholds.read_bytes()).hexdigest(),
        'code_sha256':{p:sha256(Path(p).read_bytes()).hexdigest() for p in ('research/feature_review.py','research/studies/appendix_visuals.py','research/live.py','research/workbook.py')},
        'figures':{str(p):sha256(p.read_bytes()).hexdigest() for p in FIG.glob('*.pdf') if p.stem in ('feature_review_H','feature_review_G','gh_webcam_examples','mirror_examples')},
        'checks':'Capture image hashes, both mirror pixel hashes, current model replay and unchanged prediction across target-letter comparisons pass'})
    print('Exported four data-backed figures; feature views are report renders, not screenshots.',flush=True)


if __name__=='__main__':main()
