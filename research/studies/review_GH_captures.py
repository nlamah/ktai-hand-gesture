"""Audit the latest G/H-model capture round without fitting or changing a model."""
from hashlib import sha256
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from research.inputs import ANGLE_COLUMNS,write_csv
from research.geometry import _angle
from research.live_model import load_bundle,model_inputs,preprocessed_inputs
from research.model_choices import MODEL_CHOICES

OUT=Path('results/GH_capture_review_20261031')


def main():
    records=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if Path(r['model']['path']).name=='asl_shin_angles_GH_rotation.joblib':records.append((path,r))
    if not records:raise ValueError('No captures from the G/H rotation model')
    session=records[-1][1]['session_id']
    records=[(p,r) for p,r in records if r['session_id']==session]
    OUT.mkdir(parents=True,exist_ok=True)
    bundles={key:load_bundle(path) for key,_,path in MODEL_CHOICES[:5]}
    paths={key:path for key,_,path in MODEL_CHOICES[:5]}
    hashes={key:sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    meta=json.loads(Path('data/asl_kaggle/research/inputs_metadata.json').read_text())
    arrays=np.load('data/asl_kaggle/research/inputs.npz')
    X=np.c_[arrays['X'][:,:820],arrays['angles']]
    labels=np.array([s['label'] for s in meta['samples']])
    transformed=preprocessed_inputs(bundles['angles'],X)
    comparisons=[];details=[]
    fig,axes=plt.subplots((len(records)+2)//3,3,figsize=(12,3.9*((len(records)+2)//3)))
    for ax in np.ravel(axes):ax.axis('off')
    for i,(path,r) in enumerate(records):
        assert r['model']['sha256']==hashes['GH_rotation']
        for name,info in r['files'].items():assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        inputs=model_inputs(bundles['angles'],r['detection'])
        np.testing.assert_array_equal(inputs[0],np.array(r['raw_model_features'],np.float32))
        predictions={key:str(bundle['estimator'].predict(model_inputs(bundle,r['detection']))[0]) for key,bundle in bundles.items()}
        assert predictions['GH_rotation']==r['predicted_letter']
        comparisons.append({'id':r['id'],'local_time':datetime.fromisoformat(r['saved_at_utc']).astimezone(ZoneInfo('Europe/Amsterdam')).isoformat(),'intended':r['expected_letter'],**predictions})
        z=preprocessed_inputs(bundles['angles'],inputs)[0]
        neighbors={}
        for name,section in [('joint_angles',slice(820,None)),('full_model_inputs',slice(None))]:
            distances=np.sum((transformed[:,section]-z[section])**2,axis=1)
            order=np.argsort(distances)[:5]
            neighbors[name]=[{'id':meta['samples'][j]['id'],'letter':str(labels[j]),'distance':float(np.sqrt(distances[j]))} for j in order]
        joint_details={}
        for k,column in enumerate(ANGLE_COLUMNS):
            joint_details[column]={'capture_deg':float(inputs[0,820+k]),
                'training_by_letter':{l:{'p05':float(np.percentile(arrays['angles'][labels==l,k],5)),
                                          'median':float(np.median(arrays['angles'][labels==l,k])),
                                          'p95':float(np.percentile(arrays['angles'][labels==l,k],95))}
                                      for l in sorted({r['expected_letter'],r['predicted_letter']})}}
        points=np.asarray(r['detection']['image_landmarks']);world=np.asarray(r['detection']['world_landmarks'])
        palm=np.linalg.norm(world[9]-world[0]);tip_gap=float(np.linalg.norm(world[4]-world[8])/palm)
        with Image.open(path.parent/'model_input.png') as source:
            pixel_points=points[:,:2]*source.size
        pixel_angles={joint:_angle(*pixel_points[list(indices)]) for joint,indices in
                      [('index_pip',(5,6,7)),('middle_pip',(9,10,11))]}
        ranking=sorted(r['decision_scores'],key=r['decision_scores'].get,reverse=True)
        details.append({'id':r['id'],'path':str(path),'expected':r['expected_letter'],'prediction':r['predicted_letter'],
                        'neighbors':neighbors,'joint_angles':joint_details,'pixel_projected_angles_deg':pixel_angles,
                        'intended_decision_score_rank':ranking.index(r['expected_letter'])+1,'thumb_index_tip_gap_per_palm_length':tip_gap,
                        'capture_json_sha256':sha256(path.read_bytes()).hexdigest()})
        image=Image.open(path.parent/'model_input.png');w,h=image.size
        xs=points[:,0]*w;ys=points[:,1]*h
        margin=max(np.ptp(xs),np.ptp(ys))*.2
        box=(max(0,int(xs.min()-margin)),max(0,int(ys.min()-margin)),min(w,int(xs.max()+margin)),min(h,int(ys.max()+margin)))
        ax=np.ravel(axes)[i];ax.imshow(image.crop(box))
        ax.set_title(f"{i+1}. {r['expected_letter']} -> {r['predicted_letter']}   {r['id'][9:15]} UTC\n"
                     f"Index PIP {inputs[0,823]:.0f} / middle PIP {inputs[0,826]:.0f} deg",fontsize=11)
    fig.suptitle('Selected confusions: latest GH-handrug round (not a test set)',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'hand_crops.png',dpi=160);plt.close(fig)
    write_csv(OUT/'model_comparison.csv',comparisons)
    result={'session_id':session,'round':records[0][1].get('capture_round'),'captures':len(records),
            'model_hashes':hashes,'all_capture_image_hashes_verified':True,'all_saved_predictions_replayed':True,
            'known_selected_errors_not_independent_test':True,'comparison':comparisons,'details':details}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'round':result['round'],'comparison':comparisons,
        'G_evidence':next(d for d in details if d['expected']=='G')},indent=2))


if __name__=='__main__':main()
