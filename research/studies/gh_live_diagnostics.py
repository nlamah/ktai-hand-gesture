"""Replay four fixed, user-selected live examples; no training or accuracy estimate."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from research.data import atomic_json
from research.feature_review import CONNECTIONS, hand_crop
from research.geometry import measure_hand_features
from research.inputs import ANGLE_COLUMNS, write_csv
from research.live_model import load_bundle, model_inputs, predict_letter

OUT=Path('results/gh_live_diagnostics_20261107')
CAPTURES=(
    ('20261107T161853209081Z_4883acb7','handrug zichtbaar'),
    ('20261107T161915381494Z_aac80023','handrug zichtbaar'),
    ('20261107T161907071284Z_0582e897','handrug zichtbaar'),
    ('20261107T162048609081Z_600038ed','palm zichtbaar'),
)
MODELS={'reference':Path('models/asl_shin_angles.joblib'),
        'rotation':Path('models/asl_shin_angles_GH_y180.joblib')}
MEASURES=('middle_mcp_angle_deg','middle_pip_angle_deg','middle_dip_angle_deg',
          'thumb_mcp_signed_deviation_deg')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'version':'fixed-live-gh-diagnostics-v1',
        'source_ids':[i for i,_ in CAPTURES],
        'selection':'All four new free captures found when the user reported live G failures; explicit IDs, no withdrawn guided captures',
        'scope':'Qualitative error analysis, user-provided intended letters, two camera rounds; not a representative sample or independent accuracy estimate',
        'view_labels':'Visual reading of saved photographs, not measured view angles or an expert ASL correctness assessment',
        'replay':'Original stored image/world landmarks through both frozen deployment models; no redetection, fitting, calibration or thresholds',
        'model_sha256':{k:digest(v) for k,v in MODELS.items()},
        'capture_sha256':{i:digest(Path('captures/confusions')/i/'capture.json') for i,_ in CAPTURES},
        'cache_sha256':{p:digest(p) for p in ['data/asl_kaggle/research/inputs.npz','data/asl_kaggle/research/inputs_metadata.json']},
        'source_sha256':digest(__file__)}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Use a new directory for changed protocol.'
    else:atomic_json(path,protocol)
    bundles={k:load_bundle(p) for k,p in MODELS.items()}
    rows=[];sources=[];fig,axes=plt.subplots(2,2,figsize=(9.2,6.4),layout='constrained')
    for panel,((ident,view),ax) in enumerate(zip(CAPTURES,axes.flat)):
        folder=Path('captures/confusions')/ident;r=json.loads((folder/'capture.json').read_text())
        assert r['protocol'] is None and r['detection']['status']=='ok'
        assert r['model']['sha256']==protocol['model_sha256']['rotation']
        for name,info in r['files'].items():assert digest(folder/name)==info['sha256']
        d=r['detection'];pred={k:predict_letter(b,d) for k,b in bundles.items()}
        assert pred['rotation']['letter']==r['predicted_letter']
        np.testing.assert_array_equal(model_inputs(bundles['rotation'],d)[0],np.asarray(r['raw_model_features'],np.float32))
        f=measure_hand_features(d['world_landmarks'],d['handedness'])
        row={'panel':chr(65+panel),'id':ident,'expected':r['expected_letter'],'saved_prediction':r['predicted_letter'],
             'view_visual':view,'handedness_estimate':d['handedness'],'status':d['status'],
             **{k:p['letter'] for k,p in pred.items()},**{k:f[k] for k in MEASURES}}
        rows.append(row)
        with Image.open(folder/'model_input.png') as image:
            assert list(image.size)==d['image_size']
            crop,box=hand_crop(image,d['image_landmarks'],overlay=False)
            points=np.asarray(d['image_landmarks'])[:,:2]*image.size-np.array(box[:2])
        ax.imshow(crop)
        for a,b in CONNECTIONS:ax.plot(points[[a,b],0],points[[a,b],1],color='#00ba89',lw=1.5)
        ax.scatter(points[:,0],points[:,1],s=13,c='#ffcc33',edgecolors='none',zorder=3)
        ax.set_title(f"{row['panel']}. Bedoeld {row['expected']} · {view}\nReferentie: {row['reference']} | Rotatie: {row['rotation']}",fontsize=11)
        ax.axis('off')
        sources.append({'id':ident,'crop_box':list(box),'capture_json':str(folder/'capture.json'),
                        'frame_sha256':digest(folder/'model_input.png'),'view_visual':view})
    figure=Path('paper/figures/gh_live_diagnostics.pdf');fig.savefig(figure);plt.close(fig)
    write_csv(OUT/'predictions.csv',rows)
    meta=json.loads(Path('data/asl_kaggle/research/inputs_metadata.json').read_text())
    labels=np.array([s['label'] for s in meta['samples']])
    with np.load('data/asl_kaggle/research/inputs.npz') as data:angles=data['angles']
    comparisons=[]
    for letter in ['G','H']:
        for key in MEASURES:
            v=angles[labels==letter,ANGLE_COLUMNS.index(key)]
            p=np.nanpercentile(v,[5,25,50,75,95])
            comparisons.append({'letter':letter,'measure':key,'n':len(v),**dict(zip(['p05','p25','median','p75','p95'],map(float,p)))})
    write_csv(OUT/'training_angles.csv',comparisons)
    atomic_json(OUT/'verification.json',{'status':'passed','captures_replayed':4,'models':2,
        'image_capture_model_hashes_checked':True,'recorded_prediction_and_input_vector_reproduced':True,
        'redetections':0,'training_sources_added':0,'withdrawn_capture_sources':0,
        'new_live_accuracy_estimate':False,'figure_sources':sources,'figure_sha256':digest(figure),
        'csv_sha256':{p.name:digest(p) for p in OUT.glob('*.csv')}})
    tex=[]
    for r in rows:tex.append(f"{r['panel']} & {r['expected']} & {r['view_visual']} & {r['reference']} & {r['rotation']} \\\\")
    Path('paper/gh_live_diagnostics_rows_nl.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(rows,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
