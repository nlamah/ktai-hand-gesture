"""Fixed G-versus-H diagnostic: 3D, projected 2D, and combined angles only."""
import csv
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research.data import atomic_json
from research.image_angles import projected_angles,PIXEL_ANGLE_COLUMNS
from research.inputs import ANGLE_COLUMNS,write_csv
from research.live_model import detection_inputs

CACHE=Path('data/asl_kaggle/research')
OUT=Path('results/gh_classifier_01')
MODELS=Path('models/gh_diagnostic')
VARIANTS=('3d','2d','both')
VERSION='gh-angle-diagnostic-v1'


def select_inputs(world,pixel,variant):
    if variant=='3d':return np.asarray(world,dtype=np.float32)
    if variant=='2d':return np.asarray(pixel,dtype=np.float32)
    if variant=='both':return np.concatenate([world,pixel],axis=-1).astype(np.float32)
    raise ValueError('Choose 3d, 2d or both')


def classifier():
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                         SVC(C=10,kernel='rbf',gamma='scale',class_weight='balanced'))


def measurements(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=list('GH'),zero_division=0)
    return {'images':len(y),'correct':int((y==pred).sum()),'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=list('GH'),average='macro',zero_division=0)),
        'confusion_matrix_GH':confusion_matrix(y,pred,labels=list('GH')).tolist(),
        'per_letter':{l:{'images':int(support[i]),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate('GH')}}


def load_inputs():
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    summary=json.loads((CACHE/'summary.json').read_text())
    protocol=json.loads(Path('results/angles_2d_01/protocol.json').read_text())
    assert protocol['input_hashes']==meta['hashes']
    assert sha256((CACHE/'inputs.npz').read_bytes()).hexdigest()==protocol['input_cache_sha256']
    pixel=np.load('results/angles_2d_01/pixel_angles.npz')
    assert pixel['ids'].tolist()==[s['id'] for s in meta['samples']]
    arrays=np.load(CACHE/'inputs.npz')
    labels=np.array([s['label'] for s in meta['samples']]);mask=np.isin(labels,list('GH'))
    ids=np.array([s['id'] for s in meta['samples']])[mask]
    groups=np.array([s['group'] for s in meta['samples']])[mask]
    return arrays['angles'][mask],pixel['angles'][mask],labels[mask],ids,groups,meta,summary


def main():
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    world,pixel,y,ids,groups,meta,summary=load_inputs()
    config={'version':VERSION,'variants':list(VARIANTS),'input_hashes':meta['hashes'],
        'input_cache_sha256':sha256((CACHE/'inputs.npz').read_bytes()).hexdigest(),
        'pixel_cache_sha256':sha256(Path('results/angles_2d_01/pixel_angles.npz').read_bytes()).hexdigest(),
        'folds_sha256':sha256((CACHE/'summary.json').read_bytes()).hexdigest(),
        'model':'RBF SVM C=10 gamma=scale balanced; train-only median imputation and standard scaling',
        'classes':['G','H'],'world_columns':list(ANGLE_COLUMNS),'pixel_columns':list(PIXEL_ANGLE_COLUMNS),
        'policy':'Three predeclared representations, no tuning on webcam outcomes. Captures excluded from training. Binary results cannot be compared as a 26-class improvement.'}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==config,'Protocol mismatch'
    else:atomic_json(protocol,config)
    models={};results={};oof=[];fold_rows=[]
    for variant in VARIANTS:
        X=select_inputs(world,pixel,variant);pred=np.full(len(y),'',dtype='<U1');assigned=np.zeros(len(y),dtype=int)
        for f in summary['folds']:
            train=np.flatnonzero(np.isin(groups,f['train_groups']));test=np.flatnonzero(np.isin(groups,f['test_groups']))
            assert not set(groups[train])&set(groups[test]) and set(y[train])==set('GH') and set(y[test])==set('GH')
            checkpoint=OUT/f'{variant}_fold{f["fold"]}.npz'
            if checkpoint.exists():
                saved=np.load(checkpoint);assert saved['ids'].tolist()==ids[test].tolist();pr=saved['prediction']
            else:
                estimator=classifier().fit(X[train],y[train]);pr=estimator.predict(X[test])
                np.savez_compressed(checkpoint,ids=ids[test],prediction=pr)
            pred[test]=pr;assigned[test]+=1
            fold_rows.append({'variant':variant,'fold':f['fold'],**measurements(y[test],pr)})
            print(json.dumps({'variant':variant,'fold':f['fold'],'accuracy':fold_rows[-1]['accuracy']}),flush=True)
        assert np.all(assigned==1) and not np.any(pred=='')
        results[variant]=measurements(y,pred)
        oof.extend({'variant':variant,'id':str(ids[i]),'label':str(y[i]),'prediction':str(pred[i])} for i in range(len(y)))
        model_path=MODELS/f'gh_{variant}.joblib'
        if model_path.exists():
            bundle=joblib.load(model_path);assert bundle['protocol']==config and bundle['variant']==variant
        else:
            bundle={'version':VERSION,'variant':variant,'protocol':config,'sklearn_version':sklearn.__version__,
                'training_images':len(y),'created_at_utc':datetime.now(timezone.utc).isoformat(),'estimator':classifier().fit(X,y),
                'scope':'Diagnostic G/H-only model. Always chooses G or H; not a replacement for the 26-letter recognizer.'}
            tmp=model_path.with_suffix('.partial.joblib')
            try:joblib.dump(bundle,tmp,compress=3);tmp.replace(model_path)
            finally:tmp.unlink(missing_ok=True)
            atomic_json(model_path.with_suffix('.json'),{k:v for k,v in bundle.items() if k!='estimator'})
        models[variant]=bundle['estimator']
    atomic_json(OUT/'cv_summary.json',{'representations':results,'folds':fold_rows,'images':len(y),'G_images':int((y=='G').sum()),'H_images':int((y=='H').sum())})
    write_csv(OUT/'oof_predictions.csv',oof)
    captures=[];capture_world=[];capture_pixel=[]
    for path in sorted(Path('captures/confusions').glob('*/capture.json')):
        r=json.loads(path.read_text())
        if r['expected_letter'] not in list('GH') or r['detection']['status']!='ok':continue
        for name,info in r['files'].items():assert sha256((path.parent/name).read_bytes()).hexdigest()==info['sha256']
        with Image.open(path.parent/'model_input.png') as image:size=list(image.size)
        w=detection_inputs(r['detection'])[0,820:835];p=projected_angles(r['detection']['image_landmarks'],size)
        row={'id':r['id'],'session_id':r.get('session_id'),'intended':r['expected_letter'],'saved_prediction':r['predicted_letter'],
             **{v:str(models[v].predict(select_inputs(w,p,v)[None,:])[0]) for v in VARIANTS}}
        captures.append(row);capture_world.append(w);capture_pixel.append(p)
    write_csv(OUT/'capture_predictions.csv',captures)
    atomic_json(OUT/'capture_summary.json',{'captures':captures,'correct':{v:sum(r[v]==r['intended'] for r in captures) for v in VARIANTS},
        'note':'Known selected G/H development captures only, not a representative held-out webcam test.'})
    existing={}
    with Path('results/angles/oof_predictions_svm.csv').open(encoding='utf-8-sig') as f:all_rows=list(csv.DictReader(f))
    for variant in ['shin','shin_angles']:
        rr=[r for r in all_rows if r['variant']==variant];existing[variant]={}
        for letter in 'GH':
            target=[r for r in rr if r['label']==letter];errors=Counter(r['prediction'] for r in target if r['prediction']!=letter)
            fp=sum(r['label']!=letter and r['prediction']==letter for r in rr);correct=sum(r['prediction']==letter for r in target)
            existing[variant][letter]={'images':len(target),'correct':correct,'recall':correct/len(target),
                'precision':correct/(correct+fp),'wrong_predictions':dict(errors)}
    atomic_json(OUT/'existing_26class_GH.json',existing)
    plot(world,pixel,y,capture_world,capture_pixel,captures)
    report(results,existing,captures,fold_rows)
    atomic_json(OUT/'verification.json',{'images_per_variant':len(y),'same_original_group_folds':True,
        'each_source_validated_once_per_variant':True,'train_only_preprocessing':True,'captures_excluded_from_fit':True,
        'model_hashes':{v:sha256((MODELS/f'gh_{v}.joblib').read_bytes()).hexdigest() for v in VARIANTS},
        'capture_image_hashes_verified':True,'capture_count':len(captures)})
    print(json.dumps({'cv':results,'captures':captures}),flush=True)


def plot(world,pixel,y,cw,cp,captures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,values,cvalues,title in zip(axes,[world,pixel],[cw,cp],['Estimated 3D angles','Projected 2D pixel angles']):
        for letter,color in [('G','#1976a8'),('H','#b65e18')]:
            mask=y==letter;ax.scatter(values[mask,3],values[mask,6],s=8,alpha=.17,c=color,label=f'Training {letter}')
        for i,(value,r) in enumerate(zip(cvalues,captures)):
            ax.scatter(value[3],value[6],marker='*',s=180,c='#cc1844',edgecolors='black',linewidths=.5,zorder=4)
            ax.annotate(f"{r['intended']}{i+1}",(value[3],value[6]),xytext=(6,6),textcoords='offset points',fontsize=9)
        ax.set(xlabel='Index PIP angle (degrees)',ylabel='Middle PIP angle (degrees)',title=title,xlim=(0,185),ylim=(0,185));ax.legend()
    fig.suptitle('G/H training geometry and selected webcam captures (stars)')
    fig.tight_layout();fig.savefig(OUT/'GH_angle_distributions.png',dpi=160);plt.close(fig)


def report(results,existing,captures,folds):
    lines=['# G-versus-H diagnostic classifiers','',
        'Three fixed SVMs use only 15 estimated 3D angles, 14 projected 2D angles, or all 29 together. '
        'No Shin distances/directions, synthetic augmentation or webcam training examples are used. '
        'The original five filename-block folds are restricted to G/H, with scaling and imputation fitted only on the training partition. '
        'This is an easier binary task: its scores are not evidence of improvement on all 26 letters.', '',
        '## Is G already difficult in the existing dataset?', '',
        '| Existing 26-class model | Letter | Correct | Recall | Precision | Errors |','|---|---|---:|---:|---:|---|']
    for variant,label in [('shin','Reference'),('shin_angles','Joint angles')]:
        for l in 'GH':
            m=existing[variant][l];errors=', '.join(f'{k}: {v}' for k,v in m['wrong_predictions'].items())
            lines.append(f"| {label} | {l} | {m['correct']}/{m['images']} | {100*m['recall']:.3f}% | {100*m['precision']:.3f}% | {errors} |")
    lines+=['', 'G is well recognized on the accepted dataset images. This does not imply reliable webcam recognition: '
        'only 2,430/3,000 G and 2,371/3,000 H source images were accepted by the landmark workflow. '
        'Filename blocks are not verified independent people, and online/webcam hand views may differ.', '',
        '## Binary five-fold results', '', '| Representation | Accuracy | Macro-F1 | G recall | H recall |','|---|---:|---:|---:|---:|']
    for v,label in [('3d','3D only'),('2d','2D only'),('both','2D + 3D')]:
        m=results[v];lines.append(f"| {label} | {100*m['accuracy']:.3f}% | {m['macro_f1']:.5f} | {100*m['per_letter']['G']['recall']:.3f}% | {100*m['per_letter']['H']['recall']:.3f}% |")
    lines+=['', '## Previously selected G/H webcam errors', '', '| Intended | Saved prediction | 3D only | 2D only | Both |','|---|---|---|---|---|']
    for r in captures:lines.append(f"| {r['intended']} ({r['id']}) | {r['saved_prediction']} | {r['3d']} | {r['2d']} | {r['both']} |")
    lines+=['', 'These known, manually selected errors were used to motivate this investigation. They are development diagnostics, '
        'not an independent webcam accuracy estimate. A binary model always chooses G or H, even if shown K, X or another letter. '
        'It must not be installed as an unrestricted replacement for the main recognizer.', '',
        '![Angle distributions](GH_angle_distributions.png)', '',
        'The plot shows just index/middle PIP angles; the models use all listed joint angles. '
        'Projected 2D angles are view-dependent, and 3D angles are landmark estimates rather than anatomical ground truth.', '',
        '## Reproduce', '', '```bash','MPLCONFIGDIR=/tmp/ktai-mpl python -m research.studies.compare_gh_classifier','```','',
        'Requires the frozen original inputs and `results/angles_2d_01/pixel_angles.npz` from the preceding 2D-angle experiment. '
        'Model bundles are saved separately under `models/gh_diagnostic/`; earlier models remain intact. '
        'The fixed protocol and fold checkpoints support reruns without choosing a favorable split or overwriting existing bundles.', '']
    (OUT/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
