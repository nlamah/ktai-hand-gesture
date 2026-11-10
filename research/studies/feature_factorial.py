"""Frozen eight-arm feature study; matched full CV and held-out grouped permutation."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from research.data import atomic_json
from research.feature_study import VERSION,VARIANTS,BLOCKS,matrix,source_columns,estimator,permutation_groups,permute_group
from research.inputs import LABELS,write_csv,encode_additions,encode_angles
from research.geometry import measure_hand_features
from research.references import load_reference_workbook
from research.shin import combined_features
from research.studies.compare_models import load_inputs

CACHE=Path('data/asl_kaggle/research')
OUT=Path('results/feature_factorial_01')
WORK=Path('runs/feature_factorial_01')
SEED=20261107
REPEATS=5
CAP=100


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def references():return load_reference_workbook(Path('assets/asl_feature_references.xlsx').read_bytes())


def score(y,pred):
    p,r,f,n=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
    return {'images':len(y),'correct':int((y==pred).sum()),'accuracy':float((y==pred).mean()),
            'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean())}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    assert digest('data/asl_kaggle/manifest.json')==meta['hashes']['manifest']
    assert digest('config/thresholds.json')==meta['hashes']['thresholds']
    ref=references()['A']
    paths=('research/feature_study.py','research/studies/feature_factorial.py','research/inputs.py',
           'research/geometry.py','research/workbook.py','research/references.py','research/shin.py',
           'research/studies/compare_models.py','research/classifiers.py')
    protocol={'version':VERSION,'variants':{v:{'blocks':list(b),'inputs':len(source_columns(v))} for v,b in VARIANTS.items()},
        'models':['svm','lightgbm'],'accepted_images':60251,'folds':5,
        'blocks':{'D':'190 within-image normalized distances','S':'630 axis-relative direction cosines',
                  'E':'70 one-hot values for 19 own categorical checks','J':'15 world-joint measurements'},
        'excluded':'26 workbook-match scores (source columns 890:916); all augmentation and hand canonicalization',
        'preprocessing':'D/S/E unchanged; J train-only median imputation plus standardization, same historical implementation.',
        'svm':{'C':10,'gamma':'scale','class_weight':'balanced','kernel':'rbf'},
        'lightgbm':{'n_estimators':100,'num_leaves':31,'learning_rate':.1,'min_child_samples':20,'seed':42,'threads':4},
        'permutation':{'seed':SEED,'repeats':REPEATS,'max_images_per_letter_per_fold':CAP,
            'selection':'One deterministic held-out subset shared by every arm/model. Classification CV always uses all held-out images.',
            'blocks':'Every present D/S/E/J block in every variant',
            'details':'19 categorical checks in DSE and DE; five E families and five J finger families in DSEJ and DEJ',
            'order':'Same row permutation for all columns of a group; same fold/repeat order across models/variants/groups.',
            'no_training':'Frozen estimator predicts altered validation inputs; labels unchanged; no refit or hyperparameter selection.',
            'report':'Mean fold/repeat macro-precision and macro-F1 drops, with descriptive spread; no independent confidence interval.'},
        'reference_keys':list(ref),'reference_file_sha256':digest('assets/asl_feature_references.xlsx'),
        'historical_input_hashes':meta['hashes'],
        'reference_note':'The current English workbook differs in bytes from the historical Dutch workbook. Only feature order/options are used here; no candidate-letter match scores. Raw-coordinate parity is checked before training.',
        'input_sha256':{str(CACHE/p):digest(CACHE/p) for p in ('inputs.npz','inputs_metadata.json','summary.json')},
        'source_sha256':{p:digest(p) for p in paths},
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','lightgbm')},
        'scope':'Exploratory study on repeatedly used filename-block folds, not verified participants; accepted detections only; no tuning.'}
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen factorial protocol changed; use a fresh directory.')
    atomic_json(path,protocol)
    return protocol


def inputs():
    x,y,ids,splits=load_inputs(CACHE)
    assert x.shape==(60251,931) and np.isfinite(x).all() and len(set(ids))==len(y)
    meta=json.loads((CACHE/'inputs_metadata.json').read_text())
    group=np.array([r['group'] for r in meta['samples']]);ids=np.asarray(ids)
    assignment=np.zeros(len(y),int)
    for fold,train,test in splits:
        assert not set(group[train])&set(group[test]) and not assignment[test].any()
        assignment[test]=fold
    assert (assignment>0).all()
    return x,y,ids,splits,group,assignment


def selected_rows(y,ids,splits):
    selections={}
    for fold,train,test in splits:
        rng=np.random.default_rng(SEED+fold)
        selected=np.sort(np.concatenate([rng.permutation(test[y[test]==label])[:CAP] for label in LABELS]))
        assert set(selected)<=set(test) and not set(selected)&set(train)
        selections[int(fold)]=selected
    return selections


def parity():
    x,y,ids,splits,groups,assignment=inputs();ref=references()
    thresholds=json.loads(Path('config/thresholds.json').read_text())
    samples={s['id']:s for s in json.loads(Path('data/asl_kaggle/manifest.json').read_text())['samples']}
    chosen=[]
    for fold in sorted(set(assignment)):
        for letter in LABELS:
            indices=np.flatnonzero((assignment==fold)&(y==letter))
            chosen.extend(indices[np.linspace(0,len(indices)-1,3,dtype=int)].tolist())
    category_max=joint_max=0.
    for i in chosen:
        sample=samples[ids[i]]
        assert sample['label']==y[i] and sample['status']=='ok'
        measured=measure_hand_features(sample['world_landmarks'],sample.get('handedness'))
        categories,_=encode_additions(measured,thresholds,ref)
        np.testing.assert_array_equal(categories.astype(np.float32),x[i,820:890])
        np.testing.assert_array_equal(encode_angles(measured).astype(np.float32),x[i,916:931])
        np.testing.assert_array_equal(np.array(list(combined_features(sample['image_landmarks']).values()),np.float32),x[i,:820])
    # Verify one-hot consistency for every row, using the active schema order.
    from research.feature_study import category_columns
    for columns in category_columns(ref['A']).values():
        values=x[:,columns]
        assert np.isin(values,[0,1]).all() and (values.sum(axis=1)==1).all()
    atomic_json(OUT/'input_verification.json',{'status':'passed','raw_samples_recomputed':len(chosen),
        'all_19_one_hot_groups_valid_on_images':len(y),'current_workbook_order_matches_cached_categories':True,
        'shin_categories_and_world_angles_exact_float32_parity':True})
    selections=selected_rows(y,ids,splits)
    write_csv(OUT/'permutation_selection.csv',[{'id':ids[i],'label':y[i],'fold':fold} for fold,ix in selections.items() for i in ix])
    print(f'Input parity passed: {len(chosen)} raw cases; all 60,251 categorical rows valid.',flush=True)


def train(name,variants,folds):
    x,y,ids,splits,groups,assignment=inputs()
    for variant in variants:
        data=matrix(x,variant)
        for fold,tr,te in splits:
            if fold not in folds:continue
            stem=f'{name}_{variant}_fold{fold}'
            path=WORK/f'{stem}.joblib';pred_path=WORK/f'{stem}.npz';meta_path=OUT/f'fit_{stem}.json'
            if pred_path.exists():
                with np.load(pred_path) as saved:np.testing.assert_array_equal(saved['ids'],ids[te])
                assert path.exists() and meta_path.exists()
                continue
            start=time.monotonic();print(f'Fit {name} {variant} fold {fold}, {len(tr)} sources, {data.shape[1]} inputs',flush=True)
            model=joblib.load(path) if path.exists() else estimator(name,variant,len(tr)).fit(data[tr],y[tr])
            if not path.exists():joblib.dump(model,path,compress=3)
            prediction=model.predict(data[te])
            np.savez_compressed(pred_path,ids=ids[te],prediction=prediction)
            row={'model':name,'variant':variant,'fold':int(fold),'inputs':data.shape[1],
                'training_images':len(tr),'test_images':len(te),'seconds':time.monotonic()-start,
                'training_ids_sha256':sha256('\n'.join(ids[tr]).encode()).hexdigest(),
                'test_ids_sha256':sha256('\n'.join(ids[te]).encode()).hexdigest(),
                'estimator_sha256':digest(path),**score(y[te],prediction)}
            atomic_json(meta_path,row);print(json.dumps(row),flush=True)
            del model


def permutation(name,variants,folds):
    x,y,ids,splits,groups,assignment=inputs();ref=references()['A'];selections=selected_rows(y,ids,splits)
    for variant in variants:
        data=matrix(x,variant);per_groups=permutation_groups(variant,ref)
        for fold,tr,te in splits:
            if fold not in folds:continue
            destination=OUT/f'permutation_{name}_{variant}_fold{fold}.csv'
            if destination.exists():continue
            selected=selections[int(fold)];values=data[selected];truth=y[selected]
            model=joblib.load(WORK/f'{name}_{variant}_fold{fold}.joblib')
            base=score(truth,model.predict(values));rows=[]
            for group,columns in per_groups.items():
                checkpoint=WORK/f'perm_{name}_{variant}_{fold}_{group}.json'
                if checkpoint.exists():current=json.loads(checkpoint.read_text())
                else:
                    current=[]
                    for repeat in range(REPEATS):
                        order=np.random.default_rng(SEED+10000*fold+repeat).permutation(len(values))
                        altered=permute_group(values,columns,order)
                        changed=score(truth,model.predict(altered))
                        current.append({'model':name,'variant':variant,'fold':int(fold),'group':group,'repeat':repeat,
                            'images':len(selected),'columns':len(columns),
                            'selected_ids_sha256':sha256('\n'.join(ids[selected]).encode()).hexdigest(),
                            'permutation_sha256':sha256(order.astype('<i8').tobytes()).hexdigest(),
                            **{'baseline_'+k:base[k] for k in ('macro_precision','macro_f1')},
                            **{'permuted_'+k:changed[k] for k in ('macro_precision','macro_f1')},
                            **{k+'_drop':base[k]-changed[k] for k in ('macro_precision','macro_f1')}})
                    atomic_json(checkpoint,current)
                rows.extend(current)
                print(f'Permuted {name} {variant} fold {fold}: {group}',flush=True)
            write_csv(destination,rows);del model


def export():
    import pandas as pd
    x,y,ids,splits,groups,assignment=inputs();metrics=[];letter_rows=[];pred_rows=[];paired=[]
    for name in ('svm','lightgbm'):
        baseline=None
        for variant in VARIANTS:
            pred=np.full(len(y),'',dtype='<U1')
            for fold,tr,te in splits:
                with np.load(WORK/f'{name}_{variant}_fold{fold}.npz') as saved:
                    np.testing.assert_array_equal(saved['ids'],ids[te]);pred[te]=saved['prediction']
            assert np.isin(pred,LABELS).all()
            if variant=='DS':baseline=pred.copy()
            for fold in ('pooled',*sorted(set(assignment))):
                keep=np.ones(len(y),bool) if fold=='pooled' else assignment==fold
                metrics.append({'model':name,'variant':variant,'fold':fold,'inputs':len(source_columns(variant)),**score(y[keep],pred[keep])})
                truth=y[keep];a=baseline[keep];b=pred[keep]
                fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
                paired.append({'model':name,'variant':variant,'fold':fold,'fixed':fixed,'lost':lost,'net':fixed-lost})
            p,r,f,n=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
            letter_rows.extend({'model':name,'variant':variant,'letter':l,'images':int(n[i]),
                'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i])} for i,l in enumerate(LABELS))
            pred_rows.extend({'model':name,'variant':variant,'id':ids[i],'label':y[i],'group':groups[i],
                'fold':int(assignment[i]),'prediction':pred[i]} for i in range(len(y)))
    write_csv(OUT/'metrics.csv',metrics);write_csv(OUT/'per_letter.csv',letter_rows)
    write_csv(OUT/'paired_changes.csv',paired);write_csv(OUT/'predictions.csv',pred_rows)
    per=pd.concat([pd.read_csv(OUT/f'permutation_{name}_{variant}_fold{fold}.csv',keep_default_na=False)
        for name in ('svm','lightgbm') for variant in VARIANTS for fold in range(1,6)],ignore_index=True)
    write_csv(OUT/'permutation.csv',per.to_dict('records'))
    summaries=[]
    for (name,variant,group),part in per.groupby(['model','variant','group'],sort=False):
        assert len(part)==25
        record={'model':name,'variant':variant,'group':group,'observations':len(part)}
        for metric in ('macro_precision','macro_f1'):
            values=part[metric+'_drop'].to_numpy()
            record[metric+'_mean_drop']=float(values.mean());record[metric+'_sd_fold_repeat']=float(values.std(ddof=1))
        summaries.append(record)
    write_csv(OUT/'permutation_summary.csv',summaries)
    atomic_json(OUT/'summary.json',{'images':len(y),'variants':len(VARIANTS),'models':2,'fold_estimators':80,
        'pooled_metrics':[r for r in metrics if r['fold']=='pooled'],
        'permutation_observations':len(per),'permutation_groups':len(summaries),
        'permutation_unique_images':sum(len(v) for v in selected_rows(y,ids,splits).values()),
        'scope':'All accepted original images for CV; fixed held-out capped subsets for permutation. No augmentation or workbook-match scores.'})
    print('Export complete.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('freeze','parity','train','permutation','run','export'),required=True)
    p.add_argument('--model',choices=('svm','lightgbm'),default='svm')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--folds',nargs='+',type=int,choices=range(1,6),default=list(range(1,6)))
    args=p.parse_args();freeze()
    if args.stage=='parity':parity()
    elif args.stage=='train':train(args.model,args.variants,args.folds)
    elif args.stage=='permutation':permutation(args.model,args.variants,args.folds)
    elif args.stage=='run':
        train(args.model,args.variants,args.folds)
        permutation(args.model,args.variants,args.folds)
    elif args.stage=='export':export()


if __name__=='__main__':main()
