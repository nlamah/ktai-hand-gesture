"""Isolate continuous joint angles and Excel scores on frozen Kaggle folds."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support

from research.studies.compare_models import load_inputs,model_for,scores
from research.inputs import angle_estimator,write_csv,LABELS

VARIANTS=('shin','shin_angles','shin_excel','shin_angles_excel')


def variant_inputs(X,variant):
    if variant=='shin':return X[:,:820]
    if variant=='shin_angles':return np.c_[X[:,:820],X[:,916:931]]
    if variant=='shin_excel':return np.c_[X[:,:820],X[:,890:916]]
    if variant=='shin_angles_excel':return np.c_[X[:,:820],X[:,890:916],X[:,916:931]]
    raise ValueError(variant)


def classifier(name,variant,n_train):
    if 'angles' not in variant:return model_for(name,False,n_train)
    width=835 if variant=='shin_angles' else 861
    preprocessing=angle_estimator(width-15,width).steps[0][1]
    return make_pipeline(preprocessing,model_for(name,False,n_train))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=('svm','lightgbm'),default=['svm','lightgbm'])
    parser.add_argument('--run',type=Path,default=Path('data/asl_kaggle/research'))
    parser.add_argument('--baseline-run',type=Path,default=Path('results/model_comparison'))
    parser.add_argument('--out',type=Path,default=Path('results/angles'))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    X,y,ids,splits=load_inputs(args.run)
    frozen=json.loads((args.run/'inputs_metadata.json').read_text())
    old=json.loads((args.baseline_run/'inputs.json').read_text())
    if old['input_hashes']!=frozen['hashes']:raise ValueError('Baseline uses different feature inputs')
    provenance={'input_hashes':frozen['hashes'],'run':str(args.run.resolve()),'baseline_run':str(args.baseline_run.resolve()),
                'folds_sha256':sha256((args.run/'summary.json').read_bytes()).hexdigest(),
                'ids_sha256':sha256('\n'.join(ids).encode()).hexdigest(),'variants':list(VARIANTS)}
    configfile=out/'inputs.json'
    if configfile.exists() and json.loads(configfile.read_text())!=provenance:raise ValueError('Output belongs to other inputs')
    if not configfile.exists():configfile.write_text(json.dumps(provenance,indent=2))
    with (args.baseline_run/'oof_predictions.csv').open(encoding='utf-8-sig') as handle:
        baseline={(r['model'],r['id']):r for r in csv.DictReader(handle) if r['variant']=='shin'}
    for name in args.models:
        rows=[];letter_rows=[];pred_rows=[]
        for variant in VARIANTS:
            matrix=variant_inputs(X,variant);predicted=np.full(len(y),'',dtype='<U1')
            for fold,train,test in splits:
                if variant=='shin':
                    pred=np.array([baseline[name,ids[i]]['prediction'] for i in test])
                    assert all(baseline[name,ids[i]]['label']==y[i] for i in test)
                else:
                    model_path=out/f'{name}_{variant}_fold{fold}.joblib'
                    checkpoint=out/f'{name}_{variant}_fold{fold}_predictions.npz'
                    if checkpoint.exists():
                        cached=np.load(checkpoint)
                        if cached['ids'].tolist()!=[ids[i] for i in test]:raise ValueError('Prediction checkpoint mismatch')
                        pred=cached['predictions']
                    else:
                        if model_path.exists():model=joblib.load(model_path)
                        else:
                            model=classifier(name,variant,len(train));model.fit(matrix[train],y[train]);joblib.dump(model,model_path)
                        pred=model.predict(matrix[test])
                        np.savez_compressed(checkpoint,ids=np.array(ids)[test],predictions=pred)
                predicted[test]=pred
                row={'model':name,'variant':variant,'fold':fold,'features':matrix.shape[1],'images':len(test),**scores(y[test],pred)}
                rows.append(row);print(json.dumps(row),flush=True)
            if np.any(predicted==''):raise ValueError('Missing OOF predictions')
            pooled={'model':name,'variant':variant,'fold':'pooled','features':matrix.shape[1],'images':len(y),**scores(y,predicted)}
            rows.append(pooled);print(json.dumps(pooled),flush=True)
            precision,recall,f1,support=precision_recall_fscore_support(y,predicted,labels=LABELS,zero_division=0)
            for i,label in enumerate(LABELS):letter_rows.append({'model':name,'variant':variant,'letter':label,'images':int(support[i]),
                'correct':int(((y==label)&(predicted==label)).sum()),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])})
            fold_by_index={i:fold for fold,train,test in splits for i in test}
            pred_rows.extend({'model':name,'variant':variant,'id':ids[i],'fold':fold_by_index[i],
                              'label':str(y[i]),'prediction':str(predicted[i])} for i in range(len(y)))
        write_csv(out/f'metrics_{name}.csv',rows);write_csv(out/f'per_letter_{name}.csv',letter_rows)
        write_csv(out/f'oof_predictions_{name}.csv',pred_rows)
    print(f'Completed: {out.resolve()}',flush=True)


if __name__=='__main__':main()
