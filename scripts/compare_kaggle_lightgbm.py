"""Run saved Kaggle folds with SVM/LightGBM, grouped permutations and ablations."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from scripts.cross_validate_shin_workbook import category_options,write_csv,LABELS,angle_estimator,ANGLE_COLUMNS
from gesturelab.features.asl_excel_review import load_reference_workbook
from gesturelab.training.classifiers import classifier_parameters


def feature_groups(references):
    groups={'all_extra':list(range(820,931)),'categories':list(range(820,890)),
            'angles':list(range(916,931)),'excel_scores':list(range(890,916)),
            'thumb':[],'finger_flexion':[],'spread':[],'crossing':[],'contact':[]}
    offset=820
    for feature in references['A']:
        indices=list(range(offset,offset+len(category_options(feature))));offset+=len(indices)
        family=('thumb' if feature in ('thumb_mcp_shape','flexion_thumb_ip') else
                'finger_flexion' if feature.startswith('flexion_') else
                'spread' if feature.startswith('spread_') else
                'crossing' if feature.startswith('crossing_') else 'contact')
        groups[family]+=indices
    for i,name in enumerate(ANGLE_COLUMNS):
        groups['thumb' if name.startswith('thumb_') else 'finger_flexion'].append(916+i)
    assert offset==890
    return groups


def model_for(name,combined,n_train):
    if name=='svm':
        return angle_estimator(916,931) if combined else SVC(C=10,gamma='scale',class_weight='balanced',cache_size=512)
    from lightgbm import LGBMClassifier
    parameters=classifier_parameters('lightgbm',profile='project',n_train=n_train,seed=42)
    parameters['n_jobs']=4
    clf=LGBMClassifier(**parameters)
    return make_pipeline(angle_estimator(916,931).steps[0][1],clf) if combined else clf


def scores(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}


def load_inputs(run):
    summary=json.loads((run/'summary.json').read_text());meta=json.loads((run/'inputs_metadata.json').read_text())
    arrays=np.load(run/'inputs.npz');X=np.c_[arrays['X'],arrays['angles']]
    y=np.array([r['label'] for r in meta['samples']]);ids=[r['id'] for r in meta['samples']]
    group=np.array([r['group'] for r in meta['samples']])
    splits=[]
    for fold in summary['folds']:
        train=np.flatnonzero(np.isin(group,fold['train_groups']));test=np.flatnonzero(np.isin(group,fold['test_groups']))
        assert not set(group[train])&set(group[test])
        splits.append((fold['fold'],train,test))
    return X,y,ids,splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('train','permutation','ablation'),required=True)
    parser.add_argument('--run',type=Path,default=Path('experiments/kaggle_nested_feature_weights_01'))
    parser.add_argument('--out',type=Path,default=Path('experiments/kaggle_lightgbm_importance_01'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/asl_slash_proposal_20261031/ASL_featurevoorstel.xlsx'))
    parser.add_argument('--models',nargs='+',choices=('svm','lightgbm'),default=['svm','lightgbm'])
    parser.add_argument('--groups',nargs='+',default=['all_extra','categories','angles','excel_scores','thumb','finger_flexion','spread','crossing','contact'])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--max-per-class',type=int,default=100,help='Permutation validation cap per class/fold; 0 uses all validation images.')
    args=parser.parse_args()
    if args.repeats<1 or args.max_per_class<0:parser.error('repeats >= 1 and max-per-class >= 0 required')
    args.out.mkdir(parents=True,exist_ok=True)
    refs=load_reference_workbook(args.reference.read_bytes());groups=feature_groups(refs)
    if set(args.groups)-set(groups):parser.error('Unknown groups; choose '+', '.join(groups))
    X,y,ids,splits=load_inputs(args.run)
    config={'run':str(args.run.resolve()),'reference':str(args.reference.resolve()),'rows':len(y),
            'input_hashes':json.loads((args.run/'inputs_metadata.json').read_text())['hashes']}
    configfile=args.out/'inputs.json'
    if configfile.exists() and json.loads(configfile.read_text())!=config:raise ValueError('Output belongs to different inputs')
    if not configfile.exists():
        configfile.write_text(json.dumps(config,indent=2))
    if args.stage=='train':
        metrics=[];per_letter=[];oof=[]
        for name in args.models:
            for variant in ('shin','combined'):
                predictions=np.full(len(y),'',dtype='<U1')
                for fold,train,test in splits:
                    target=args.out/f'{name}_{variant}_fold{fold}.joblib'
                    width=931 if variant=='combined' else 820
                    if target.exists():model=joblib.load(target)
                    else:
                        model=model_for(name,variant=='combined',len(train));model.fit(X[train,:width],y[train]);joblib.dump(model,target)
                    pred=model.predict(X[test,:width]);predictions[test]=pred
                    row={'model':name,'variant':variant,'fold':fold,'images':len(test),**scores(y[test],pred)}
                    metrics.append(row);print(json.dumps(row),flush=True)
                pooled={'model':name,'variant':variant,'fold':'pooled','images':len(y),**scores(y,predictions)}
                metrics.append(pooled);print(json.dumps(pooled),flush=True)
                recalls=recall_score(y,predictions,labels=LABELS,average=None,zero_division=0)
                per_letter.extend({'model':name,'variant':variant,'letter':label,'images':int((y==label).sum()),'recall':float(value)} for label,value in zip(LABELS,recalls))
                oof.extend({'model':name,'variant':variant,'id':key,'label':str(truth),'prediction':str(pred)} for key,truth,pred in zip(ids,y,predictions))
        write_csv(args.out/'model_metrics.csv',metrics);write_csv(args.out/'per_letter.csv',per_letter);write_csv(args.out/'oof_predictions.csv',oof)
    elif args.stage=='permutation':
        rows=[];letters=[]
        for name in args.models:
            for fold,train,test in splits:
                model=joblib.load(args.out/f'{name}_combined_fold{fold}.joblib')
                rng=np.random.default_rng(4200+fold)
                selected=(np.sort(np.concatenate([rng.permutation(test[y[test]==label])[:args.max_per_class] for label in LABELS]))
                          if args.max_per_class else test)
                validation=X[selected].copy();truth=y[selected];base_pred=model.predict(validation);base=scores(truth,base_pred)
                base_recall=recall_score(truth,base_pred,labels=LABELS,average=None,zero_division=0)
                for group in args.groups:
                    for repeat in range(args.repeats):
                        # Same row permutation for every column of the feature group.
                        # Same seed per model permits paired SVM/LightGBM comparisons.
                        order=np.random.default_rng(42+10000*fold+repeat).permutation(len(selected))
                        altered=validation.copy();indices=groups[group];altered[:,indices]=validation[order][:,indices]
                        pred=model.predict(altered);changed=scores(truth,pred)
                        rows.append({'model':name,'fold':fold,'group':group,'repeat':repeat,'images':len(selected),
                                     'baseline_accuracy':base['accuracy'],'baseline_macro_f1':base['macro_f1'],
                                     'accuracy_drop':base['accuracy']-changed['accuracy'],'macro_f1_drop':base['macro_f1']-changed['macro_f1']})
                        recall=recall_score(truth,pred,labels=LABELS,average=None,zero_division=0)
                        letters.extend({'model':name,'fold':fold,'group':group,'repeat':repeat,'letter':label,
                                        'images':int((truth==label).sum()),'baseline_recall':float(base_recall[i]),
                                        'permuted_recall':float(recall[i]),'recall_drop':float(base_recall[i]-recall[i])} for i,label in enumerate(LABELS))
                    print(f'Permutation {name} fold {fold} group {group}',flush=True)
        suffix=f'r{args.repeats}_cap{args.max_per_class}'
        if len(args.models)==1:suffix+='_'+args.models[0]
        write_csv(args.out/f'permutation_{suffix}.csv',rows);write_csv(args.out/f'permutation_letters_{suffix}.csv',letters)
        aggregates=[]
        for name in args.models:
            for group in args.groups:
                selected=[r for r in rows if r['model']==name and r['group']==group]
                for metric in ('accuracy_drop','macro_f1_drop'):
                    values=[r[metric] for r in selected]
                    aggregates.append({'model':name,'group':group,'metric':metric,'mean_drop':float(np.mean(values)),
                                       'sd_fold_repeat':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'observations':len(values)})
        write_csv(args.out/f'permutation_summary_{suffix}.csv',aggregates)
    else:
        rows=[];letters=[]
        for name in args.models:
            for group in args.groups:
                keep=np.array([i for i in range(931) if i not in groups[group]])
                for fold,train,test in splits:
                    base=joblib.load(args.out/f'{name}_combined_fold{fold}.joblib');base_pred=base.predict(X[test]);base_scores=scores(y[test],base_pred)
                    # Fit the original preprocessing only on training data; then
                    # drop original columns. The angle pipeline retains column order.
                    preprocess=angle_estimator(916,931).steps[0][1]
                    fit=preprocess.fit_transform(X[train]);validation=preprocess.transform(X[test])
                    indicators=preprocess.named_transformers_['angles'].steps[0][1].indicator_.features_
                    mapping=list(range(931))+[916+int(i) for i in indicators]
                    if fit.shape[1]!=len(mapping):raise ValueError('Unexpected preprocessing column mapping')
                    kept_transformed=[i for i,source in enumerate(mapping) if source not in groups[group]]
                    model=model_for(name,False,len(train));model.fit(fit[:,kept_transformed],y[train]);pred=model.predict(validation[:,kept_transformed])
                    current=scores(y[test],pred)
                    rows.append({'model':name,'group_removed':group,'fold':fold,'images':len(test),**current,
                                 'accuracy_drop':base_scores['accuracy']-current['accuracy'],
                                 'macro_f1_drop':base_scores['macro_f1']-current['macro_f1']})
                    before=recall_score(y[test],base_pred,labels=LABELS,average=None,zero_division=0)
                    after=recall_score(y[test],pred,labels=LABELS,average=None,zero_division=0)
                    letters.extend({'model':name,'group_removed':group,'fold':fold,'letter':label,'images':int((y[test]==label).sum()),'baseline_recall':float(before[i]),
                                        'ablated_recall':float(after[i]),'recall_drop':float(before[i]-after[i])} for i,label in enumerate(LABELS))
                    print(f'Ablation {name} fold {fold} without {group}',flush=True)
        suffix='_'+args.models[0] if len(args.models)==1 else ''
        write_csv(args.out/f'ablation{suffix}.csv',rows);write_csv(args.out/f'ablation_letters{suffix}.csv',letters)
    print(f'Results: {args.out.resolve()}',flush=True)


if __name__=='__main__':main()
