"""Independent score, historical parity, preprocessing and permutation audit."""
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix

from research.data import atomic_json
from research.feature_study import VARIANTS,matrix,source_columns,permutation_groups,permute_group
from research.inputs import LABELS,write_csv
from research.studies.feature_factorial import OUT,WORK,inputs,references,selected_rows,SEED,digest


def check_scores(truth,pred,row):
    p,r,f,n=precision_recall_fscore_support(truth,pred,labels=LABELS,zero_division=0)
    measured={'images':len(truth),'correct':int((truth==pred).sum()),'accuracy':float((truth==pred).mean()),
              'macro_precision':p.mean(),'macro_recall':r.mean(),'macro_f1':f.mean()}
    for key,value in measured.items():np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=0)
    return p,r,f,n


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for group in ('input_sha256','source_sha256'):
        for path,expected in protocol[group].items():assert digest(path)==expected,path
    assert digest('assets/asl_feature_references.xlsx')==protocol['reference_file_sha256']
    assert json.loads((OUT/'input_verification.json').read_text())['status']=='passed'
    backend=None
    if (OUT/'execution_backend.json').exists():
        backend=json.loads((OUT/'execution_backend.json').read_text())
        assert backend['parent_protocol_sha256']==digest(OUT/'protocol.json')
        for path,expected in backend['source_sha256'].items():assert digest(path)==expected,path
        kernel=json.loads((OUT/'kernel_verification.json').read_text())
        assert kernel['status']=='passed'
        assert kernel['selected_native_baseline_predictions_checked']==104000
        assert kernel['computed_permutation_rows']+kernel['reused_native_checkpoint_rows']==1950
    x,y,ids,splits,groups,assignment=inputs();ref=references()['A'];selections=selected_rows(y,ids,splits)
    frame=pd.read_csv(OUT/'predictions.csv',keep_default_na=False)
    metrics=pd.read_csv(OUT/'metrics.csv',keep_default_na=False)
    letters=pd.read_csv(OUT/'per_letter.csv',keep_default_na=False)
    paired=pd.read_csv(OUT/'paired_changes.csv',keep_default_na=False)
    per=pd.read_csv(OUT/'permutation.csv',keep_default_na=False)
    summary=pd.read_csv(OUT/'permutation_summary.csv',keep_default_na=False)
    assert len(frame)==16*60251 and not frame.duplicated(['model','variant','id']).any()
    assert len(per)==3900 and len(summary)==156
    assert per.groupby(['fold','repeat']).permutation_sha256.nunique().eq(1).all()
    assert per.groupby('fold').selected_ids_sha256.nunique().eq(1).all()
    metric_count=replayed=permutation_replayed=scalers=0;confusions=[]
    for name in ('svm','lightgbm'):
        old=pd.read_csv(f'results/angles/oof_predictions_{name}.csv',keep_default_na=False)
        base=None
        for variant in VARIANTS:
            current=frame[(frame.model==name)&(frame.variant==variant)].set_index('id').loc[ids]
            np.testing.assert_array_equal(current.label,y);np.testing.assert_array_equal(current.fold,assignment)
            pred=current.prediction.to_numpy()
            if variant=='DS':base=pred
            if variant in ('DS','DSJ'):
                previous=old[old.variant==('shin' if variant=='DS' else 'shin_angles')].set_index('id').loc[ids]
                np.testing.assert_array_equal(previous.label,y);np.testing.assert_array_equal(previous.fold,assignment)
                np.testing.assert_array_equal(previous.prediction,pred)
            for fold in ('pooled',*sorted(set(assignment))):
                keep=np.ones(len(y),bool) if fold=='pooled' else assignment==fold
                row=metrics[(metrics.model==name)&(metrics.variant==variant)&(metrics.fold==str(fold))].iloc[0]
                detail=check_scores(y[keep],pred[keep],row);metric_count+=1
                changes=paired[(paired.model==name)&(paired.variant==variant)&(paired.fold==str(fold))].iloc[0]
                a,b=base[keep],pred[keep];truth=y[keep]
                fixed=int(((a!=truth)&(b==truth)).sum());lost=int(((a==truth)&(b!=truth)).sum())
                np.testing.assert_array_equal(changes[['fixed','lost','net']].to_numpy(),[fixed,lost,fixed-lost])
                if fold=='pooled':
                    stored=letters[(letters.model==name)&(letters.variant==variant)].set_index('letter').loc[LABELS]
                    for key,value in zip(('precision','recall','f1','images'),detail):
                        np.testing.assert_allclose(stored[key],value,atol=1e-12,rtol=0)
            cm=confusion_matrix(y,pred,labels=LABELS)
            confusions.extend({'model':name,'variant':variant,'true_letter':label,**dict(zip(LABELS,map(int,cm[j])))}
                              for j,label in enumerate(LABELS))
            data=matrix(x,variant);plan=permutation_groups(variant,ref)
            for fold,tr,te in splits:
                stem=f'{name}_{variant}_fold{fold}'
                fit=json.loads((OUT/f'fit_{stem}.json').read_text());path=WORK/f'{stem}.joblib'
                assert fit['estimator_sha256']==digest(path)
                assert fit['training_ids_sha256']==sha256('\n'.join(ids[tr]).encode()).hexdigest()
                assert fit['test_ids_sha256']==sha256('\n'.join(ids[te]).encode()).hexdigest()
                assert not set(groups[tr])&set(groups[te])
                model=joblib.load(path)
                assert model.n_features_in_==data.shape[1] and list(model.classes_)==LABELS
                if 'J' in variant:
                    scaler=model[0].named_transformers_['angles'].named_steps['standardscaler']
                    np.testing.assert_allclose(scaler.mean_,x[tr,916:931].mean(axis=0,dtype=np.float64),atol=1e-10,rtol=0)
                    sample=te[np.linspace(0,len(te)-1,26,dtype=int)]
                    np.testing.assert_array_equal(model[0].transform(data[sample])[:,:-15],data[sample,:-15])
                    scalers+=1
                else:sample=te[np.linspace(0,len(te)-1,26,dtype=int)]
                np.testing.assert_array_equal(model.predict(data[sample]),pred[sample]);replayed+=len(sample)
                records=per[(per.model==name)&(per.variant==variant)&(per.fold==fold)]
                assert len(records)==5*len(plan) and set(records.group)==set(plan)
                selected=selections[int(fold)]
                for repeat in range(5):
                    order=np.random.default_rng(SEED+10000*fold+repeat).permutation(len(selected))
                    selected_records=records[records.repeat==repeat]
                    assert selected_records.permutation_sha256.eq(sha256(order.astype('<i8').tobytes()).hexdigest()).all()
                for group,columns in plan.items():assert records[records.group==group]['columns'].eq(len(columns)).all()
                # Replay two interventions on fold 1 for every model/arm, including a fine group where available.
                if fold==1:
                    values=data[selected];truth=y[selected]
                    baseline_pred=model.predict(values)
                    p0,_,f0,_=precision_recall_fscore_support(truth,baseline_pred,labels=LABELS,zero_division=0)
                    keys=list(plan);chosen=list(dict.fromkeys([keys[0],keys[-1]]))
                    for group in chosen:
                        order=np.random.default_rng(SEED+10000*fold).permutation(len(selected))
                        altered=permute_group(values,plan[group],order)
                        result=model.predict(altered)
                        p1,_,f1,_=precision_recall_fscore_support(truth,result,labels=LABELS,zero_division=0)
                        row=records[(records.group==group)&(records.repeat==0)].iloc[0]
                        np.testing.assert_allclose([p0.mean(),f0.mean(),p1.mean(),f1.mean()],
                            [row.baseline_macro_precision,row.baseline_macro_f1,row.permuted_macro_precision,row.permuted_macro_f1],atol=1e-12,rtol=0)
                        permutation_replayed+=1
                del model
    for metric in ('macro_precision','macro_f1'):
        np.testing.assert_allclose(per[metric+'_drop'],per['baseline_'+metric]-per['permuted_'+metric],atol=1e-12,rtol=0)
    for row in summary.to_dict('records'):
        part=per[(per.model==row['model'])&(per.variant==row['variant'])&(per.group==row['group'])]
        assert len(part)==25
        for metric in ('macro_precision','macro_f1'):
            values=part[metric+'_drop']
            np.testing.assert_allclose([values.mean(),values.std(ddof=1)],
                [row[metric+'_mean_drop'],row[metric+'_sd_fold_repeat']],atol=1e-12,rtol=0)
    write_csv(OUT/'confusions.csv',confusions)
    atomic_json(OUT/'verification.json',{'status':'passed','models_checked':80,'metric_rows_recomputed':metric_count,
        'paired_rows_recomputed':len(paired),'letter_rows_recomputed':len(letters),
        'held_out_predictions_replayed':replayed,'train_only_scalers_checked':scalers,
        'permutation_interventions_replayed':permutation_replayed,'permutation_observations_checked':len(per),
        'historical_DS_and_DSJ_predictions_identical':True,'shared_held_out_subset_and_seed_orders_verified':True,
        'equivalent_RBF_execution_addendum_checked':backend is not None,
        'results_sha256':{p:digest(OUT/p) for p in ('protocol.json','metrics.csv','per_letter.csv','paired_changes.csv',
            'predictions.csv','permutation.csv','permutation_summary.csv','confusions.csv')},
        'scope':'Computational verification; filename folds and permutations do not establish causal feature value or unseen-person performance.'})
    print(json.dumps(json.loads((OUT/'verification.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
