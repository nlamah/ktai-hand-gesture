"""Equivalent cached-kernel execution of the frozen SVM permutation plan."""
from pathlib import Path
from hashlib import sha256
import json,time

import joblib
import numpy as np

from research.data import atomic_json
from research.feature_study import matrix,permutation_groups,permute_group
from research.svm_permutation import RBFPermutation
from research.studies.feature_factorial import OUT,WORK,SEED,inputs,selected_rows,references,score,digest


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():assert digest(path)==expected,path
    addendum={'parent_protocol_sha256':digest(OUT/'protocol.json'),
        'method':'Identical grouped permutation and frozen RBF SVC; float64 distance decomposition, LIBSVM coefficients and native fallback near pairwise margins.',
        'no_model_or_sampling_changes':True,
        'source_sha256':{p:digest(p) for p in ('research/svm_permutation.py','research/studies/fast_factorial_permutation.py')},
        'native_baseline_check':'Every selected prediction must equal the saved ordinary-SVC OOF prediction.',
        'native_perturbation_check':'52 evenly spread cases for first and last groups per fold, repeat 0; existing native checkpoints are reconciled when reused.'}
    target=OUT/'execution_backend.json'
    if target.exists():assert json.loads(target.read_text())==addendum
    atomic_json(target,addendum)
    x,y,ids,splits,groups,assignment=inputs();selection=selected_rows(y,ids,splits);ref=references()['A']
    checked=0;baseline_checked=0;computed=0;cached=0;fallback=0
    for variant in ('DSEJ','DSE','DEJ','DE','DSJ','DJ','DS','D'):
        data=matrix(x,variant);plan=permutation_groups(variant,ref)
        for fold,tr,te in splits:
            path=WORK/f'svm_{variant}_fold{fold}.joblib'
            assert digest(path)==json.loads((OUT/f'fit_svm_{variant}_fold{fold}.json').read_text())['estimator_sha256']
            model=joblib.load(path);selected=selection[int(fold)];values=data[selected];truth=y[selected]
            with np.load(WORK/f'svm_{variant}_fold{fold}.npz') as saved:
                np.testing.assert_array_equal(saved['ids'],ids[te]);lookup=dict(zip(te,saved['prediction']))
                expected=np.array([lookup[i] for i in selected])
            base=score(truth,expected)
            orders=[np.random.default_rng(SEED+10000*fold+repeat).permutation(len(values)) for repeat in range(5)]
            alternate = ('precomputed_perm_directions' if variant=='DSEJ' else
                         'precomputed_perm_no_directions' if variant=='DEJ' else None)
            with RBFPermutation(model,values) as fast:
                np.testing.assert_array_equal(fast.baseline(),expected);baseline_checked+=len(expected)
                keys=list(plan)
                for group,columns in plan.items():
                    checkpoint=WORK/f'perm_svm_{variant}_{fold}_{group}.json'
                    alt=WORK/alternate/checkpoint.name if alternate else None
                    previous=checkpoint if checkpoint.exists() else alt if alt and alt.exists() else None
                    if previous:
                        rows=json.loads(previous.read_text());assert len(rows)==5
                        for row in rows:
                            assert row['variant']==variant and row['fold']==fold and row['group']==group
                            np.testing.assert_allclose([row['baseline_macro_precision'],row['baseline_macro_f1']],
                                                       [base['macro_precision'],base['macro_f1']],atol=1e-12,rtol=0)
                        cached+=5
                    else:
                        predictions=fast.group_predictions(columns,orders);rows=[]
                        if group in (keys[0],keys[-1]):
                            sample=np.linspace(0,len(values)-1,52,dtype=int)
                            native=model.predict(permute_group(values,columns,orders[0])[sample])
                            np.testing.assert_array_equal(predictions[0][sample],native);checked+=len(sample)
                        for repeat,pred in enumerate(predictions):
                            changed=score(truth,pred)
                            rows.append({'model':'svm','variant':variant,'fold':int(fold),'group':group,'repeat':repeat,
                                'images':len(selected),'columns':len(columns),
                                'selected_ids_sha256':sha256('\n'.join(ids[selected]).encode()).hexdigest(),
                                'permutation_sha256':sha256(orders[repeat].astype('<i8').tobytes()).hexdigest(),
                                **{'baseline_'+k:base[k] for k in ('macro_precision','macro_f1')},
                                **{'permuted_'+k:changed[k] for k in ('macro_precision','macro_f1')},
                                **{k+'_drop':base[k]-changed[k] for k in ('macro_precision','macro_f1')}})
                        computed+=5
                    # Atomic whole-group checkpoints can be resumed by the ordinary runner.
                    atomic_json(checkpoint,rows)
                    if alt:atomic_json(alt,rows)
                    print(f'Cached exact SVM permutation {variant} fold {fold}: {group}',flush=True)
                fallback+=fast.fallback_rows
            del model
    atomic_json(OUT/'kernel_verification.json',{'status':'passed','selected_native_baseline_predictions_checked':baseline_checked,
        'native_perturbed_cases_checked':checked,'computed_permutation_rows':computed,'reused_native_checkpoint_rows':cached,
        'near_margin_native_fallback_rows':fallback,'execution_backend_sha256':digest(target),
        'scope':'Same inference function and frozen protocol, accelerated through cached float64 RBF distance blocks.'})
    print('FAST_KERNEL_EXECUTION_COMPLETE',flush=True)


if __name__=='__main__':main()
