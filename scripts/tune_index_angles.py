"""Nested grouped tuning of index MCP/PIP/DIP weights for Shin + angles SVM."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_kaggle_lightgbm import load_inputs,scores
from scripts.cross_validate_shin_workbook import ANGLE_COLUMNS,write_csv,LABELS
from scripts.tune_kaggle_feature_weights import scale_angles,bounded_indices,svm

WEIGHTS=(.5,1.,2.)
JOINTS=('index_mcp_angle_deg','index_pip_angle_deg','index_dip_angle_deg')
ARMS=('unweighted','pip_2','tuned_pip','tuned_index')


def weighted_input(shin,angles,weights):
    result=angles.copy()
    for joint,weight in zip(JOINTS,weights):result[:,ANGLE_COLUMNS.index(joint)]*=weight
    return np.c_[shin,result]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('experiments/kaggle_nested_feature_weights_01'))
    parser.add_argument('--baseline',type=Path,default=Path('experiments/kaggle_angles_scores_5fold_01'))
    parser.add_argument('--out',type=Path,default=Path('experiments/kaggle_index_weights_5fold_01'))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed run exists')
    raw,y,ids,splits=load_inputs(args.run);shin=raw[:,:820].copy();angles=raw[:,916:931].copy();del raw
    metadata=json.loads((args.run/'inputs_metadata.json').read_text());groups=np.array([r['group'] for r in metadata['samples']])
    with (args.baseline/'oof_predictions_svm.csv').open(encoding='utf-8-sig') as handle:
        baseline={r['id']:r for r in csv.DictReader(handle) if r['variant']=='shin_angles'}
    if set(baseline)!=set(ids):raise ValueError('Baseline IDs differ')
    config={'input_hashes':metadata['hashes'],'folds_sha256':sha256((args.run/'summary.json').read_bytes()).hexdigest(),
            'baseline_sha256':sha256((args.baseline/'oof_predictions_svm.csv').read_bytes()).hexdigest(),
            'weights':list(WEIGHTS),'joints':list(JOINTS),'inner_per_class':150}
    configfile=out/'inputs.json'
    if configfile.exists() and json.loads(configfile.read_text())!=config:raise ValueError('Checkpoint provenance mismatch')
    if not configfile.exists():configfile.write_text(json.dumps(config,indent=2))
    predictions={arm:np.full(len(y),'',dtype='<U1') for arm in ARMS};assignments=np.zeros(len(y),dtype=int)
    fold_rows=[];search_rows=[];fold_details=[]
    for fold,train,test in splits:
        assignments[test]=fold;checkpoint=out/f'fold_{fold}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if saved['test_ids']!=[ids[i] for i in test]:raise ValueError('Outer checkpoint IDs differ')
            for arm in ARMS:predictions[arm][test]=saved['predictions'][arm]
            fold_rows+=saved['metrics'];search_rows+=saved['search'];fold_details.append(saved['details'])
            print(f'Restored outer fold {fold}',flush=True);continue
        parts=[];inner_splits=[]
        for inner_fold,(a,b) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=420+fold).split(shin[train],y[train],groups[train]),1):
            fit=bounded_indices(train[a],y,10000+100*fold+inner_fold)
            validate=bounded_indices(train[b],y,20000+100*fold+inner_fold)
            assert not set(groups[fit])&set(groups[validate]) and not set(groups[fit])&set(groups[test])
            if set(y[fit])!=set(LABELS):raise ValueError('Missing training class')
            fa,va=scale_angles(angles[fit],angles[validate]);parts.append((fit,validate,fa,va))
            inner_splits.append({'fold':inner_fold,'fit_ids':[ids[i] for i in fit],'validation_ids':[ids[i] for i in validate],
                                 'fit_groups':sorted(set(groups[fit])),'validation_groups':sorted(set(groups[validate]))})
        searchfile=out/f'inner_search_fold_{fold}.json'
        local=json.loads(searchfile.read_text()) if searchfile.exists() else []
        memo={}
        def evaluate(weights):
            weights=tuple(weights)
            if weights in memo:return memo[weights]
            rows=[r for r in local if tuple(r['weights'])==weights]
            if len(rows)!=3:
                rows=[]
                for inner_fold,(fit,validate,fa,va) in enumerate(parts,1):
                    model=svm();model.fit(weighted_input(shin[fit],fa,weights),y[fit])
                    pred=model.predict(weighted_input(shin[validate],va,weights))
                    row={'outer_fold':fold,'inner_fold':inner_fold,'weights':list(weights),**scores(y[validate],pred)}
                    rows.append(row)
                local[:]=[r for r in local if tuple(r['weights'])!=weights]+rows
                searchfile.write_text(json.dumps(local))
            objective=(float(np.mean([r['macro_f1'] for r in rows])),float(np.mean([r['accuracy'] for r in rows])))
            memo[weights]=objective;print(f'Outer {fold}: weights MCP/PIP/DIP={weights}, inner F1={objective[0]:.5f}',flush=True)
            return objective
        best=(1.,1.,1.);objective=evaluate(best)
        pip_best=best
        for joint in (1,0,2):
            selected=best;selected_score=objective
            for weight in WEIGHTS:
                candidate=list(best);candidate[joint]=weight;candidate=tuple(candidate)
                result=evaluate(candidate)
                if result>selected_score:selected,selected_score=candidate,result
            best,objective=selected,selected_score
            if joint==1:pip_best=best
        selected={'unweighted':(1.,1.,1.),'pip_2':(1.,2.,1.),'tuned_pip':pip_best,'tuned_index':best}
        fa,va=scale_angles(angles[train],angles[test]);outer_cache={};metrics=[]
        for arm,weights in selected.items():
            if weights==(1.,1.,1.):
                pred=np.array([baseline[ids[i]]['prediction'] for i in test])
                assert all(baseline[ids[i]]['label']==y[i] and int(baseline[ids[i]]['fold'])==fold for i in test)
            elif weights in outer_cache:pred=outer_cache[weights]
            else:
                tag='_'.join(str(w).replace('.','p') for w in weights);path=out/f'outer_{fold}_weights_{tag}.joblib'
                if path.exists():model=joblib.load(path)
                else:
                    model=svm();model.fit(weighted_input(shin[train],fa,weights),y[train]);joblib.dump(model,path)
                pred=model.predict(weighted_input(shin[test],va,weights));outer_cache[weights]=pred
            predictions[arm][test]=pred
            row={'fold':fold,'variant':arm,'images':len(test),'mcp':weights[0],'pip':weights[1],'dip':weights[2],**scores(y[test],pred)}
            metrics.append(row);print(json.dumps(row),flush=True)
        details={'fold':fold,'train_groups':sorted(set(groups[train])),'test_groups':sorted(set(groups[test])),
                 'selected_weights':{k:list(v) for k,v in selected.items()},'inner_splits':inner_splits}
        saved={'test_ids':[ids[i] for i in test],'predictions':{arm:predictions[arm][test].tolist() for arm in ARMS},
               'metrics':metrics,'search':local,'details':details}
        checkpoint.write_text(json.dumps(saved));fold_rows+=metrics;search_rows+=local;fold_details.append(details)
    summaries={};letters=[];confusions=[];paired={}
    for arm in ARMS:
        pred=predictions[arm]
        if np.any(pred==''):raise ValueError('Incomplete OOF predictions')
        summaries[arm]={**scores(y,pred),'correct':int((y==pred).sum())}
        precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
        for i,label in enumerate(LABELS):letters.append({'variant':arm,'letter':label,'images':int(support[i]),
            'correct':int(((y==label)&(pred==label)).sum()),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])})
        matrix=confusion_matrix(y,pred,labels=LABELS)
        write_csv(out/f'confusion_{arm}.csv',[{'true_letter':label,**dict(zip(LABELS,map(int,row)))} for label,row in zip(LABELS,matrix)])
        for true,predicted in [('O','D'),('D','O'),('M','N'),('N','M'),('R','U'),('U','R')]:
            confusions.append({'variant':arm,'true_letter':true,'predicted_letter':predicted,'count':int(matrix[LABELS.index(true),LABELS.index(predicted)])})
        base=predictions['unweighted']==y;correct=pred==y
        paired[arm]={'recovered':int((~base&correct).sum()),'lost':int((base&~correct).sum())}
    write_csv(out/'fold_metrics.csv',fold_rows);write_csv(out/'per_letter.csv',letters);write_csv(out/'target_confusions.csv',confusions)
    write_csv(out/'oof_predictions.csv',[{'id':ids[i],'label':str(y[i]),'fold':int(assignments[i]),
               **{arm:str(predictions[arm][i]) for arm in ARMS}} for i in range(len(y))])
    flattened=[{k:v for k,v in r.items() if k!='weights'}|dict(zip(('mcp','pip','dip'),r['weights'])) for r in search_rows]
    write_csv(out/'inner_search.csv',flattened)
    result={'metrics':summaries,'paired_vs_unweighted':paired,'folds':fold_details,'images':len(y),'config':config,
            'protocol':'Frozen 5 outer folds; 3 inner grouped folds; at most 150 images/class per inner fitting/validation partition. '
                       'Choose mean inner macro-F1 then accuracy; coordinate order PIP, MCP, DIP, one pass; weights 0.5/1/2. '
                       'Full outer training and validation; imputation/scaling fitted only on each training partition. '
                       'Fixed RBF SVM C=10, gamma=scale, balanced. All other angles and Shin unchanged.',
            'limitations':'Index hypothesis was chosen after inspecting previous results on these same development folds; '
                          'nested weight selection does not turn them into a fresh independent test. Filename blocks are not known people.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2),flush=True)
    lines=['# Gerichte weging wijsvingerhoeken: SVM + Shin + 15 hoeken','',result['protocol'],'',result['limitations'],'',
           '| Variant | Accuracy | Macro-F1 | Correct | Hersteld / verloren t.o.v. ongewogen |',
           '|---|---:|---:|---:|---:|']
    for arm,m in summaries.items():lines.append(f"| {arm} | {m['accuracy']:.3%} | {m['macro_f1']:.5f} | {m['correct']} | {paired[arm]['recovered']} / {paired[arm]['lost']} |")
    lines+=['','unweighted: alle hoekgewichten 1. pip_2: vooraf vastgelegde PIP-verhoging naar 2. '
            'tuned_pip: alleen PIP intern gekozen. tuned_index: PIP, daarna MCP en DIP intern afgestemd.', '',
            '## Gewichten per buitenfold','', '| Fold | Variant | MCP | PIP | DIP |','|---|---|---:|---:|---:|']
    for detail in fold_details:
        for arm,weights in detail['selected_weights'].items():lines.append(f"| {detail['fold']} | {arm} | "+' | '.join(map(str,weights))+' |')
    lines+=['','## Gerichte verwarringen','', '| Echte letter → voorspelling | Ongewogen | PIP 2 | PIP getuned | Alle wijsvingerhoeken getuned |','|---|---:|---:|---:|---:|']
    for true,predicted in [('O','D'),('D','O'),('M','N'),('N','M'),('R','U'),('U','R')]:
        counts=[next(r['count'] for r in confusions if r['variant']==a and r['true_letter']==true and r['predicted_letter']==predicted) for a in ARMS]
        lines.append(f'| {true} → {predicted} | '+' | '.join(map(str,counts))+' |')
    lines+=['','## Recall per letter','', '| Letter | Beelden | Ongewogen | PIP 2 | PIP getuned | Alle wijsvingerhoeken getuned |','|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        rr={r['variant']:r for r in letters if r['letter']==label}
        lines.append(f"| {label} | {rr['unweighted']['images']} | "+' | '.join(f"{rr[a]['recall']:.2%}" for a in ARMS)+' |')
    lines+=['','Gewichten worden na standaardisatie toegepast. Factor 2 betekent factor 4 in het betreffende gekwadrateerde afstandsdeel van de RBF-kernel. '
            'Er is niet geselecteerd op alleen O/D en er is geen outer-fold keuze achteraf per letter toegepast. '
            'Dit onderzoek betreft bruikbare detecties; de ontbrekende detecties veranderen niet.', '',
            '```bash','python -m scripts.tune_index_angles','```','']
    (out/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
