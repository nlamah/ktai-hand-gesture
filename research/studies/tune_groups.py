"""Nested grouped weighting study on Shin's Kaggle ASL Alphabet data source."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support


from research.shin import combined_features
from research.inputs import encode_additions,encode_angles,write_csv,LABELS

WEIGHTS=(.25,.5,1.,2.)
INNER_PER_CLASS=150
ARMS=('shin_project','shin_paper_profile','combined_unweighted','combined_tuned')


def scale_angles(train,validation):
    imputer=SimpleImputer(strategy='median')
    scaler=StandardScaler()
    return scaler.fit_transform(imputer.fit_transform(train)),scaler.transform(imputer.transform(validation))


def weighted_input(X,angle_values,weights):
    angle_weight,category_weight,score_weight=weights
    return np.c_[X[:,:820],X[:,820:890]*category_weight,
                 X[:,890:916]*score_weight,angle_values*angle_weight]


def bounded_indices(indices,y,seed):
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(indices[y[indices]==label])[:INNER_PER_CLASS]
                                  for label in LABELS]))


def score(y,p):return {'accuracy':float(accuracy_score(y,p)),
                       'macro_f1':float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0))}


def svm(paper=False):
    return SVC(C=1000 if paper else 10,kernel='rbf',gamma='scale',
               class_weight=None if paper else 'balanced',cache_size=512)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=Path('data/asl_kaggle/research'))
    parser.add_argument('--reference',type=Path,default=Path('assets/asl_feature_references.xlsx'))
    parser.add_argument('--thresholds',type=Path,default=Path('config/thresholds.json'))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Completed experiment exists')
    started=time.monotonic()
    from research.studies.compare_models import load_inputs
    raw,y,ids,_=load_inputs(args.run)
    X=raw[:,:916];angles=raw[:,916:931]
    metadata=json.loads((args.run/'inputs_metadata.json').read_text())
    groups=np.array([s['group'] for s in metadata['samples']]);hashes=metadata['hashes']
    predictions={arm:np.full(len(ids),'',dtype='<U1') for arm in ARMS};assignments=np.zeros(len(ids),dtype=int)
    fold_results=[];inner_records=[];fold_records=[]
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(train,test) in enumerate(outer.split(X,y,groups),1):
        assert not set(groups[train])&set(groups[test])
        if set(y[train])!=set(LABELS):raise ValueError('Missing outer training labels')
        fold_file=out/f'fold_{fold}.json'
        if fold_file.exists():
            saved=json.loads(fold_file.read_text())
            if saved['test_ids']!=[ids[i] for i in test]:raise ValueError('Checkpoint differs')
            for arm in ARMS:predictions[arm][test]=saved['predictions'][arm]
            fold_results.extend(saved['metrics']);inner_records.extend(saved['inner_records']);fold_records.append(saved['split']);assignments[test]=fold
            print(f'Restored fold {fold}',flush=True);continue
        inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=420+fold)
        parts=[];inner_splits=[]
        for inner_fold,(fit,validate) in enumerate(inner.split(X[train],y[train],groups[train]),1):
            fit=train[fit];validate=train[validate]
            assert not set(groups[fit])&set(groups[validate])
            if set(y[fit])!=set(LABELS):raise ValueError('Missing inner training labels')
            fit=bounded_indices(fit,y,10000+100*fold+inner_fold)
            validate=bounded_indices(validate,y,20000+100*fold+inner_fold)
            a,b=scale_angles(angles[fit],angles[validate]);parts.append((fit,validate,a,b))
            inner_splits.append({'inner_fold':inner_fold,'training_groups':sorted(set(groups[fit])),
                                 'validation_groups':sorted(set(groups[validate])),
                                 'training_ids':[ids[i] for i in fit],'validation_ids':[ids[i] for i in validate],
                                 'training_images':len(fit),'validation_images':len(validate)})
        memo={};local=[]
        def evaluate(weights):
            key=tuple(weights)
            if key in memo:return memo[key]
            trial=[]
            for inner_fold,(fit,validate,a,b) in enumerate(parts,1):
                model=svm();model.fit(weighted_input(X[fit],a,key),y[fit])
                pred=model.predict(weighted_input(X[validate],b,key));metrics=score(y[validate],pred)
                record={'outer_fold':fold,'inner_fold':inner_fold,'angle_weight':key[0],
                        'category_weight':key[1],'score_weight':key[2],**metrics}
                trial.append(metrics);local.append(record)
            objective=(float(np.mean([m['macro_f1'] for m in trial])),float(np.mean([m['accuracy'] for m in trial])))
            memo[key]=objective
            print(f'Outer {fold}/5 inner weights={key} macroF1={objective[0]:.5f} accuracy={objective[1]:.5f}',flush=True)
            return objective
        # One coordinate pass: 10 distinct configurations, not an exhaustive 64-point search.
        best=(1.,1.,1.);best_score=evaluate(best)
        for block in range(3):
            selected=best;selected_score=best_score
            for weight in WEIGHTS:
                candidate=list(best);candidate[block]=weight;candidate=tuple(candidate)
                candidate_score=evaluate(candidate)
                if candidate_score>selected_score:selected,selected_score=candidate,candidate_score
            best,best_score=selected,selected_score
        print(f'Outer {fold}: chosen weights {best}',flush=True)
        a,b=scale_angles(angles[train],angles[test]);metrics_rows=[]
        for arm in ARMS:
            if arm.startswith('shin_'):
                fit_X,test_X=X[train,:820],X[test,:820]
            else:
                weights=best if arm=='combined_tuned' else (1.,1.,1.)
                fit_X,test_X=weighted_input(X[train],a,weights),weighted_input(X[test],b,weights)
            model=svm(paper=arm=='shin_paper_profile');model.fit(fit_X,y[train]);pred=model.predict(test_X)
            predictions[arm][test]=pred
            row={'fold':fold,'variant':arm,'train_images':len(train),'validation_images':len(test),**score(y[test],pred)}
            metrics_rows.append(row);print(json.dumps(row),flush=True)
        assignments[test]=fold
        split={'fold':fold,'train_groups':sorted(set(groups[train])),'test_groups':sorted(set(groups[test])),
               'weights':list(best),'inner_splits':inner_splits,'inner_objective':list(best_score)}
        saved={'test_ids':[ids[i] for i in test],'predictions':{arm:predictions[arm][test].tolist() for arm in ARMS},
               'metrics':metrics_rows,'inner_records':local,'split':split}
        fold_file.write_text(json.dumps(saved));fold_results.extend(metrics_rows);inner_records.extend(local);fold_records.append(split)
    summaries={};letters=[]
    for arm in ARMS:
        pred=predictions[arm];summaries[arm]=score(y,pred)
        summaries[arm].update(correct_images=int((y==pred).sum()),accuracy_all_images=float((y==pred).sum()/78000),
                              fold_accuracy_mean=float(np.mean([r['accuracy'] for r in fold_results if r['variant']==arm])),
                              fold_accuracy_sd=float(np.std([r['accuracy'] for r in fold_results if r['variant']==arm],ddof=1)))
        _,recall,f1,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
        for i,label in enumerate(LABELS):letters.append({'letter':label,'variant':arm,'images':int(support[i]),
                'correct_images':int(((y==label)&(pred==label)).sum()),'recall':float(recall[i]),'f1':float(f1[i])})
    paired={};base=predictions['shin_project']==y
    for arm in ARMS[1:]:
        correct=predictions[arm]==y;paired[arm]={'recovered':int((~base&correct).sum()),'lost':int((base&~correct).sum())}
    result={'dataset_source':'https://www.kaggle.com/grassknoted/asl-alphabet',
            'article':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8434249/',
            'hashes':hashes,'all_images':78000,'usable_images':len(ids),'source_groups':sorted(set(groups)),
            'features':{'shin':820,'categories':70,'excel_scores':26,'angles':15},
            'protocol':'5 outer and 3 inner StratifiedGroupKFold; session filename blocks, seed 42; one coordinate pass over 0.25/0.5/1/2. '
                       'Bound internal search to 150 images/class in each fitting and validation partition; outer training and evaluation use all usable images. '
                       'Choose inner mean macro-F1 then accuracy. Outer validation never used to select weights. '
                       'Angle imputation/scaling fitted on inner training partitions and refitted on outer training partition.',
            'limitations':'Same public dataset source as Shin ASL Alphabet experiment, not the exact published split, detector or a replication of published scores. '
                          'Filename blocks are not verified subjects. Fixed categorical thresholds/Excel proposals came from separate Roboflow development data; no Kaggle-fold threshold tuning. '
                          'Only letter classes A-Z. The provided 26 test photos duplicated training pixels and were excluded; no independent final test. '
                          'Coordinate search is bounded and not exhaustive. Current fixed SVM C/gamma not jointly tuned.',
            'metrics':summaries,'paired_vs_project':paired,'folds':fold_records,'elapsed_seconds':round(time.monotonic()-started,1)}
    write_csv(out/'per_letter.csv',letters);write_csv(out/'fold_metrics.csv',fold_results);write_csv(out/'inner_search.csv',inner_records)
    write_csv(out/'oof_predictions.csv',[{'id':ids[i],'label':y[i],'group':groups[i],'fold':int(assignments[i]),
                **{arm:str(predictions[arm][i]) for arm in ARMS}} for i in range(len(ids))])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Kaggle ASL Alphabet: geneste tuning van featuregewichten','',
           'Datasetbron uit Shin: [grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet), '
           'bevestigd in [het artikel](https://pmc.ncbi.nlm.nih.gov/articles/PMC8434249/). '
           '78.000 letterbeelden, 60.251 bruikbare detecties; alle 26 letters. De datasetbron komt overeen, '
           'maar detector en splitsing zijn geen exacte reproductie van het artikel.', '',
           result['protocol'],'',result['limitations'],'',
           '| Variant | Accuracy bruikbaar | Macro-F1 | Accuracy alle beelden |',
           '|---|---:|---:|---:|']
    for arm,m in summaries.items():lines.append(f"| {arm} | {m['accuracy']:.3%} | {m['macro_f1']:.5f} | {m['accuracy_all_images']:.3%} |")
    lines+=['','`shin_project`: C=10, balanced. `shin_paper_profile`: C=1000, geen class weighting. '
            'Beide RBF, gamma=scale. De gecombineerde modellen gebruiken C=10 en balanced om alleen featuregewicht te vergelijken. '
            'Excel-scores worden voor alle 26 kandidaatletters berekend, zonder het echte label als input.', '',
            '## Gekozen gewichten per buitenfold', '', '| Fold | Hoeken | Categorieën | Excel-scores |','|---|---:|---:|---:|']
    for row in fold_records:lines.append(f"| {row['fold']} | "+' | '.join(map(str,row['weights']))+' |')
    lines+=['','## Recall per letter','', '| Letter | Beelden | Shin project | Shin artikelprofiel | Ongewogen | Getuned | Verschil getuned − Shin |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        rr={r['variant']:r for r in letters if r['letter']==label}
        lines.append(f"| {label} | {rr['shin_project']['images']} | "+' | '.join(f"{rr[a]['recall']:.2%}" for a in ARMS)+
                     f" | {100*(rr['combined_tuned']['recall']-rr['shin_project']['recall']):+.2f} pp |")
    lines+=['','Alle detailresultaten: `per_letter.csv`, `fold_metrics.csv`, `inner_search.csv`, `oof_predictions.csv`, `summary.json`. '
            'De historische 97,43% gebruikte drie folds; vergelijk die niet rechtstreeks met deze nieuwe vijf folds. '
            'De correcte vergelijking is tussen de modellen op de gedeelde huidige folds.','']
    (out/'report.md').write_text('\n'.join(lines));print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
