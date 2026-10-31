"""Paired grouped five-fold SVM comparison with frozen workbook-derived features."""
import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from gesturelab.features.categorical_hand import FINGER_JOINTS

from gesturelab.datasets.data import Detector, MODEL_SHA256, atomic_json
from gesturelab.features.asl_excel_review import read_coco_split, split_fingerprint, load_reference_workbook, categorize_workbook_features, measure_hand_features
from gesturelab.features.paper_shin2021 import combined_features, VERSION
from gesturelab.training.classifiers import estimator
from gesturelab.features.asl_threshold_tuning import _source_group

LABELS=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
VARIANTS=['shin','shin_categories','shin_categories_original','shin_categories_proposal']

ANGLE_COLUMNS=tuple(f"{joint}_angle_deg" for joint in FINGER_JOINTS)+("thumb_mcp_signed_deviation_deg",)
ANGLE_VARIANTS=['shin_angles','shin_angles_other_categories','shin_categories_angles','shin_categories_proposal_angles']


def encode_angles(measurements):
    """Fourteen internal joint angles and signed thumb-MCP deviation, in degrees."""
    return np.array([np.nan if measurements[key] is None else measurements[key] for key in ANGLE_COLUMNS])


def angle_estimator(angle_start, n_columns):
    # Keep Shin and categorical values unchanged. Fit imputation and scaling of
    # degree-valued inputs on the fitting fold only.
    preprocessing=ColumnTransformer([
        ('base','passthrough',list(range(angle_start))),
        ('angles',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),
         list(range(angle_start,n_columns)))],sparse_threshold=0)
    return make_pipeline(preprocessing,SVC(C=10,gamma='scale',kernel='rbf',class_weight='balanced'))


def category_options(feature):
    if feature=='thumb_mcp_shape':return ['B-vormig','not_B-vormig','unknown']
    if feature.startswith('flexion_'):return ['bent','partly_bent','extended','unknown']
    if feature.startswith('spread_'):return ['spread','not_spread','unknown']
    if feature=='crossing_index_middle':return ['suspected_crossed','not_crossed','unknown']
    return ['contact','no_contact','unknown']


def encode_additions(measurements, parameters, references):
    """Image geometry and fixed reference tables only; no sample label input."""
    actual=categorize_workbook_features(measurements,parameters)
    for feature in list(actual):
        if feature.startswith('thumb_tip_') and actual[feature] in ('near','far'):
            actual[feature]='no_contact'
    keys=list(references['A'])
    categories=np.array([float(actual[feature]==value) for feature in keys for value in category_options(feature)])
    scores=[]
    for label in LABELS:
        cells=[(feature,cell['allowed']) for feature,cell in references[label].items() if cell['allowed'] is not None]
        scores.append(sum(actual[feature] in allowed for feature,allowed in cells)/len(cells))
    return categories,np.array(scores)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reuse-run',type=Path)
    parser.add_argument('--joint-angles',action='store_true')
    parser.add_argument('--dataset',type=Path,default=Path('data/external/asl_letters_v1/train'))
    parser.add_argument('--original',type=Path,default=Path('assets/asl_feature_references.xlsx'))
    parser.add_argument('--proposal',type=Path,default=Path('outputs/asl_slash_proposal_20261031/ASL_featurevoorstel.xlsx'))
    parser.add_argument('--thresholds',type=Path,default=Path('config/asl_landmark_thresholds_tuned.json'))
    args=parser.parse_args();out=args.out
    variants=VARIANTS+ANGLE_VARIANTS if args.joint_angles else VARIANTS
    prior=json.loads((args.reuse_run/'summary.json').read_text()) if args.reuse_run else None
    if (out/'summary.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    samples=read_coco_split(args.dataset);fingerprint=split_fingerprint(args.dataset,samples)
    cache_path=(args.reuse_run/'full_landmarks.json') if args.reuse_run else out/'full_landmarks.json'
    if cache_path.exists():
        cache=json.loads(cache_path.read_text())
        if cache['dataset_fingerprint']!=fingerprint:raise ValueError('Cache mismatch')
        samples=cache['samples']
    else:
        detector=Detector()
        try:
            for i,sample in enumerate(samples):
                with Image.open(sample['image_path']) as image:
                    detection=detector.analyze(image)
                    sample['width'],sample['height']=image.size
                sample.update({key:value for key,value in detection.items() if key!='features'})
                if (i+1)%100==0:print(f'Detection {i+1}/{len(samples)}',flush=True)
        finally:detector.close()
        atomic_json(cache_path,{'dataset_fingerprint':fingerprint,'samples':samples},compact=True)
    refs=load_reference_workbook(args.original.read_bytes());proposal=load_reference_workbook(args.proposal.read_bytes())
    parameters=prior['thresholds'] if prior else json.loads(args.thresholds.read_text())
    if prior:
        if prior['dataset_fingerprint']!=fingerprint:raise ValueError('Prior dataset differs')
        if prior['reference_sha256']!=sha256(args.original.read_bytes()).hexdigest():raise ValueError('Original references changed')
        if prior['proposal_sha256']!=sha256(args.proposal.read_bytes()).hexdigest():raise ValueError('Proposal changed')
    groups=np.array([_source_group(sample) for sample in samples]);y=np.array([sample['label'] for sample in samples])
    valid=np.zeros(len(samples),dtype=bool);data={v:[] for v in variants};invalid=[]
    for i,sample in enumerate(samples):
        if sample.get('status')!='ok':continue
        try:
            shin=np.array(list(combined_features(sample['image_landmarks']).values()))
            measurements=measure_hand_features(sample['world_landmarks'],sample.get('handedness'))
            cats,old_scores=encode_additions(measurements,parameters,refs)
            _,new_scores=encode_additions(measurements,parameters,proposal)
            rows=[shin,np.r_[shin,cats],np.r_[shin,cats,old_scores],np.r_[shin,cats,new_scores]]
            if not all(np.isfinite(row).all() for row in rows):raise ValueError('Nonfinite inputs')
            if args.joint_angles:
                angles=encode_angles(measurements)
                other_mask=np.array([not(feature.startswith('flexion_') or feature=='thumb_mcp_shape')
                                     for feature in refs['A'] for value in category_options(feature)])
                rows.extend([np.r_[shin,angles],np.r_[shin,cats[other_mask],angles],
                             np.r_[shin,cats,angles],np.r_[shin,cats,new_scores,angles]])
        except (ValueError,KeyError) as error:
            invalid.append({'id':sample['id'],'error':str(error)});continue
        valid[i]=True
        for variant,row in zip(variants,rows):data[variant].append(row)
    X={variant:np.array(rows) for variant,rows in data.items()}
    valid_indices=np.flatnonzero(valid);full_to_valid={index:i for i,index in enumerate(valid_indices)}
    predictions={variant:np.full(len(samples),'__no_detection__',dtype='<U20') for variant in variants}
    assignments=np.zeros(len(samples),dtype=int);fold_rows=[];split_rows=[]
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    splits=list(splitter.split(np.zeros(len(y)),y,groups))
    if args.reuse_run:
        index_by_id={sample['id']:i for i,sample in enumerate(samples)}
        previous_splits=json.loads((args.reuse_run/'folds.json').read_text())
        splits=[(np.array([index_by_id[key] for key in row['training_ids']]),
                 np.array([index_by_id[key] for key in row['validation_ids']])) for row in previous_splits]
    for fold,(train,test) in enumerate(splits,1):
        assert not set(groups[train])&set(groups[test])
        fit=np.array([full_to_valid[i] for i in train if valid[i]])
        evaluate=np.array([full_to_valid[i] for i in test if valid[i]])
        test_valid=test[valid[test]];train_valid=train[valid[train]]
        if set(y[train_valid])!=set(LABELS):raise ValueError('Missing training class')
        assignments[test]=fold
        split_rows.append({'fold':fold,'training_ids':[samples[i]['id'] for i in train],
                           'validation_ids':[samples[i]['id'] for i in test],
                           'training_groups':sorted(set(groups[train])), 'validation_groups':sorted(set(groups[test]))})
        for variant in variants:
            model=(angle_estimator(X[variant].shape[1]-len(ANGLE_COLUMNS),X[variant].shape[1])
                   if variant in ANGLE_VARIANTS else
                   estimator('svm',variant='shin2021_combined',profile='project',seed=42))
            model.fit(X[variant][fit],y[train_valid])
            pred=model.predict(X[variant][evaluate]);predictions[variant][test_valid]=pred
            row={'fold':fold,'variant':variant,'train_usable':len(fit),'validation_all':len(test),
                 'validation_usable':len(evaluate),'accuracy_usable':accuracy_score(y[test_valid],pred),
                 'macro_f1_usable':f1_score(y[test_valid],pred,labels=LABELS,average='macro',zero_division=0),
                 'accuracy_all':accuracy_score(y[test],predictions[variant][test])}
            fold_rows.append(row);print(json.dumps(row),flush=True)
    summaries={};letter_rows=[]
    for variant in variants:
        pred=predictions[variant]
        summary={'features':X[variant].shape[1], 'accuracy_usable':accuracy_score(y[valid],pred[valid]),
                 'macro_f1_usable':f1_score(y[valid],pred[valid],labels=LABELS,average='macro',zero_division=0),
                 'accuracy_all':accuracy_score(y,pred),'correct_images':int((y==pred).sum()),
                 'fold_accuracy_mean':float(np.mean([r['accuracy_usable'] for r in fold_rows if r['variant']==variant])),
                 'fold_accuracy_std':float(np.std([r['accuracy_usable'] for r in fold_rows if r['variant']==variant],ddof=1))}
        summaries[variant]=summary
        _,recall,f1,support=precision_recall_fscore_support(y[valid],pred[valid],labels=LABELS,zero_division=0)
        for i,label in enumerate(LABELS):
            letter_rows.append({'letter':label,'variant':variant,'usable_images':int(support[i]),
                                'correct_images':int(((y==label)&(pred==label)).sum()),
                                'recall':float(recall[i]),'f1':float(f1[i])})
        write_csv(out/f'confusion_{variant}.csv',[{'true_letter':label,**dict(zip(LABELS,map(int,row)))}
                   for label,row in zip(LABELS,confusion_matrix(y[valid],pred[valid],labels=LABELS))])
    paired={}
    base=predictions['shin']==y
    for variant in variants[1:]:
        correct=predictions[variant]==y
        paired[variant]={'recovered':int((~base&correct&valid).sum()),'lost':int((base&~correct&valid).sum())}
    write_csv(out/'per_letter.csv',letter_rows);write_csv(out/'fold_metrics.csv',fold_rows)
    write_csv(out/'oof_predictions.csv',[{'id':sample['id'],'filename':sample['filename'],'true_label':sample['label'],
               'group':groups[i],'fold':int(assignments[i]),'usable':bool(valid[i]),
               **{variant:str(predictions[variant][i]) for variant in variants}} for i,sample in enumerate(samples)])
    atomic_json(out/'folds.json',split_rows)
    summary={'dataset_fingerprint':fingerprint,'dataset':str(args.dataset),'total_images':len(samples),'usable_images':int(valid.sum()),
             'source_groups':len(set(groups)),'detection_statuses':dict(Counter(sample.get('status') for sample in samples)),
             'angle_columns':list(ANGLE_COLUMNS) if args.joint_angles else [],
             'reuse_run':str(args.reuse_run) if args.reuse_run else None,
             'angle_preprocessing':'Median imputation plus StandardScaler, fitted only on each training fold; missingness indicators.',
             'input_failures':invalid,'seed':42,'folds':5,'split':'StratifiedGroupKFold on filename before .rf.; no subject identity available',
             'classifier':{'profile':'project','kernel':'rbf','C':10,'gamma':'scale','class_weight':'balanced',
                           'preprocessing':'Shin normalized per image; one-hot categories and 0–1 agreement scores unscaled.'},
             'shin_version':VERSION,'mediapipe_model_sha256':MODEL_SHA256,
             'reference_sha256':sha256(args.original.read_bytes()).hexdigest(),
             'proposal_sha256':sha256(args.proposal.read_bytes()).hexdigest(),'thresholds':parameters,
             'limitation':'Exploratory fixed-feature CV: threshold settings and slash proposals were previously selected using overlapping training data. '
                          'Model fitting is out of fold, but feature design is not fully nested. Not an unbiased final or subject-independent test. '
                          'No true letter is used in feature encoding; all 26 reference similarities computed for every image. No test-split images used.',
             'metrics':summaries,'paired_vs_shin':paired}
    atomic_json(out/'summary.json',summary)
    lines=['# Shin + workbookfeatures: vijfvoudige crossvalidatie','',summary['limitation'],'',
           f"Dataset: {len(samples)} beelden, {int(valid.sum())} bruikbaar. {len(set(groups))} brongroepen. "
           'Dezelfde folds, beelden en SVM-instellingen voor alle varianten. Detectiefouten tellen in de totaalscore als onjuist.', '',
           '| Variant | Features | Accuracy bruikbaar | Macro-F1 bruikbaar | Accuracy alle beelden |',
           '|---|---:|---:|---:|---:|']
    for v,s in summaries.items():lines.append(f"| {v} | {s['features']} | {s['accuracy_usable']:.2%} | {s['macro_f1_usable']:.4f} | {s['accuracy_all']:.2%} |")
    lines+=['','De categorievariant voegt gemeten buigstanden, binaire duim-MCP, spreiding, kruising en contact toe. '
            'De twee Excelvarianten voegen daarnaast 26 overeenkomstscores toe (één per mogelijke letter); zo hebben slashwaarden effect.', '',
            '## Per letter (recall op bruikbare beelden)', '',
            '| Letter | Beelden | '+' | '.join(variants)+' | Verschil laatste variant − Shin |',
            '|---|---:|'+'---:|'*(len(variants)+1)]
    for label in LABELS:
        rr={r['variant']:r for r in letter_rows if r['letter']==label}
        values=' | '.join(f"{rr[v]['recall']:.1%}" for v in variants)
        delta=100*(rr[variants[-1]]['recall']-rr['shin']['recall'])
        lines.append(f"| {label} | {rr['shin']['usable_images']} | {values} | {delta:+.1f} pp |")
    lines+=['','## Gekoppelde vergelijking met Shin','']
    for v,c in paired.items():lines.append(f"- {v}: {c['recovered']} eerder foute beelden hersteld; {c['lost']} eerder juiste beelden verloren.")
    lines+=['','Macro-F1 geeft elke letter gelijk gewicht. Per-letterverschillen zijn beschrijvend; '
            'kleine aantallen en gecorreleerde augmentaties beperken zekerheid. '
            'Er is geen hyperparameterzoekactie gedaan op de validatiefolds. '
            'Voor een onafhankelijke conclusie zijn opnieuw geneste featureafstelling of nieuwe ongebruikte beelden nodig.','']
    if args.joint_angles:
        lines+=['','## Gewrichtshoeken', '',
                'Veertien binnenhoeken (duim MCP/IP en MCP/PIP/DIP van de vier overige vingers), plus de getekende duim-MCP-afwijking. '
                'Alle waarden beginnen in graden; imputatie en standaardisatie worden uitsluitend per trainingsfold geleerd. '
                'Shin-invoer blijft ongewijzigd. Geen hyperparameters of featuregewichten aangepast op validatie-uitkomsten.', '',
                '- shin_angles: Shin plus alleen 15 continue hoekmetingen.',
                '- shin_angles_other_categories: buig- en duim-MCP-categorieën vervangen door hoeken; overige categorieën behouden.',
                '- shin_categories_angles: hoeken bovenop alle gemeten categorieën.',
                '- shin_categories_proposal_angles: hoeken bovenop categorieën en de 26 voorgestelde Excel-scores.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
