"""Compare frozen reference/angle models on completed, matched A-Z sessions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix

from research.alphabet_protocol import AlphabetSession
from research.data import atomic_json
from research.inputs import LABELS,write_csv
from research.live_model import DEFAULT_MODEL,DEFAULT_BASELINE_MODEL,load_bundle,model_inputs


def metrics(y,pred):
    if not len(y):return {'images':0,'accuracy':None,'macro_f1':None}
    return {'images':len(y),'correct':int((np.asarray(y)==np.asarray(pred)).sum()),
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions',type=Path,nargs='+')
    parser.add_argument('--out',type=Path,default=Path('results/alphabet_evaluation'))
    args=parser.parse_args()
    paths={'reference':DEFAULT_BASELINE_MODEL,'angles':DEFAULT_MODEL}
    hashes={k:sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    bundles={k:load_bundle(p) for k,p in paths.items()}
    predictions=[];seen=set();profiles=[];example_plan=None;repeat_count=None
    for directory in args.sessions:
        s=AlphabetSession(directory,hashes['angles'])
        if s.current is not None:raise ValueError(f'Session is incomplete: {directory}')
        if example_plan is None:example_plan=s.plan['examples'];repeat_count=s.plan['repeats']
        if s.plan['examples']!=example_plan or s.plan['repeats']!=repeat_count:
            raise ValueError('Sessions use different examples or repetition counts.')
        key=s.plan['participant']['name'].strip().casefold()
        if key in seen:raise ValueError('The same participant name was supplied more than once; use one session per person.')
        seen.add(key);profiles.append({'session_id':s.plan['id'],**s.plan['participant']})
        for name,p in paths.items():assert s.plan['frozen_models'][str(p.resolve())]==hashes[name]
        for task in s.plan['tasks']:
            folder=Path(s.completed[task['id']]);record=json.loads((folder/'capture.json').read_text())
            for filename,info in record['files'].items():assert sha256((folder/filename).read_bytes()).hexdigest()==info['sha256']
            with Image.open(folder/'model_input.png') as image:size=list(image.size)
            detection={**record['detection'],'image_size':size};ok=detection['status']=='ok'
            pr={k:str(b['estimator'].predict(model_inputs(b,detection))[0]) if ok else '?' for k,b in bundles.items()}
            if ok:assert pr['angles']==record['predicted_letter'],'Saved prediction cannot be replayed'
            predictions.append({'id':record['id'],'session_id':s.plan['id'],'participant':s.plan['participant']['name'],
                'kind':s.plan['participant']['kind'],'experience':s.plan['participant']['experience'],
                'letter':task['letter'],'repeat':task['repeat'],'detection_status':detection['status'],
                'operator_guidance':bool(record['protocol'].get('operator_guidance')),**pr})
    args.out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    groups=[('all_participants',predictions),('new_people_only',[r for r in predictions if r['kind']=='new']),
            ('development_people',[r for r in predictions if r['kind']=='development'])]
    groups += [(p['session_id'],[r for r in predictions if r['session_id']==p['session_id']]) for p in profiles]
    per_letter=[]
    for group,rows in groups:
        if not rows:continue
        accepted=[r for r in rows if r['detection_status']=='ok'];y=[r['letter'] for r in rows]
        summary={'group':group,'images':len(rows),'accepted_detections':len(accepted),'detection_coverage':len(accepted)/len(rows),
                 'guided_attempts':sum(r['operator_guidance'] for r in rows),'models':{}}
        for model in bundles:
            pred=[r[model] for r in rows]
            summary['models'][model]={'all_attempts':metrics(y,pred),
                'accepted_only':metrics([r['letter'] for r in accepted],[r[model] for r in accepted]),
                'confusion_labels':LABELS+['?'],'confusion':confusion_matrix(y,pred,labels=LABELS+['?']).tolist()}
            precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
            per_letter += [{'group':group,'model':model,'letter':l,'images':int(support[i]),
                'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate(LABELS)]
        summary['fixed']=sum(r['reference']!=r['letter'] and r['angles']==r['letter'] for r in rows)
        summary['lost']=sum(r['reference']==r['letter'] and r['angles']!=r['letter'] for r in rows)
        summaries.append(summary)
    atomic_json(args.out/'summary.json',{'models':{k:{'path':str(paths[k]),'sha256':hashes[k]} for k in bundles},
        'participants':profiles,'groups':summaries,
        'scope':'Example-guided static dataset-letter reproduction, not natural signing or dynamic J/Z. New people are separated from development participants. No models were trained or selected by this evaluation.'})
    write_csv(args.out/'predictions.csv',predictions);write_csv(args.out/'per_letter.csv',per_letter)
    lines=['# Paired alphabet evaluation','',
        '| Group | Photos | Detection coverage | Reference accuracy | Angles accuracy | Reference macro-F1 | Angles macro-F1 | Fixed / lost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        a=s['models']['reference']['all_attempts'];b=s['models']['angles']['all_attempts']
        lines.append(f"| {s['group']} | {s['images']} | {100*s['detection_coverage']:.1f}% | {100*a['accuracy']:.2f}% | {100*b['accuracy']:.2f}% | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {s['fixed']} / {s['lost']} |")
    lines += ['', 'The table counts failed detections as incorrect attempts. `summary.json` additionally reports accuracy/F1 conditional on accepted detections. '
        'Predictions are paired on the same images. Experience and guidance flags are retained. '
        'The number of people is small; these results do not justify a population-level superiority claim. '
        'Examples come from the training dataset and were not independently ASL-validated. This is a reference-guided reproduction task, not an assessment of spontaneous signing. '
        'Once results are inspected for model choices, these people are development participants for subsequent choices.','']
    (args.out/'report.md').write_text('\n'.join(lines));print(args.out/'report.md')


if __name__=='__main__':main()
