"""Compact local reports, explicit selected examples and editable human notes."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from gesturelab.datasets.data import overlay
from gesturelab.training.experiments import heatmap
from gesturelab.validation.report import write_json


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def _scores(out, result):
    summaries = []
    for side, scores in result['scores'].items():
        if scores is None:
            continue
        summaries.append(dict(model=side, **{key: scores[key] for key in ('accuracy', 'macro_f1', 'n')}))
        write_csv(out / (side + '_per_letter.csv'), scores['per_class'])
        table = pd.DataFrame(scores['confusion'], index=scores['labels'], columns=scores['predicted_columns'])
        table.to_csv(out / (side + '_confusion_counts.csv'))
        normalized = table.div(table.sum(axis=1).replace(0, 1), axis=0)
        normalized.to_csv(out / (side + '_confusion_normalized.csv'))
        heatmap(normalized, out / (side + '_confusion.png'), f'{side.upper()} | rows actual, columns predicted | same usable photos')
    write_csv(out / 'scores.csv', summaries)
    write_csv(out / 'detection_coverage.csv', result['coverage'])
    condition_rows = [dict(condition=condition, model=side, **{k: score[k] for k in ('accuracy', 'macro_f1', 'n')})
                      for condition, models in result['conditions'].items() for side, score in models.items() if score is not None]
    write_csv(out / 'condition_scores.csv', condition_rows)
    first, second = result['scores'].get('a'), result['scores'].get('b')
    if first and second:
        write_csv(out / 'letter_differences.csv', [dict(label=a['label'], a_f1=a['f1'], b_f1=b['f1'],
            f1_difference=b['f1'] - a['f1'], support=a['support']) for a, b in zip(first['per_class'], second['per_class'])])
    return summaries


def _examples(out, rows, samples, project):
    records = {s['id']: s for s in samples}
    ordered = sorted(rows, key=lambda r: r['id'])
    errors = [r for r in ordered if r['category'] != 'both_correct']
    correct = [r for r in ordered if r['category'] == 'both_correct']
    candidates = errors[:24] + correct[:12]
    folder = out / 'examples'
    folder.mkdir(exist_ok=True)
    selected = []
    for row in candidates:
        sample = records[row['id']]
        with Image.open(Path(project) / sample['image_path']) as image:
            overlay(image, sample).save(folder / (row['id'] + '.png'))
        selected.append({**row, 'example_file': 'examples/' + row['id'] + '.png'})
    write_csv(out / 'selected_examples.csv', selected)
    return selected


def _cards(examples):
    cards = []
    for row in examples:
        identity = html.escape(row['id'], quote=True)
        truth, predicted = html.escape(row['label']), html.escape(row['a'])
        cards.append(f'<figure data-id="{identity}" data-truth="{truth}" data-pred="{predicted}">'
            f'<img width="240" src="{row["example_file"]}"><figcaption>True: {truth}; A: {predicted}; B: {html.escape(row["b"])}'
            f'<br>{html.escape(row["condition"])}; {html.escape(row["category"])}</figcaption>'
            '<select><option>Uncertain</option><option>Gesture unclear/incorrect</option><option>Landmarks may be incorrect</option>'
            '<option>Letter confusion with usable landmarks</option><option>Possible viewpoint effect</option></select>'
            '<textarea placeholder="Human observation, not an automatically proved cause"></textarea></figure>')
    return ''.join(cards)


def _note_script():
    return '''<script>
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('figure[data-id]').forEach(f=>f.hidden=!!b.dataset.truth&&(f.dataset.truth!==b.dataset.truth||f.dataset.pred!==b.dataset.pred));});
document.getElementById('notes').onclick=()=>{
 const notes=[...document.querySelectorAll('figure[data-id]')].filter(f=>f.querySelector('textarea').value.trim()).map(f=>({id:f.dataset.id,category:f.querySelector('select').value,note:f.querySelector('textarea').value}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='human_notes.json';a.click();};
</script>'''


def _condition_table(result):
    rows = [dict(condition=condition, model=side, **{key: score[key] for key in ('n', 'accuracy', 'macro_f1')})
            for condition, models in result['conditions'].items() for side, score in models.items() if score]
    return '<h2>Scores per nominal condition</h2>' + pd.DataFrame(rows).to_html(index=False)


def export_evaluation(out, result, rows, samples, project):
    out = Path(out)
    write_json(out / 'result.json', result)
    write_csv(out / 'predictions.csv', rows)
    summaries = _scores(out, result)
    examples = _examples(out, rows, samples, project)
    pairs = Counter((r['label'], r['a']) for r in rows if r['usable'] and r['included'] and r['a'] != r['label'])
    write_csv(out / 'confusing_pairs.csv', [dict(actual=a, predicted=b, count=n) for (a, b), n in pairs.most_common()])
    buttons = '<button data-filter="1">All selected examples</button>' + ''.join(
        f'<button data-filter="1" data-truth="{html.escape(a, quote=True)}" data-pred="{html.escape(b, quote=True)}">{html.escape(a)} → {html.escape(b)} ({n})</button>'
        for (a, b), n in pairs.most_common(12))
    models = pd.DataFrame([dict(model=chr(65 + i), name=m['name'], classifier=m['classifier'], variant=m['variant'], sha256=m['model_sha256'])
                           for i, m in enumerate(result['models'])]).to_html(index=False)
    text = '<html lang="en"><meta charset="utf-8"><title>Viewpoint research</title><body style="font:16px Arial;margin:30px">'
    text += '<h1>Viewpoint research — ' + html.escape(result['phase']) + '</h1><p>' + html.escape(result['policy']) + '</p>'
    if any(m.get('synthetic') for m in result['models']):
        text += '<p><strong>SYNTHETIC SOFTWARE TEST — not a real ASL recognition measurement.</strong></p>'
    text += f'<p>Offered: {result["offered"]}; included: {result["included"]}; usable: {result["usable"]}. No CV error bars.</p>'
    text += models + pd.DataFrame(summaries).to_html(index=False) + pd.DataFrame(result['coverage']).to_html(index=False)
    text += _condition_table(result)
    if result.get('previously_viewed_retest'):
        text += '<p><strong>Previously viewed test data: this is a logged retest, not new independent confirmation.</strong></p>'
    text += '<p>' + html.escape(result.get('limitation', '')) + '</p>'
    text += '<p>Prediction columns retain the complete model alphabet. Outside-stress answers are wrong. Model-specific errors stay in the usable-row denominator.</p>'
    text += '<p>One participant remains one participant; multiple viewpoints/frames are not new people. Human assessments and real angle/label validity have not been automatically established.</p>'
    text += f'<p>Participants: {len({s["participant"] for s in samples})}; sessions: {len({(s["participant"], s["session"]) for s in samples})}; prior evaluations: {result.get("previous_evaluations", 0)}.</p>'
    text += '<div>' + ''.join(f'<img style="max-width:100%" src="{side}_confusion.png">' for side in result['scores'] if result['scores'][side]) + '</div>'
    text += '<h2>Selected error/correct examples and human notes</h2><p>Filters apply to displayed examples; full counts are in predictions.csv.</p>' + buttons
    text += '<button id="notes">Download entered human notes</button><div style="display:flex;flex-wrap:wrap">' + _cards(examples) + '</div>'
    (out / 'report.html').write_text(text + _note_script() + '</body></html>', encoding='utf-8')
