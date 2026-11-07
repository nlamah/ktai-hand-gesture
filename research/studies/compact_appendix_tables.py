"""Select and translate the frozen results actually cited by the concise report."""
from hashlib import sha256
from pathlib import Path

import pandas as pd

from research.data import atomic_json

PAPER=Path('paper')
SOURCES={}


def read(path):
    SOURCES[path]=sha256(Path(path).read_bytes()).hexdigest()
    return pd.read_csv(path,keep_default_na=False)


def fmt(value,digits=5):return f'{float(value):.{digits}f}'.replace('.',',')
def integer(value):return f'{int(value):,}'.replace(',','.')
def write(name,rows):
    (PAPER/name).write_text('% Afgeleid van opgeslagen resultaten; gegenereerd door compact_appendix_tables.py.\n'+
                           '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n')


def main():
    base=read('results/model_comparison/model_metrics.csv')
    angles={model:read(f'results/angles/metrics_{model}.csv') for model in ('svm','lightgbm')}
    control=read('results/direction_joint_01/metrics.csv')
    def pooled(frame,model,variant):
        rows=frame[(frame.model==model)&(frame.variant==variant)&(frame.fold=='pooled')]
        assert len(rows)==1 and int(rows.iloc[0].images)==60251
        return rows.iloc[0]
    summary=[]
    for title,kind in [('Basis (Shin)','shin'),('Alle toevoegingen','combined'),
                       ('+ Wereldgewrichten','world'),('+ Zelfde-coördinatenhoeken','same')]:
        values=[]
        for model in ('svm','lightgbm'):
            if kind in ('shin','combined'):r=pooled(base,model,kind)
            elif kind=='world':r=pooled(angles[model],model,'shin_angles')
            else:r=pooled(control,model,'shin_joint')
            values.extend((fmt(100*r.accuracy,5),fmt(r.macro_f1,6)))
        summary.append([title,*values])
    write('appendix_feature_summary_nl.tex',summary)
    letters=read('results/angles/per_letter_svm.csv')
    letter_control=read('results/direction_joint_01/per_letter.csv')
    rows=[]
    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        choices=[]
        for frame,variant in ((letters,'shin'),(letters,'shin_angles'),(letter_control,'shin_joint')):
            r=frame[(frame.model=='svm')&(frame.variant==variant)&(frame.letter==label)].iloc[0]
            choices.append(r)
        assert len({int(r.images) for r in choices})==1
        rows.append([label,integer(choices[0].images),*[x for r in choices for x in (fmt(100*r.precision,2),fmt(r.f1,5))]])
    write('appendix_feature_letters_nl.tex',rows)
    mirror=read('results/mirror_01/metrics.csv')
    rows=[]
    for presentation,title in [('original','Origineel'),('mirrored_rgb','Gespiegeld'),('balanced_mix','Vast mengsel')]:
        for arm,training in [('original_training','Origineel'),('mirror_training','+ spiegeling')]:
            r=mirror[(mirror.model=='svm')&(mirror.presentation==presentation)&(mirror.scope=='all_attempts')&
                     (mirror.fold=='pooled')&(mirror.training==arm)].iloc[0]
            assert int(r.images)==78000
            rows.append([title,training,integer(round(r.images*r.coverage)),fmt(100*r.macro_precision,3),
                         fmt(r.macro_f1,5),integer(r.correct)])
    write('appendix_mirror_nl.tex',rows)
    gh=read('results/rotation_GH_y180_01/metrics.csv')
    rows=[]
    for arm,title in [('original','Origineel'),('gh_y180','+ G/H-rotatie')]:
        r=gh[(gh.model==arm)&(gh.fold=='pooled')].iloc[0]
        rows.append([title,fmt(100*r.accuracy,5),fmt(100*r.macro_precision,5),fmt(r.macro_f1,8),integer(r.correct)])
    write('appendix_gh_summary_nl.tex',rows)
    gh_letters=read('results/rotation_GH_y180_01/per_letter.csv')
    rows=[]
    for label in ('G','H'):
        for arm,title in [('original','Origineel'),('gh_y180','+ rotatie')]:
            r=gh_letters[(gh_letters.model==arm)&(gh_letters.fold=='pooled')&(gh_letters.letter==label)].iloc[0]
            rows.append([label,title,integer(r.images),fmt(r.precision,6),fmt(r.recall,6),fmt(r.f1,6)])
    write('appendix_gh_letters_nl.tex',rows)
    folds=read('results/rotation_GH_y180_01/paired_changes.csv')
    assert int(folds[folds.fold=='pooled'].iloc[0]['net'])==-1
    row=folds[folds.fold=='pooled'].iloc[0]
    assert int(row.fixed)==4 and int(row.lost)==5
    atomic_json(PAPER/'appendix_table_sources.json',{
        'status':'passed','selection':'Only tables supporting the five referenced Dutch appendix sections',
        'sources':SOURCES,'generated_rows':{'feature_summary':4,'feature_letters':26,'mirror':6,'gh_summary':2,'gh_letters':4},
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'note':'No training, tuning, new predictions or changes to frozen experiment evidence.'})
    print('Selected 42 frozen result rows; Dutch tables generated.',flush=True)


if __name__=='__main__':main()
