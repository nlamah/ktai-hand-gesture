"""Build the mirror experiment report and figures from saved results."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from research.data import atomic_json
from research.inputs import write_csv
from research.studies.mirror_augmentation import VIEWS

OUT = Path('results/mirror_01')
PAPER = Path('paper')
NAMES = {'original_training': 'Original training', 'mirror_training': 'Mirror training'}
PRESENTATIONS = [('original', 'Original'), ('mirrored_rgb', 'Mirrored RGB'), ('balanced_mix', 'Balanced mix')]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((OUT/'summary.json').read_text())
    metrics = pd.read_csv(OUT/'metrics.csv')
    pooled = metrics[metrics.fold == 'pooled']
    lookup = {(r.model, r.training, r.presentation, r.scope): r for r in pooled.itertuples()}
    manifest_raw = Path('data/asl_kaggle/manifest.json').read_bytes()
    protocol = json.loads((OUT/'extraction_protocol.json').read_text())
    if sha256(manifest_raw).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('Original manifest changed.')
    manifest = json.loads(manifest_raw)['samples']; del manifest_raw
    original_hands = Counter(s.get('handedness', '') for s in manifest if s['status'] == 'ok')
    accepted = sum(original_hands.values())
    historical = {'source_images': len(manifest), 'accepted': accepted,
        'detector_Right': original_hands['Right'], 'detector_Left': original_hands['Left'],
        'right_fraction_among_accepted': original_hands['Right']/accepted,
        'left_fraction_among_accepted': original_hands['Left']/accepted,
        'not_accepted': len(manifest)-accepted, 'labels_are': 'MediaPipe predictions, not manually verified hands or signer dominance'}
    atomic_json(OUT/'historical_handedness.json', historical)
    lines = ['# Horizontal RGB mirroring for hand-side robustness', '',
        'This is one fixed augmentation experiment, separate from the primary hand-feature study. '
        'It asks whether training with original images plus horizontal RGB mirrors improves precision and F1 '
        'on held-out mirror presentations without sacrificing original-image performance.', '',
        '## Original dataset handedness', '',
        f"In the historical cache, {original_hands['Right']:,}/{accepted:,} accepted images "
        f"({100*historical['right_fraction_among_accepted']:.2f}%) are labelled Right by MediaPipe, "
        f"and {original_hands['Left']:,} ({100*historical['left_fraction_among_accepted']:.2f}%) Left. "
        f"The other {historical['not_accepted']:,} source images lack an accepted single-hand detection. "
        'These are detector estimates of the depicted hand, not human-confirmed handedness or signer dominance.', '',
        '## Frozen protocol', '',
        '- Process all 78,000 original A-Z images and their horizontal pixel reflections with the same detector, model and 0.5/0.5 thresholds in IMAGE mode. No pixel interpolation, rotation or anatomical landmark-index swapping.',
        '- Redetect both views independently. Use the detector-returned image/world landmarks and handedness, then the unchanged live 835-input encoder. Merely changing a handedness label is not augmentation.',
        '- Use the five existing source filename-block folds. A source and every derivative always share a group and held-out fold. The folds are not verified signer groups.',
        '- Train the baseline on newly accepted original fitting sources. For the candidate, add an accepted mirrored view only for those same sources. Failed mirrors do not remove originals; mirror-only recoveries do not expand the source population.',
        '- Preserve total fitting weight one per source (0.5 + 0.5 when both views are available). Fit angle preprocessing on original fitting rows only; preserve SVM gamma and original balanced class weights. Fixed classifier recipes, no tuning.',
        '- Evaluate the same models on originals, mirrors, and a fixed random mix with one view per source. The seed is 20261106; within each letter/fold, half the sources select a mirror.',
        '- All-attempt scores retain failed detections as no prediction (?). Conditional and paired-both-accepted scores are also exported. Precision is macro-averaged over A-Z; no prediction lowers recall/F1 but does not create a false positive.', '',
        f"The random mix contains **{summary['balanced_original']:,} originals and {summary['balanced_mirrored']:,} mirrors**. "
        f"Of its accepted detections, {summary['balanced_detected_Left']:,} are labelled Left and "
        f"{summary['balanced_detected_Right']:,} Right. Equal reflection frequency does not guarantee exactly equal detected hands.", '',
        '## Detection coverage', '', '| View | Accepted / attempts | Coverage | Detector Left | Detector Right |',
        '|---|---:|---:|---:|---:|']
    for r in summary['detection']:
        lines.append(f"| {r['view']} | {r['accepted']:,}/{r['images']:,} | {100*r['coverage']:.2f}% | {r['detected_Left']:,} | {r['detected_Right']:,} |")
    lines += ['', f"Both views are accepted for {summary['both_views_accepted']:,} sources; "
              f"handedness predictions swap for {summary['handedness_swaps_when_both_accepted']:,} of those pairs.", '',
              '## Main results: all attempted held-out sources', '',
              '| Learner | Presentation | Training | Accuracy | Macro-precision | Macro-recall | Macro-F1 |',
              '|---|---|---|---:|---:|---:|---:|']
    for model in ('svm', 'lightgbm'):
        for presentation, label in PRESENTATIONS:
            for arm in NAMES:
                r = lookup[model, arm, presentation, 'all_attempts']
                lines.append(f'| {model.upper()} | {label} | {NAMES[arm]} | {100*r.accuracy:.3f}% | {100*r.macro_precision:.3f}% | {100*r.macro_recall:.3f}% | {r.macro_f1:.6f} |')
    lines += ['', '## Paired changes', '']
    for model in ('svm', 'lightgbm'):
        for presentation, label in PRESENTATIONS:
            a = lookup[model, 'original_training', presentation, 'all_attempts']
            b = lookup[model, 'mirror_training', presentation, 'all_attempts']
            change = next(r for r in summary['paired'] if r['model']==model and r['presentation']==presentation and r['scope']=='all_attempts')
            lines.append(f"- {model.upper()}, {label}: macro-precision {100*(b.macro_precision-a.macro_precision):+.3f} percentage points; "
                f"macro-F1 {b.macro_f1-a.macro_f1:+.6f}; {change['fixed']:,} corrected, {change['lost']:,} lost, net {change['net']:+,}.")
    lines += ['', '## Balanced-mix fold differences', '',
        '| Learner | Fold | Precision change (pp) | F1 change |', '|---|---:|---:|---:|']
    fold_effects=[]
    for model in ('svm','lightgbm'):
        subset=metrics[(metrics.model==model)&(metrics.presentation=='balanced_mix')&
                       (metrics.scope=='all_attempts')&(metrics.fold!='pooled')]
        for fold in sorted(subset.fold.unique()):
            pair=subset[subset.fold==fold].set_index('training')
            dp=100*(pair.loc['mirror_training','macro_precision']-pair.loc['original_training','macro_precision'])
            df=pair.loc['mirror_training','macro_f1']-pair.loc['original_training','macro_f1']
            lines.append(f'| {model.upper()} | {fold} | {dp:+.4f} | {df:+.6f} |')
            fold_effects.append({'model':model,'fold':fold,'precision_change_pp':float(dp),'f1_change':float(df)})
    write_csv(OUT/'fold_effects.csv',fold_effects)
    lines += ['', 'These five fold differences are descriptive. Training sets overlap, and filename blocks are not independent signer samples.']
    a=lookup['svm','original_training','balanced_mix','all_attempts']
    b=lookup['svm','mirror_training','balanced_mix','all_attempts']
    oa=lookup['svm','original_training','original','all_attempts']
    ob=lookup['svm','mirror_training','original','all_attempts']
    lines += ['', f'The SVM balanced-mix F1 change is {b.macro_f1-a.macro_f1:+.6f}, while its original-view change is '
        f'{ob.macro_f1-oa.macro_f1:+.6f}. This distinguishes reflection robustness from original-distribution performance. '
        'The observed mixed-presentation gain is not a universal improvement or a measured real-left-signer effect.']
    lines += ['', '![All-attempt precision and F1](metrics.png)', '',
        '## Interpretation and limits', '',
        'These results measure robustness to synthetic horizontal RGB reflection. No real left-handed signer test was collected. '
        'The dataset hand labels are estimated, the source folds are not known people, and the same family of development folds has been inspected before. '
        'The balanced mix is a controlled presentation distribution, not an estimate of population prevalence. '
        'J and Z are static dataset poses; their movement is not evaluated. '
        'Precision/F1 gains on mirrors must be read alongside original-image regressions, detection coverage and all-letter results. '
        'The earlier 97.748% world-joint score used historical accepted detections; compare the two fresh arms here instead of attributing any redetection differences to augmentation.', '',
        'Per-letter precision/recall/F1 are in `per_letter.csv`; all fold scores in `metrics.csv`; '
        'no-prediction columns are retained in confusion matrices. `handedness_metrics.csv` is a detector-labelled diagnostic, not a ground-truth left/right evaluation.', '',
        '## Reproduce and try live', '', '```bash',
        'python -m RQ3 mirrors',
        'python -m RQ3 mirrors --action verify',
        'python -m research.live_camera --model models/asl_shin_angles_mirrored.joblib', '```', '',
        'The extraction stage resumes matching chunks; CV reuses fold models only under the saved protocol. '
        'Full-data training refuses to overwrite existing live bundles. The previous default live model remains available.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for i, model in enumerate(('svm', 'lightgbm')):
        for j, (metric, title) in enumerate([('macro_precision', 'Macro-precision'), ('macro_f1', 'Macro-F1')]):
            ax = axes[i,j]; x = np.arange(3)
            for offset, (arm, label), color in zip((-.19,.19), NAMES.items(), ('#375b79','#147e72')):
                values = [100*getattr(lookup[model,arm,p,'all_attempts'],metric) for p,_ in PRESENTATIONS]
                bars=ax.bar(x+offset,values,.36,label=label,color=color)
                ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
            ax.set_xticks(x,[p[1] for p in PRESENTATIONS]);ax.set_ylim(0,108)
            ax.set_ylabel(title+' (%)');ax.set_title(model.upper())
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'metrics.png',dpi=170);fig.savefig(PAPER/'figures/mirror_metrics.pdf');plt.close(fig)
    # Deterministic visual examples: first accepted pair for each named letter, no outcome-based selection.
    byid = {s['id']: s for s in manifest};del manifest
    chosen = {}
    for path in sorted(Path('runs/mirror_01/detections').glob('chunk_*.npz')):
        with np.load(path) as data:
            for letter in 'AGHR':
                if letter in chosen:continue
                ix=np.flatnonzero((data['labels']==letter)&(data['status']=='ok').all(axis=1))
                if len(ix):
                    k=ix[0];chosen[letter]=(str(data['ids'][k]),data['image_landmarks'][k].copy(),data['handedness'][k].tolist())
        if len(chosen)==4:break
    sheet=Image.new('RGB',(800,4*260),'white');draw=ImageDraw.Draw(sheet)
    for row,letter in enumerate('AGHR'):
        id_, points, hands=chosen[letter];s=byid[id_]
        with Image.open(Path('data/asl_kaggle')/s['image_path']) as im:original=im.convert('RGB')
        from research.mirroring import mirror_image
        for view,image in enumerate((original,mirror_image(original))):
            image=image.resize((220,220));x=view*400+90;y=row*260+28;sheet.paste(image,(x,y))
            draw.text((view*400+10,row*260+8),f'{letter} / {VIEWS[view]} / detector {hands[view]}',fill='black')
            for px,py,_ in points[view]:
                cx=x+float(px)*220;cy=y+float(py)*220;draw.ellipse((cx-2,cy-2,cx+2,cy+2),fill='#00ffae')
    sheet.save(OUT/'rgb_examples.png')
    print(OUT/'report.md')


if __name__ == '__main__':main()
