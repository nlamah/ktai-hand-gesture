"""Recompute plot data with pandas and export research figures from a completed run."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from verify_results import verify
from path_utils import resolve_path, stored_path


VARIANTS = {'dilatatie': 'Dilation', 'erosie': 'Erosion', 'canny_overlay': 'Canny overlay'}
COLOURS = ['#176E68', '#416987', '#8499AA', '#17334F']


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    verify(run)
    paired = pd.read_csv(run / 'per_image.csv', encoding='utf-8-sig', keep_default_na=False)
    if paired.empty:
        raise ValueError('No baseline misses available to plot.')
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    success = paired[list(VARIANTS)].eq('ok')
    counts = success.sum()
    counts.loc['any_variant_unique'] = success.any(axis=1).sum()
    labels = list(VARIANTS.values()) + ['Any variant\n(unique images)']
    table = counts.rename('recovered').to_frame()
    table['baseline_misses'] = len(paired)
    table['recovery_percent'] = table['recovered'] / len(paired) * 100
    table.to_csv(out / 'recovery_by_variant_data.csv', index_label='variant')
    overlap = success.sum(axis=1).value_counts().reindex(range(4), fill_value=0)
    overlap.rename('images').to_csv(out / 'recovery_overlap_data.csv', index_label='successful_variants')
    fig, ax = plt.subplots(figsize=(8, 5.3))
    bars = ax.bar(labels, table['recovery_percent'], color=COLOURS, width=.62)
    ax.bar_label(bars, labels=[f'{p:.2f}%\n(n = {n:,})' for p, n in
                              zip(table['recovery_percent'], table['recovered'])], padding=5)
    ax.set_ylim(0, max(table['recovery_percent']) * 1.3 + 1)
    ax.set_ylabel('Recovered baseline misses (%)')
    ax.set_title('Technical hand-detection recovery', loc='left', fontweight='bold', pad=15)
    ax.set_axisbelow(True)
    ax.yaxis.grid(alpha=.18)
    fig.text(.02, .015, f'All bars use {len(paired):,} original no_hand images as the denominator.\n'
             'A photo may recover under multiple filters, so filter percentages overlap and must not be added.\n'
             f'Any variant is the union: {counts.iloc[:3].sum():,} successful image-filter pairs correspond to\n'
             f"{counts.loc['any_variant_unique']:,} distinct recovered images. Recovery is technical; visual review is pending.", fontsize=8.5)
    fig.tight_layout(rect=(0, .17, 1, 1))
    save(fig, out, 'recovery_by_variant')

    letter_data = paired.assign(recovered=success.any(axis=1)).groupby('letter').agg(
        baseline_misses=('id', 'size'), recovered=('recovered', 'sum'))
    letter_data['recovery_percent'] = 100 * letter_data['recovered'] / letter_data['baseline_misses']
    letter_data = letter_data.sort_values('recovery_percent', ascending=False)
    letter_data.to_csv(out / 'recovery_by_letter_data.csv')
    fig, ax = plt.subplots(figsize=(8, 9))
    bars = ax.barh(letter_data.index, letter_data['recovery_percent'], color=COLOURS[0])
    ax.bar_label(bars, labels=[f'{r.recovery_percent:.1f}% ({r.recovered}/{r.baseline_misses})'
                              for r in letter_data.itertuples()], padding=4, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(letter_data['recovery_percent']) + 24)
    ax.set_xlabel('Recovered baseline misses (%)')
    ax.set_title('Recovery by letter: any successful variant', loc='left', fontweight='bold', pad=14)
    ax.set_axisbelow(True)
    ax.xaxis.grid(alpha=.18)
    fig.text(.02, .015, 'Labels show recovered / originally missed images within each letter.\n'
             'Each image is counted once. Letters have different numbers of baseline misses.', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, out, 'recovery_by_letter')

    example = paired.loc[(paired['filename'] == 'N47.jpg') & (paired['letter'] == 'N')]
    if not example.empty:
        row = example.iloc[0]
        processed = pd.read_csv(run / 'processed.csv', encoding='utf-8-sig')
        transformed = processed.loc[(processed['id'] == row['id']) &
                                    (processed['variant'] == 'dilatatie')].iloc[0]
        with (run / 'retry_landmarks.jsonl').open(encoding='utf-8') as handle:
            detection = next(r['detection'] for r in map(json.loads, handle)
                             if r['id'] == row['id'] and r['variant'] == 'dilatatie')
        with Image.open(resolve_path(row['original_path'])) as im:
            original = im.convert('RGB')
        with Image.open(resolve_path(transformed['processed_path'])) as im:
            dilated = im.convert('RGB')
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
        for ax, picture, title in zip(axes, [original, dilated, original],
                ['1  Original', '2  Dilation (3 x 3)', '3  Detection retry']):
            ax.imshow(picture)
            ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        points = detection.get('image_landmarks', [])
        axes[2].scatter([p[0]*original.width for p in points],
                        [p[1]*original.height for p in points], s=13,
                        c='#B6FF60', edgecolors='#17334F', linewidths=.35)
        axes[0].set_xlabel(row['status'])
        axes[1].set_xlabel('Processed input')
        axes[2].set_xlabel(row['dilatatie'] + ' - points on original')
        fig.text(.02, .025, 'N47 development example. Detector and thresholds are identical in both attempts.\n'
                 'The plotted points come from the stored dilation retry; visual plausibility requires review.', fontsize=9)
        fig.tight_layout(rect=(0, .16, 1, 1))
        save(fig, out, 'n47_comparison')
    else:
        print('N47 is absent from this run; its comparison was skipped.')
    inputs = ['per_image.csv', 'processed.csv', 'retry_landmarks.jsonl', 'settings.json', 'summary.csv']
    provenance = {'source_run': stored_path(run), 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
                  'input_sha256': {n: sha256((run/n).read_bytes()).hexdigest() for n in inputs},
                  'plot_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'figure_sources.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Figures and underlying tables saved to {out}')


if __name__ == '__main__':
    main()
