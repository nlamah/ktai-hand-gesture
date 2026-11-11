"""Create research tables and a visual review gallery from completed run exports."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from verify_results import read_csv, verify
from path_utils import gallery_url


VARIANTS = ('erosie', 'dilatatie', 'canny_overlay')


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def gallery(run, rows, processed, detections):
    cards = []
    for row in rows:
        pictures = []
        for variant in VARIANTS:
            key = row['id'], variant
            detection = detections[key]['detection']
            pictures.append(dict(variant=variant, status=row[variant],
                                 url=gallery_url(processed[key]['processed_path'], run),
                                 points=detection.get('image_landmarks', [])))
        cards.append(dict(id=row['id'], filename=row['filename'], letter=row['letter'],
                          original=gallery_url(row['original_path'], run), pictures=pictures,
                          recovered=bool(row['recovered_variants']),
                          still_no_hand=all(row[v] == 'no_hand' for v in VARIANTS)))
    data = json.dumps(cards).replace('<', '\\u003c')
    template = Path(__file__).with_name('gallery_template.html').read_text(encoding='utf-8')
    (run / 'review_gallery.html').write_text(template.replace('__DATA__', data), encoding='utf-8')


def analyze(run):
    verification = verify(run)
    baseline = read_csv(run, 'baseline.csv')
    rows = read_csv(run, 'per_image.csv')
    processed = {(r['id'], r['variant']): r for r in read_csv(run, 'processed.csv')}
    with (run / 'retry_landmarks.jsonl').open(encoding='utf-8') as handle:
        detections = {(r['id'], r['variant']): r for r in map(json.loads, handle)}
    fields = ['id', 'filename', 'letter', 'original_path', 'status', *VARIANTS, 'recovered_variants']
    recovered = [r for r in rows if r['recovered_variants']]
    still_no_hand = [r for r in rows if all(r[v] == 'no_hand' for v in VARIANTS)]
    write_csv(run / 'recovered.csv', recovered, fields)
    write_csv(run / 'still_no_hand.csv', still_no_hand, fields)
    outcomes = []
    for variant in VARIANTS:
        counts = Counter(r[variant] for r in rows)
        outcomes.append(dict(variant=variant, total=len(rows), **{s: counts[s] for s in
                            ('ok', 'no_hand', 'multiple_hands', 'invalid_geometry')}))
    write_csv(run / 'retry_outcomes.csv', outcomes,
              ['variant', 'total', 'ok', 'no_hand', 'multiple_hands', 'invalid_geometry'])
    by_letter = []
    for letter in sorted({r['letter'] for r in baseline}):
        originals = [r for r in baseline if r['letter'] == letter]
        group = [r for r in rows if r['letter'] == letter]
        accepted = sum(r['status'] == 'ok' for r in originals)
        unique = sum(bool(r['recovered_variants']) for r in group)
        by_letter.append(dict(letter=letter, total=len(originals), baseline_ok=accepted,
                              baseline_misses=len(group),
                              **{v: sum(r[v] == 'ok' for r in group) for v in VARIANTS},
                              unique_recovered=unique,
                              still_no_hand=sum(all(r[v] == 'no_hand' for v in VARIANTS) for r in group),
                              recovery_percent=round(100*unique/len(group), 3) if group else 0,
                              coverage_before=round(100*accepted/len(originals), 3),
                              coverage_after=round(100*(accepted+unique)/len(originals), 3)))
    write_csv(run / 'by_letter.csv', by_letter, list(by_letter[0]))
    review = []
    for row in rows:
        for variant in VARIANTS:
            review.append(dict(id=row['id'], filename=row['filename'], letter=row['letter'],
                               variant=variant, retry_status=row[variant],
                               original_path=row['original_path'],
                               processed_path=processed[row['id'], variant]['processed_path'],
                               judgement='', visible_change='', possible_explanation='',
                               alternative_explanation='', reviewer=''))
    write_csv(run / 'visual_review_template.csv', review,
              ['id', 'filename', 'letter', 'variant', 'retry_status', 'original_path',
               'processed_path', 'judgement', 'visible_change', 'possible_explanation',
               'alternative_explanation', 'reviewer'])
    n47 = [r for r in baseline if r['filename'] == 'N47.jpg' and r['letter'] == 'N']
    write_csv(run / 'n47_example.csv', [r for r in rows if r['filename'] == 'N47.jpg'], fields)
    other = [r for r in rows if r['filename'] != 'N47.jpg']
    total, accepted = len(baseline), verification['originally_accepted']
    text = ['# Detection recovery results', '', f'Source run: `{run.name}`.',
            'Evidence checks passed against the baseline, paired outcomes and landmark exports.', '',
            '## Development example', '']
    if n47:
        text.append(f"N47 baseline status: `{n47[0]['status']}`. See `n47_example.csv` and the gallery's N47 button.")
    else:
        text.append('N47 is absent from this run. No N47 result is inferred from other runs.')
    text.extend(['', '## Dataset results', '', f'Original images: {total}. Baseline accepted: {accepted}. '
                 f'Initially missed: {len(rows)}. Unique technical recoveries: {len(recovered)}.',
                 f'Coverage: {100*accepted/total:.2f}% before and {100*(accepted+len(recovered))/total:.2f}% '
                 'after accepting any successful variant.',
                 f'Still no hand in all three variants: {len(still_no_hand)}. '
                 f'Without an accepted retry, including other rejection states: {len(rows)-len(recovered)}.',
                 f"Excluding N47: {sum(bool(r['recovered_variants']) for r in other)} technical recoveries "
                 f'among {len(other)} baseline misses. Other previously inspected pilot images remain included.', '',
                 '| Variant | Recovered | Still no hand | Multiple hands | Invalid geometry |',
                 '| --- | ---: | ---: | ---: | ---: |'])
    for row in outcomes:
        text.append(f"| {row['variant']} | {row['ok']} | {row['no_hand']} | {row['multiple_hands']} | {row['invalid_geometry']} |")
    text.extend(['', '## Interpreting why an outcome changed', '',
                 'Use `review_gallery.html` to compare original pixels, each processed input and estimated points. '
                 'Copy `visual_review_template.csv` to a separate working file before recording judgements. '
                 'The template is regenerated by analysis. Use plausible, implausible, uncertain or not_applicable '
                 'for the landmark judgement. Describe visible changes before proposing an explanation.',
                 'A paired status change measures the effect of this complete transformation on this detector. '
                 'It does not isolate which contour, pixel or contrast change caused the model response. '
                 'Mechanistic explanations require additional controlled comparisons. '
                 'Visual plausibility and classifier accuracy are not measured by these automatic tables.', '',
                 '## Evidence files', '',
                 '`recovered.csv`: filenames and successful variants. `still_no_hand.csv`: zero detections for '
                 'every variant. `unrecovered.csv`: all photos without an accepted retry. '
                 '`retry_outcomes.csv`: rejection statuses by variant. `by_letter.csv`: coverage and recovery '
                 'per letter. `processed.csv`: processed image paths and hashes. '
                 '`retry_landmarks.jsonl`: exact retry coordinates. `settings.json`: recorded configuration.'])
    (run / 'analysis.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    gallery(run, rows, processed, detections)
    print(f"Analysis written to {run / 'analysis.md'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run.resolve())
