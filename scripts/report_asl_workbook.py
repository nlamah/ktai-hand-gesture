"""Report workbook feature agreement on a validated COCO training landmark cache."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from gesturelab.features.asl_excel_review import (
    compare_sample, load_landmark_cache, load_reference_workbook,
)
from gesturelab.features.asl_live import _FEATURE_TITLES
from gesturelab.features.asl_threshold_tuning import _grouped_split, _source_group


def summarize(results):
    counts = Counter(item['status'] for result in results
                     for item in result['comparisons'].values())
    scored = sum(counts[key] for key in ('match', 'mismatch', 'unknown'))
    detected = sum(result.get('status') == 'ok' for result in results)
    exact = sum(all(item['status'] in ('match', 'not_scored')
                    for item in result['comparisons'].values()) for result in results)
    return {
        'images': len(results), 'detected_images': detected,
        'detection_statuses': dict(Counter(result.get('status', 'missing') for result in results)),
        'fully_matching_images': exact,
        'fully_matching_fraction_all_images': exact / len(results) if results else None,
        'fully_matching_fraction_detected_images': exact / detected if detected else None,
        'scored_feature_values': scored, 'feature_statuses': dict(counts),
        'feature_match_fraction_including_unknown': counts['match'] / scored if scored else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(cache, references, parameters, output, workbook_bytes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'feature_agreement.md').exists():
        raise FileExistsError('Agreement report already exists; choose a new output directory.')
    training, holdout, *_ = _grouped_split(cache['samples'])
    holdout_groups = {_source_group(sample) for sample in holdout}
    results = [compare_sample(sample, references, parameters) for sample in cache['samples']]
    partition = lambda result: 'holdout' if _source_group(result) in holdout_groups else 'fit'
    summary = {
        'workbook_sha256': sha256(workbook_bytes).hexdigest(),
        'dataset_fingerprint': cache.get('dataset_fingerprint'),
        'test_split_used': False,
        'notes': 'All rows are from the COCO train split. Fit and holdout share no source groups. '
                 'Holdout is reused development data, not an independent final test. '
                 'Unknown features count as failures; kan is excluded. '
                 'Slash values accept only listed alternatives. Agreement is not anatomical ground truth.',
        'all_training_data': summarize(results),
        'detected_training_data': summarize([result for result in results if result.get('status') == 'ok']),
        'fit_groups': summarize([result for result in results if partition(result) == 'fit']),
        'development_holdout': summarize([result for result in results if partition(result) == 'holdout']),
    }
    feature_rows, image_rows, letter_rows = [], [], []
    for result in results:
        counts = Counter(cell['status'] for cell in result['comparisons'].values())
        image_rows.append({
            'sample_id': result['id'], 'label': result['label'], 'partition': partition(result),
            'filename': result.get('filename'), 'detector_status': result.get('status'),
            'matches': counts['match'], 'mismatches': counts['mismatch'], 'unknown': counts['unknown'],
            'not_scored': counts['not_scored'],
            'fully_matches': counts['mismatch'] == 0 and counts['unknown'] == 0,
        })
    for label in sorted(references):
        group = [result for result in results if result['label'] == label]
        letter_summary = summarize(group)
        ranked = []
        for feature in references[label]:
            cells = [result['comparisons'][feature] for result in group]
            counts = Counter(cell['status'] for cell in cells)
            scored = counts['match'] + counts['mismatch'] + counts['unknown']
            wrong_values = Counter(cell['actual'] for cell in cells if cell['status'] == 'mismatch')
            row = {
                'letter': label, 'feature': feature, 'feature_name': _FEATURE_TITLES.get(feature, feature),
                'expected': references[label][feature]['raw'], 'images': len(group),
                'matches': counts['match'], 'mismatches': counts['mismatch'], 'unknown': counts['unknown'],
                'not_scored': counts['not_scored'],
                'failure_fraction': (counts['mismatch'] + counts['unknown']) / scored if scored else None,
                'most_common_wrong_values': json.dumps(dict(wrong_values.most_common(3)), ensure_ascii=False),
            }
            feature_rows.append(row)
            if scored:
                ranked.append(row)
        ranked.sort(key=lambda row: (-row['failure_fraction'], -row['mismatches'], row['feature']))
        letter_rows.append({'letter': label, **letter_summary, 'top_failures': ranked[:3]})
    summary['per_letter'] = letter_rows
    thumb_rows = [row for row in feature_rows if row['feature'] == 'thumb_mcp_shape']
    summary['binary_thumb'] = {key: sum(row[key] for row in thumb_rows)
                               for key in ('matches', 'mismatches', 'unknown')}
    thumb = summary['binary_thumb']
    thumb['known_match_fraction'] = thumb['matches'] / (thumb['matches'] + thumb['mismatches'])

    write_csv(output / 'per_letter_features.csv', feature_rows)
    write_csv(output / 'per_image_agreement.csv', image_rows)
    (output / 'agreement_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    total = summary['all_training_data']
    lines = [
        '# Feature-overeenkomst met spreadsheet versie 3', '',
        'De duim-MCP wordt uitsluitend beoordeeld als B-vormig of niet B-vormig. '
        'Duim-IP en de andere vingers behouden de drie buigstanden. Slashwaarden zijn expliciete alternatieven; '
        '`gebogen/gestrekt` accepteert dus niet automatisch `licht gebogen`. `kan` telt niet mee.', '',
        'Dit is afstelling van geometrische featuregrenzen op opgeslagen MediaPipe-landmarks, '
        'geen training van MediaPipe of een letterclassifier. Alleen de COCO-trainset is gebruikt; '
        'bronafbeeldingen en augmentaties blijven tijdens afstelling in dezelfde groep. '
        'De bestaande holdout is eerder gebruikte ontwikkeldata. De totaalscore bevat ook fitbeelden '
        'en is daarom een beschrijving van deze trainingsdata, geen onafhankelijke nauwkeurigheid.', '',
        '| Deel | Beelden | Detectie bruikbaar | Alle features goed | Featurewaarden goed* |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, title in [('all_training_data', 'Alle trainingsdata'), ('detected_training_data', 'Alleen bruikbare detecties'), ('fit_groups', 'Afstelgroepen'),
                       ('development_holdout', 'Ontwikkelholdout')]:
        row = summary[key]
        lines.append(f"| {title} | {row['images']} | {row['detected_images']} | "
                     f"{row['fully_matching_images']} ({row['fully_matching_fraction_all_images']:.1%}) | "
                     f"{row['feature_match_fraction_including_unknown']:.1%} |")
    lines += ['', f"Binaire duim-MCP: {thumb['matches']}/{thumb['matches'] + thumb['mismatches']} "
              f"({thumb['known_match_fraction']:.1%}) overeenkomst op beoordeelbare beelden; "
              f"{thumb['unknown']} onbekend.", '', '*Onbekende en door detectie ontbrekende featurewaarden tellen als niet voldaan. '
              'Niet beoordeelde (`kan`) waarden worden uitgesloten. Alle features goed vereist dat iedere '
              'beoordeelde feature matcht; afwijkingen kunnen ook door onjuiste landmarks of referenties ontstaan.', '',
              '## Meest afwijkende features per letter', '',
              'Per feature staan het aantal afwijkingen (A) en onbekende waarden (O). '
              'De rangschikking gebruikt (A + O) / beoordeelde beelden. Gelijke percentages zijn mogelijk. '
              'De CSV bevat alle features, verwachtingen en de meest voorkomende verkeerde categorieën.', '',
              '| Letter | Beelden | Alles goed | Top 1 | Top 2 | Top 3 |',
              '|---|---:|---:|---|---|---|']
    for row in letter_rows:
        top = [f"{item['feature_name']}: {item['failure_fraction']:.0%} "
               f"(A {item['mismatches']}, O {item['unknown']})" for item in row['top_failures']]
        lines.append(f"| {row['letter']} | {row['images']} | {row['fully_matching_images']} | " + ' | '.join(top) + ' |')
    lines += ['', '## Bestanden', '', '- `per_letter_features.csv`: alle letter-featurecombinaties.',
              '- `per_image_agreement.csv`: uitslag per trainingsbeeld, inclusief fit/holdout.',
              '- `agreement_summary.json`: totalen en herkomstgegevens.',
              '- `calibration_report.json`: afstelling en gekozen grenzen.', '']
    (output / 'feature_agreement.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/reports/asl_letters_v1_train_landmarks.json'))
    parser.add_argument('--dataset', type=Path, default=Path('data/external/asl_letters_v1/train'))
    parser.add_argument('--reference', type=Path, default=Path('assets/asl_feature_references.xlsx'))
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cache = load_landmark_cache(args.cache, args.dataset)
    if cache is None:
        raise ValueError('Landmark cache does not match dataset.')
    contents = args.reference.read_bytes()
    summary = build_report(cache, load_reference_workbook(contents),
                           json.loads(args.thresholds.read_text()), args.output_dir, contents)
    print(json.dumps({key: summary[key] for key in ('all_training_data', 'fit_groups', 'development_holdout')}, indent=2))


if __name__ == '__main__':
    main()
