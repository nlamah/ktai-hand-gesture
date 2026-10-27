"""Align comparable predictions and enforce frozen IDs, labels and fold assignments."""
from itertools import combinations

import pandas as pd


def validate_predictions(predictions, plan, records, final_test=False):
    """Never silently intersect results or accept altered validation membership."""
    ids = plan['test_ids'] if final_test else plan['development_ids']
    expected = {sample_id: str(records[sample_id]['label']) for sample_id in ids}
    folds = {sample_id: index for index, fold in enumerate(plan['folds']) for sample_id in fold['validation_ids']}
    for variant, table in predictions.items():
        if table.id.duplicated().any() or set(table.id) != set(ids):
            raise ValueError(f'{variant}: predictions must match the frozen unique photo IDs.')
        if any(str(row.label) != expected[row.id] for row in table.itertuples()):
            raise ValueError(f'{variant}: prediction labels differ from the frozen ground truth.')
        if not final_test and ('fold' not in table or any(row.fold != folds[row.id] for row in table.itertuples())):
            raise ValueError(f'{variant}: predictions do not match the frozen validation folds.')


def pair_predictions(predictions):
    """Preserve the original two-variant CSV contract."""
    if len(predictions) != 2:
        raise ValueError('Choose exactly two distinct training input variants.')
    first, second = predictions.values()
    if set(first.id) != set(second.id) or any(table.id.duplicated().any() for table in predictions.values()):
        raise ValueError('The two results do not have exactly the same unique photo IDs.')
    paired = first[['id', 'label', 'prediction']].rename(columns={'prediction': 'first'}).merge(
        second[['id', 'label', 'prediction']].rename(columns={'prediction': 'second'}),
        on=['id', 'label'], validate='one_to_one')
    if len(paired) != len(first):
        raise ValueError('Ground-truth labels differ between the two results.')
    paired['first_correct'] = paired.label == paired['first']
    paired['second_correct'] = paired.label == paired['second']
    return paired


def align_predictions(predictions):
    """One row per photo, with a prediction/correctness column for each variant."""
    if len(predictions) == 2:
        return pair_predictions(predictions)
    first = next(iter(predictions.values()))
    columns = ['id', 'label'] + (['fold'] if 'fold' in first else [])
    aligned = first[columns].set_index('id')
    for variant, table in predictions.items():
        values = table.set_index('id').prediction
        aligned[variant + '_prediction'] = values
        aligned[variant + '_correct'] = aligned.label == values
    return aligned.reset_index()


def pairwise_scores(predictions, summary):
    """Paired error counts for every pair; these do not establish significance."""
    scores = summary.set_index('variant').macro_f1
    rows = []
    for first, second in combinations(predictions, 2):
        paired = pair_predictions({first: predictions[first], second: predictions[second]})
        rows.append({
            'first': first, 'second': second,
            'macro_f1_difference_second_minus_first': float(scores[second] - scores[first]),
            'only_second_correct': int((paired.second_correct & ~paired.first_correct).sum()),
            'only_first_correct': int((paired.first_correct & ~paired.second_correct).sum()),
            'both_wrong': int((~paired.first_correct & ~paired.second_correct).sum()),
        })
    return pd.DataFrame(rows)
