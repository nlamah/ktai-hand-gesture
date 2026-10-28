"""Independent ID, role, group and stored-pixel-hash audit of frozen plans."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from gesturelab.validation.report import read_json, sha256


def group_for(sample, grouping):
    if grouping == 'participant':
        return str(sample['participant'])
    if grouping == 'session':
        return str(sample['participant']), str(sample['session'])
    if grouping == 'source':
        value = sample.get('provenance', {}).get('source_group')
        if not value:
            raise ValueError('Missing source_group provenance.')
        return str(value)
    raise ValueError(f'Unsupported grouping: {grouping}')


def canonical_assignment(plan):
    """Compare contents, independent of timestamp, hashes, model or list ordering."""
    return dict(labels=sorted(plan['labels']), grouping=plan['grouping'],
        development_ids=sorted(plan['development_ids']), test_ids=sorted(plan['test_ids']),
        folds=[dict(train=sorted(f['train_ids']), valid=sorted(f['validation_ids'])) for f in plan['folds']])


def load_context(experiment, cache, project_override=None):
    experiment = Path(experiment).resolve()
    plan = read_json(experiment / 'split.json')
    project = Path(project_override or plan['project']).resolve()
    if project not in cache:
        manifest = read_json(project / 'manifest.json')
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache[project] = dict(manifest=manifest, fingerprint=digest,
                              records={s['id']: s for s in manifest['samples']})
    return dict(experiment=experiment, project=project, plan=plan, **cache[project])


def audit_metadata(report, context):
    p, records = context['plan'], context['records']
    dev, test, excluded = p['development_ids'], p['test_ids'], p.get('excluded_development_ids', [])
    all_ids = dev + test + excluded
    duplicates = [i for i, n in Counter(all_ids).items() if n != 1]
    absent = sorted(set(all_ids) - records.keys())
    report.add('data.identity', 'Compare all declared IDs with manifest identities and roles.',
        'Distinct known IDs in disjoint development/excluded/test partitions.',
        dict(duplicates=duplicates[:20], missing=absent[:20], n=len(all_ids)), not duplicates and not absent,
        [context['experiment'] / 'split.json', context['project'] / 'manifest.json'])
    wrong = []
    for ids, role, status in ((dev, 'development', 'ok'), (test, 'test', None), (excluded, 'development', 'failed')):
        for i in ids:
            s = records.get(i, {})
            if (s.get('role') != role or s.get('label') not in p['labels'] or not s.get('included')
                    or (status == 'ok' and s.get('status') != 'ok') or (status == 'failed' and s.get('status') == 'ok')):
                wrong.append(i)
    report.add('data.labels_roles', 'Check every declared photo label, inclusion, role and detection status.',
        'Declared development/test membership agrees with manifest.', dict(wrong_ids=wrong[:20]), not wrong)
    report.add('data.fingerprint', 'Recompute the documented canonical manifest hash.', p['dataset_fingerprint'],
        context['fingerprint'], p['dataset_fingerprint'] == context['fingerprint'],
        limitation='A hash detects change; it does not prove label truth or correct original collection.')
    report.add('data.manifest_ids', 'Count sample IDs before constructing the manifest index.',
        'One record per sample ID.', dict(records=len(records), rows=len(context['manifest']['samples'])),
        len(records) == len(context['manifest']['samples']))


def _fold_check(plan, records):
    issues, seen = [], Counter()
    dev, test = set(plan['development_ids']), set(plan['test_ids'])
    for index, fold in enumerate(plan['folds']):
        train, valid = fold['train_ids'], fold['validation_ids']
        seen.update(valid)
        if len(train) != len(set(train)) or len(valid) != len(set(valid)):
            issues.append(f'fold {index}: duplicate IDs')
        if set(train) & set(valid) or set(train) | set(valid) != dev or set(train) != dev - set(valid):
            issues.append(f'fold {index}: invalid training/validation partition')
        if (set(train) | set(valid)) & test:
            issues.append(f'fold {index}: reserved test leakage')
        if records:
            groups_train = {group_for(records[i], plan['grouping']) for i in train}
            groups_valid = {group_for(records[i], plan['grouping']) for i in valid}
            if groups_train & groups_valid:
                issues.append(f'fold {index}: group overlap')
            if set(plan['labels']) != {records[i]['label'] for i in train} or set(plan['labels']) != {records[i]['label'] for i in valid}:
                issues.append(f'fold {index}: missing label coverage')
    expected = Counter({i: int(plan.get('cv_repeats', 1)) for i in dev})
    if seen != expected:
        issues.append('validation coverage does not match declared CV repeats (default one)')
    return issues


def audit_folds(report, context):
    plan, records = context['plan'], context['records']
    try:
        issues = _fold_check(plan, records)
        dev_groups = {group_for(records[i], plan['grouping']) for i in plan['development_ids']}
        test_groups = {group_for(records[i], plan['grouping']) for i in plan['test_ids']}
        if dev_groups & test_groups:
            issues.append('development and reserved test share groups')
    except (KeyError, ValueError) as error:
        issues = [str(error)]
    report.add('splits.partitions_groups', 'Recompute ID complements, labels and group identities from manifest; count validation occurrences.',
        'Disjoint train/validation/test groups and complete declared CV coverage.', issues[:30], not issues,
        [context['experiment'] / 'split.json', context['project'] / 'manifest.json'])
    report.add('splits.historical_fit_rows', 'Look for actual fit traces, not just a planned split.',
        'Historical per-fold fit inputs prove actual training membership.',
        'Saved folds describe intended membership; original runs contain no per-fold fitted models or fit-input trace.',
        limitation='Neither final-fit CSV nor an OOF prediction proves which rows were actually used during each historical fit.',
        next_step='Use the instrumented one-variant reproduction; retain prospective fit traces for future runs.')
    report.add('validity.independent_test', 'Inspect reserved test roles and collection metadata.',
        'Independent final data and verified collection groups for generalization claims.',
        dict(test_photos=len(plan['test_ids']), grouping=plan['grouping'], note=plan.get('evaluation_note', '')),
        limitation='Filename blocks are not verified people/sessions. Disjoint participant codes alone do not verify real identities.',
        next_step='Describe results as development CV; independently review collection and final-test independence.')


def pixel_hashes(context):
    p = context['plan']
    ids = set(p['development_ids'] + p['test_ids'] + p.get('excluded_development_ids', []))
    result, missing = {}, []
    for i in ids:
        sample = context['records'].get(i, {})
        value = sample.get('pixel_sha256')
        if not value:
            try:
                from PIL import Image
                path = (context['project'] / sample['image_path']).resolve()
                if not path.is_relative_to(context['project']):
                    raise ValueError('Image path escapes dataset.')
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    value = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            except (OSError, KeyError, ValueError):
                missing.append(i)
        if value:
            result[i] = value
    return result, missing


def audit_pixels(report, context):
    hashes, missing = pixel_hashes(context)
    p = context['plan']
    issues = []
    for n, fold in enumerate(p['folds']):
        train = {hashes[i] for i in fold['train_ids'] if i in hashes}
        valid = {hashes[i] for i in fold['validation_ids'] if i in hashes}
        if train & valid:
            issues.append(dict(fold=n, shared_pixel_hashes=len(train & valid)))
    dev = {hashes[i] for i in p['development_ids'] + p.get('excluded_development_ids', []) if i in hashes}
    test = {hashes[i] for i in p['test_ids'] if i in hashes}
    if dev & test:
        issues.append(dict(development_test_shared_pixels=len(dev & test)))
    condition = False if issues else (None if missing else True)
    report.add('splits.exact_pixels', 'Compare stored RGB pixel hashes across each split; decode input images when hashes are absent.',
        'No exact pixel duplicates cross train/validation or development/test.',
        dict(issues=issues, missing_hashes=len(missing), known=len(hashes)), condition,
        [context['project'] / 'manifest.json'],
        limitation='Different IDs are not proof of different images. Existing pixel hashes were not recomputed from every file; no held-out test check when none is reserved.')
    report.add('splits.near_duplicates', 'No perceptual-image matching is performed in this limited audit.',
        'Independent source photographs, beyond exact byte/pixel identity.', 'Near-duplicates and collection identities were not examined.',
        limitation='Near-duplicate photos can remain across filename blocks.', next_step='Human/source review; avoid claims of person independence.')


def audit_snapshots(report, context):
    experiment = context['experiment']
    missing, changed, current = [], [], []
    package = Path(__file__).resolve().parents[1]
    for name, expected in context['plan'].get('code_hashes', {}).items():
        path = experiment / 'code_snapshot' / name
        if not path.is_file():
            missing.append(name)
        elif sha256(path) != expected:
            changed.append(name)
        if not (package / name).is_file() or sha256(package / name) != expected:
            current.append(name)
    report.add('provenance.snapshot', 'Verify every available frozen source hash against its snapshot.',
        'Archived code is complete and unchanged.', dict(missing=missing, changed=changed),
        False if changed else (None if missing or not context['plan'].get('code_hashes') else True),
        [experiment / 'code_snapshot'])
    report.add('provenance.current_code', 'Compare current files with archived code without bypassing experiment guards.',
        'Matching code for current-code reproduction/inference.', dict(different_files=current),
        None if current else True, limitation='A newer source version does not itself refute stored arithmetic; old-code execution is not established here.')
