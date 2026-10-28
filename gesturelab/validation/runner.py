"""Read-only audit orchestration; never calls training or a production report writer."""
from pathlib import Path
import shutil

from gesturelab.validation.report import AuditReport, ReportScope, new_output, read_json, sha256, write_json
from gesturelab.validation.features import audit_features
from gesturelab.validation.pipeline import audit_pipeline
from gesturelab.validation.provenance import environment_record, InputWatch
from gesturelab.validation.splits import (load_context, audit_metadata, audit_folds, audit_pixels,
                                         audit_snapshots, canonical_assignment)
from gesturelab.validation.scores import audit_variant_scores, audit_final_scores, compare_numbers
from gesturelab.validation.artifacts import audit_artifacts
from gesturelab.validation.metrics import read_csv


def experiments_in(paths):
    found = set()
    for item in paths:
        path = Path(item).resolve()
        if path.is_file() and path.name == 'split.json':
            found.add(path.parent)
        elif (path / 'split.json').is_file():
            found.add(path)
        else:
            found.update(p.parent for p in path.rglob('split.json'))
    return sorted(found)


def _attempt(report, name, operation, *args):
    try:
        return operation(report, *args)
    except FileNotFoundError as error:
        report.add(name, operation.__name__, 'Required evidence is available.', str(error),
                   next_step='Restore the complete archived experiment/data; do not claim this check passed.')
    except Exception as error:
        report.add(name, operation.__name__, 'Well-formed, consistent evidence.', f'{type(error).__name__}: {error}', False,
                   next_step='Investigate the source artifact without overwriting it.')
    return None


def _study_summary(report, context, variant, independent, tolerance):
    path = context['experiment'].parent / 'metrics.csv'
    if not path.exists():
        return
    rows = [r for r in read_csv(path) if r.get('variant') == variant and r.get('classifier') == context['plan']['model']]
    expected = {key: independent['pooled'][key] for key in ('accuracy', 'macro_f1', 'n')}
    expected.update({'cv_' + k: v for k, v in independent['folds']['summary'].items() if k.startswith(('accuracy', 'macro_f1'))})
    compare_numbers(report, 'scores.study_summary', expected, rows[0] if len(rows) == 1 else {}, [path], tolerance)


def _audit_one(report, context, out, tolerance):
    for operation in (audit_metadata, audit_folds, audit_pixels, audit_snapshots):
        _attempt(report, operation.__name__, operation, context)
    predictions = {}
    for folder in sorted(context['experiment'].iterdir()):
        if not (folder / 'training.json').is_file():
            continue
        print('Auditing ' + str(folder), flush=True)
        scope = ReportScope(report, folder.name)
        found = _attempt(scope, 'scores.available', audit_variant_scores, context, folder.name, out, tolerance)
        _attempt(scope, 'artifacts.available', audit_artifacts, context, folder.name)
        _attempt(scope, 'final.available', audit_final_scores, context, folder.name, out, tolerance)
        if found is not None:
            predictions[folder.name] = found[0]
            _attempt(scope, 'study_summary', _study_summary, context, folder.name, found[1], tolerance)
    return predictions


def compare_prediction_sets(report, tables):
    from collections import Counter
    mappings = {name: Counter((r['id'], r['label'], r.get('fold', '')) for r in rows) for name, rows in tables.items()}
    reference = next(iter(mappings.values()), None)
    same = bool(mappings) and all(value == reference for value in mappings.values())
    report.add('comparison.evaluated_mappings', 'Compare actual (ID, truth label, validation fold) multiplicities across variants.',
        'Every compared result evaluates the same mapping.', dict(variants=list(tables), same=same),
        same if len(tables) > 1 else None, limitation='This verifies evaluated rows, not unrecorded historical training execution.')


def _paper_reference(report, out, paper):
    if paper and Path(paper).is_file():
        target = out / 'references/shin2021.pdf'
        target.parent.mkdir()
        shutil.copyfile(paper, target)
        report.add('paper.source_available', 'Archive the supplied full article as read-only audit evidence.',
            'PDF source available for human formula/table review.', dict(sha256=sha256(target)), True, [target],
            limitation='Availability and Codex transcription are not independent scientific peer review.')
    else:
        report.add('paper.source_available', 'Locate an optional full PDF; formulas are mapped in PAPER_SHIN2021.md.',
            'Full article available for independent review.', 'No PDF archived by this command.',
            files=['https://doi.org/10.3390/s21175856'], next_step='Provide --paper with a local PDF and review the mapping.')


def _contexts(report, paths, cache, project, watch):
    result = []
    for path in paths:
        scope = ReportScope(report, str(path))
        try:
            context = load_context(path, cache, project)
        except (OSError, ValueError, KeyError) as error:
            scope.add('source.available', 'Read frozen plan and dataset manifest.', 'Complete evidence.', str(error))
            continue
        watch.add(context['project'] / 'manifest.json', digest=True)
        result.append(context)
    return result


def run_audit(experiments, out, project=None, paper=None, require_shared_splits=False, tolerance=1e-9):
    paths = experiments_in(experiments)
    if not paths:
        raise ValueError('No split.json found under the supplied experiment paths.')
    plans = [read_json(p / 'split.json') for p in paths]
    protected = paths + [Path(project or p['project']) for p in plans]
    out = new_output(out, protected)
    report, watch, cache = AuditReport('Research validation audit'), InputWatch(), {}
    for path in paths:
        watch.tree(path)
    environment_record(out)
    audit_features(report, out)
    audit_pipeline(report, out)
    _paper_reference(report, out, paper)
    contexts = _contexts(report, paths, cache, project, watch)
    all_predictions = {}
    for index, context in enumerate(contexts):
        scope = ReportScope(report, str(context['experiment']))
        evidence = out / 'experiments' / f'{index:02d}_{context["experiment"].parent.name}_{context["experiment"].name}'
        predictions = _audit_one(scope, context, evidence, tolerance)
        compare_prediction_sets(scope, predictions)
        all_predictions.update({str(context['experiment']) + '/' + k: v for k, v in predictions.items()})
    _compare_experiments(report, contexts, all_predictions, require_shared_splits)
    watch.finish(report)
    report.save(out, scope='Stored landmarks to classification; no detector/image-processing optimization or model search.',
                experiments=list(map(str, paths)), tolerance=tolerance, reproduction='Not run by this command; use scripts.reproduce_experiment.')
    return out


def _compare_experiments(report, contexts, predictions, required):
    assignments = [canonical_assignment(c['plan']) for c in contexts]
    same = len(assignments) > 1 and all(a == assignments[0] for a in assignments[1:])
    report.add('comparison.cross_experiment_assignments', 'Compare content: all train/validation ID sets, test IDs, labels and grouping; ignore irrelevant metadata/hash differences.',
        'Identical intended training and evaluation assignments when a shared comparison is requested.',
        dict(experiments=len(assignments), identical=same), same if required and len(assignments) > 1 else None,
        limitation='Canonical plans describe intent; actual historic fold training still requires execution evidence.')
    if required:
        compare_prediction_sets(report, predictions)
