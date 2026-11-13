"""Export fixed inputs and selected RQ3 evidence, excluding captures and models."""
from pathlib import Path
import json
import zipfile
from baseline.check import digest
from baseline.runner import ROOT


RESULT_SETS = ('feature_factorial_01', 'feature_diagnostics_01', 'direction_joint_01')


def create_results_bundle(output):
    """Export reported RQ3 tables/protocols, excluding models, photographs and captures."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new filename: {output}')
    required = ['protocol.json', 'metrics.csv', 'predictions.csv']
    for name in required:
        if not (ROOT / 'results/feature_factorial_01' / name).is_file():
            raise FileNotFoundError(f'Missing RQ3 evidence: {name}')
    files = sorted(p for name in RESULT_SETS for p in (ROOT / 'results' / name).rglob('*')
                   if p.is_file() and not p.is_symlink() and p.suffix in ('.csv', '.json'))
    names = [str(p.relative_to(ROOT)) for p in files]
    record = {'version': 'rq3-results-v1', 'files_sha256': {n: digest(ROOT / n) for n in names},
              'excluded': ['raw images', 'webcam captures', 'trained models', 'run checkpoints'],
              'usage': 'Extract at the repository root, then python -m baseline verify-results.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in names:
                archive.write(ROOT / name, name)
            archive.writestr('rq3_results_manifest.json', json.dumps(record, indent=2) + '\n')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output), 'bytes': output.stat().st_size, 'files': len(names),
            'sha256': digest(output), 'scope': 'RQ3 CSV/JSON evidence only; inputs supplied separately.'}


def verify_results():
    record = json.loads((ROOT / 'rq3_results_manifest.json').read_text())
    for name, expected in record['files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'results':
            raise ValueError(f'Invalid result path in bundle manifest: {name}')
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed RQ3 result: {name}')
    return {'status': 'passed', 'result_files_verified': len(record['files_sha256'])}


def create_bundle(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new filename: {output}')
    names = ['data/asl_kaggle/manifest.json', 'data/asl_kaggle/research/inputs.npz',
             'data/asl_kaggle/research/inputs_metadata.json', 'data/asl_kaggle/research/summary.json']
    for name in names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    record = {'version': 'shared-paper-inputs-v1', 'files_sha256': {name: digest(ROOT / name) for name in names},
              'excluded': ['webcam captures', 'raw images', 'trained fold models', 'credentials'],
              'usage': 'Extract from the repository root, then python -m baseline check. Read README.md first.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in names:
                # Already compressed arrays need not be compressed again.
                archive.write(ROOT / name, name, compress_type=zipfile.ZIP_STORED if name.endswith('.npz') else zipfile.ZIP_DEFLATED)
            archive.writestr('reproduction_inputs.json', json.dumps(record, indent=2) + '\n')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output), 'bytes': output.stat().st_size,
            'files': len(names), 'sha256': digest(output)}
