"""Export only fixed dataset inputs; never webcam captures, models or credentials."""
from pathlib import Path
import json
import zipfile
from baseline.check import digest
from baseline.runner import ROOT


def create_bundle(output, with_mirrors=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new filename: {output}')
    names = ['data/asl_kaggle/manifest.json', 'data/asl_kaggle/research/inputs.npz',
             'data/asl_kaggle/research/inputs_metadata.json', 'data/asl_kaggle/research/summary.json']
    if with_mirrors:
        parts = sorted((ROOT / 'runs/mirror_01/detections').glob('chunk_*.npz'))
        if not parts:
            raise FileNotFoundError('The mirror detection cache is missing.')
        names += [str(p.relative_to(ROOT)) for p in parts]
    for name in names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    record = {'version': 'shared-paper-inputs-v1', 'with_mirrors': with_mirrors,
              'files_sha256': {name: digest(ROOT / name) for name in names},
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
            'files': len(names), 'sha256': digest(output), 'with_mirrors': with_mirrors}
