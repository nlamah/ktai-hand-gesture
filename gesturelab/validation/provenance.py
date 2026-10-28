"""Record code, environment and input immutability without collecting arbitrary env secrets."""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import shutil

from gesturelab.validation.report import sha256, write_json

THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONHASHSEED')


def _git(*arguments):
    result = subprocess.run(['git', *arguments], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def _archive_source(out, paths):
    for path in paths:
        destination = out / 'current_source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def environment_record(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'lightgbm', 'mediapipe', 'joblib', 'pillow'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    from threadpoolctl import threadpool_info
    code = [p for root in ('gesturelab', 'scripts', 'tests') for p in Path(root).rglob('*.py')]
    _archive_source(out, code)
    record = dict(commit=(_git('rev-parse', 'HEAD') or '').strip() or None,
        worktree_status=_git('status', '--short'), python=platform.python_version(), platform=platform.platform(),
        packages=packages, thread_environment={k: os.environ.get(k) for k in THREAD_KEYS},
        thread_pools=threadpool_info(), current_python_sha256={str(p): sha256(p) for p in code})
    diff = _git('diff', '--no-ext-diff', '--binary', 'HEAD', '--', 'gesturelab', 'scripts', 'tests', 'docs', 'README.md', 'requirements.txt', 'environment.yml')
    (out / 'worktree.patch').write_text(diff or '', encoding='utf-8')
    write_json(out / 'environment.json', record)
    return record


class InputWatch:
    def __init__(self):
        self.before = {}

    def add(self, path, digest=False):
        path = Path(path).resolve()
        if path.is_file() and path not in self.before:
            stat = path.stat()
            self.before[path] = (stat.st_size, stat.st_mtime_ns, sha256(path) if digest else None)

    def tree(self, root):
        for path in Path(root).rglob('*'):
            if path.is_file():
                self.add(path, path.suffix in ('.json', '.py'))

    def finish(self, report):
        changed = []
        for path, (size, stamp, digest) in self.before.items():
            if not path.exists() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, stamp):
                changed.append(str(path))
            elif digest is not None and sha256(path) != digest:
                changed.append(str(path))
        report.add('inputs.unchanged', 'Compare before/after size/mtime for source artifacts; SHA-256 additionally for watched JSON/source files.',
            'No source input files changed.', dict(watched=len(self.before), changed=changed), not changed,
            limitation='Large CSV/model/image bytes are not all hashed twice; audit writes are restricted to the new output directory.')
