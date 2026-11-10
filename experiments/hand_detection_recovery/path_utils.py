"""Resolve RQ2 export paths independently of a user's checkout location."""
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def stored_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # Explicit external source/run directories still work on their own host.
        return path.as_posix()


def gallery_url(value, run):
    path = resolve_path(value)
    try:
        relative = Path(os.path.relpath(path, Path(run).resolve())).as_posix()
    except ValueError:
        return path.as_uri()  # An explicitly selected external Windows drive.
    return quote(relative, safe='/')
