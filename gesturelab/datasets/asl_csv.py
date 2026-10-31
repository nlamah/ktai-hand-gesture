"""Import an explicitly described external A–Z image dataset from CSV."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from gesturelab.datasets.data import (
    _store_image, atomic_json, decode_image, project_lock, read_manifest,
)
from gesturelab.features.asl_schema import LETTERS

CSV_COLUMNS = (
    "image_path", "label", "participant", "session", "condition", "role", "source_group",
)


def _read_entries(csv_path, images_root):
    entries = []
    groups = {}
    pixels = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(CSV_COLUMNS) - set(reader.fieldnames):
            missing = sorted(set(CSV_COLUMNS) - set(reader.fieldnames or ()))
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}.")
        for line, row in enumerate(reader, start=2):
            values = {key: (row.get(key) or "").strip() for key in CSV_COLUMNS}
            if any(not values[key] for key in CSV_COLUMNS):
                raise ValueError(f"CSV row {line} has an empty required value.")
            if values["label"] not in LETTERS:
                raise ValueError(f"CSV row {line} has an invalid A–Z label: {values['label']!r}.")
            if values["role"] not in ("development", "test"):
                raise ValueError(f"CSV row {line} role must be development or test.")
            source = (images_root / values["image_path"]).resolve()
            if not source.is_relative_to(images_root) or not source.is_file():
                raise ValueError(f"CSV row {line} image is missing or outside --images-root.")
            group_key = values["source_group"]
            group_value = values["role"]
            if group_key in groups and groups[group_key] != group_value:
                raise ValueError(f"CSV row {line} assigns source group {group_key!r} to conflicting metadata.")
            groups[group_key] = group_value

            data = source.read_bytes()
            image = decode_image(data)
            pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            prior = pixels.get(pixel_hash)
            current = (values["role"], values["label"], group_key)
            if prior is not None and prior != current:
                raise ValueError(f"CSV row {line} duplicates image pixels across labels, roles, or source groups.")
            pixels[pixel_hash] = current
            entries.append({
                **values, "source": source, "pixel_hash": pixel_hash,
                "original_sha256": hashlib.sha256(data).hexdigest(),
            })
    if not entries:
        raise ValueError("CSV contains no images.")
    return entries


def import_asl_csv(csv_file, images_root, project, source_name):
    """Create a separate pending-analysis dataset; this function never runs MediaPipe."""
    csv_file = Path(csv_file).resolve()
    images_root = Path(images_root).resolve()
    project = Path(project).resolve()
    if not csv_file.is_file():
        raise ValueError(f"CSV file not found: {csv_file}.")
    if not images_root.is_dir():
        raise ValueError(f"Image root not found: {images_root}.")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("Provide the actual external dataset name as --source-name.")
    if project.exists():
        raise FileExistsError("Output project already exists. Choose a new --project directory.")
    entries = _read_entries(csv_file, images_root)
    if not project.parent.is_dir():
        project.parent.mkdir(parents=True, exist_ok=True)
    with project_lock(project) as locked_project:
        manifest = read_manifest(locked_project)
        if manifest["samples"]:
            raise ValueError("Output project is not empty. Choose a new --project directory.")
        imported = {}
        duplicates = 0
        for entry in entries:
            previous = imported.get(entry["pixel_hash"])
            if previous is not None:
                duplicates += 1
                continue
            data = entry["source"].read_bytes()
            image = decode_image(data)
            sample = _store_image(
                locked_project, data, entry["source"].name, image, entry["pixel_hash"])
            sample.update(
                label=entry["label"], participant=entry["participant"],
                session=entry["session"], condition=entry["condition"],
                role=entry["role"], included=True,
                provenance={
                    "type": "external_asl_csv", "language": "ASL",
                    "source_name": source_name.strip(),
                    "source_group": entry["source_group"],
                    "source_file": entry["image_path"],
                    "source_sha256": entry["original_sha256"],
                },
            )
            manifest["samples"].append(sample)
            imported[entry["pixel_hash"]] = sample
        manifest["import"] = {
            "type": "external_asl_csv", "source_name": source_name.strip(),
            "csv_file": csv_file.name, "entries": len(entries),
            "duplicates_skipped": duplicates, "complete": True,
            "grouping_note": "source_group must identify related or duplicated source images.",
        }
        sources = locked_project / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(csv_file, sources / "asl_dataset.csv")
        atomic_json(locked_project / "manifest.json", manifest, compact=True)
    return {"added": len(imported), "duplicates": duplicates, "total": len(manifest["samples"])}
