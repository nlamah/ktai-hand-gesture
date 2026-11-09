"""Run the local A-Z rejection baseline, or a small end-to-end demonstration."""
import argparse
from pathlib import Path
from shutil import copy2

from research.data import analyze_project, read_manifest
from research.detection_report import report
from research.kaggle import LETTERS, _split_folder, import_kaggle_dataset


ROOT = Path(__file__).resolve().parents[1]


def demo_source(source, destination):
    train = _split_folder(source, "asl_alphabet_train")
    test = _split_folder(source, "asl_alphabet_test")
    for letter in "ABC":
        for number in range(1, 3001, 100):
            relative = Path("asl_alphabet_train") / letter / f"{letter}{number}.jpg"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(train / letter / target.name, target)
        target = destination / "asl_alphabet_test" / f"{letter}_test.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(test / target.name, target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use 30 images each from A, B and C.")
    args = parser.parse_args()
    source = ROOT / "data/asl-alphabet-download"
    run = ROOT / "runs" / ("detection_demo" if args.demo else "detection_default")
    project = run / "project"
    if args.demo:
        source = demo_source(source, run / "source")
    else:
        train = _split_folder(source, "asl_alphabet_train")
        for letter in LETTERS:
            folder = train / letter
            expected = {f"{letter}{n}.jpg" for n in range(1, 3001)}
            if not folder.is_dir() or not expected.issubset(p.name for p in folder.iterdir()):
                raise SystemExit(f"Letter {letter} is incompleet. Pak eerst de volledige dataset uit.")

    state = read_manifest(project).get("import", {})
    if not state.get("complete"):
        print("Foto's importeren...", flush=True)
        import_kaggle_dataset(source, project, resume=bool(state), exclude_overlapping_test=True,
                              progress=lambda done, total: print(f"Import {done}/{total}", flush=True))
    print("Handlandmarks bepalen. Ctrl+C om te pauzeren.", flush=True)
    analyze_project(project, checkpoint_every=1000, cache_features=False)
    report(project, run)


if __name__ == "__main__":
    main()
