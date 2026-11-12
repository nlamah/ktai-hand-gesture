"""Run the occlusion studies with project-relative source identifiers."""
import argparse
from importlib import import_module
from pathlib import Path
import sys

from baseline.runner import ROOT


STUDIES = ('gh_occlusion', 'gh_occlusion_rotated_test')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=STUDIES)
    args, remaining = parser.parse_known_args(argv)
    study = import_module('research.studies.' + args.study)
    original_file, original_argv = study.__file__, sys.argv
    try:
        # The numerical implementation and its checksum are unchanged. Only the
        # metadata identifier is relative, matching the distributed protocols.
        study.__file__ = str(Path(original_file).resolve().relative_to(ROOT))
        sys.argv = [study.__file__, *remaining]
        return study.main()
    finally:
        study.__file__, sys.argv = original_file, original_argv


if __name__ == '__main__':
    main()
