"""Git provenance must preserve UTF-8 source text on Windows too."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

# main removed the historical gesturelab package. Retain this regression test
# for checkouts that still provide it, without restoring that retired codebase.
PROVENANCE = Path(__file__).resolve().parents[1] / 'gesturelab/validation/provenance.py'
if PROVENANCE.is_file():
    from gesturelab.validation.provenance import _git


@unittest.skipUnless(PROVENANCE.is_file(), 'Historical gesturelab provenance module is no longer part of main')
@unittest.skipUnless(shutil.which('git'), 'Git is required for provenance integration')
class GitProvenanceTests(unittest.TestCase):
    def test_staged_diff_preserves_unicode_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git', 'init', '--quiet', str(root)], check=True, capture_output=True)
            source = "example = 'Zo\u00eb \u2014 \u201d'\n"
            (root / 'example.py').write_text(source, encoding='utf-8')
            subprocess.run(['git', '-C', str(root), 'add', 'example.py'],
                           check=True, capture_output=True)
            diff = _git('-C', str(root), 'diff', '--cached', '--text')
            self.assertIsNotNone(diff)
            self.assertIn('+' + source.rstrip(), diff)


if __name__ == '__main__':
    unittest.main()
