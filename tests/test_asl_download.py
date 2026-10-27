"""Downloader behavior without installing Roboflow, using keys or accessing the network."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_asl_dataset import download_dataset, WORKSPACE, PROJECT


class ASLDownloadTests(unittest.TestCase):
    def test_download_selects_the_supplied_project_and_local_coco_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'download'
            client = Mock()
            version = client.workspace.return_value.project.return_value.version.return_value
            version.download.return_value = SimpleNamespace(location=str(output))
            factory = Mock(return_value=client)
            with patch.dict(sys.modules, {'roboflow': SimpleNamespace(Roboflow=factory)}):
                actual = download_dataset(output, api_key='unit-test-key')
            factory.assert_called_once_with(api_key='unit-test-key')
            client.workspace.assert_called_once_with(WORKSPACE)
            client.workspace.return_value.project.assert_called_once_with(PROJECT)
            client.workspace.return_value.project.return_value.version.assert_called_once_with(1)
            version.download.assert_called_once_with('coco', location=str(output.resolve()), overwrite=False)
            self.assertEqual(actual, output)

    def test_existing_output_is_not_silently_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'keep.txt'
            marker.write_text('existing data')
            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                download_dataset(tmp, api_key='unit-test-key')
            self.assertEqual(marker.read_text(), 'existing data')

    def test_empty_prompt_does_not_contact_roboflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            factory = Mock()
            with patch.dict(sys.modules, {'roboflow': SimpleNamespace(Roboflow=factory)}), \
                 patch.dict('os.environ', {'ROBOFLOW_API_KEY': ''}), \
                 patch('scripts.download_asl_dataset.getpass.getpass', return_value='   '):
                with self.assertRaisesRegex(ValueError, 'API key is required'):
                    download_dataset(Path(tmp) / 'download')
            factory.assert_not_called()

    def test_sdk_errors_do_not_expose_the_key_in_the_cli_error_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            factory = Mock(side_effect=RuntimeError('Network error with unit-test-key'))
            with patch.dict(sys.modules, {'roboflow': SimpleNamespace(Roboflow=factory)}):
                with self.assertRaises(RuntimeError) as error:
                    download_dataset(Path(tmp) / 'download', api_key='unit-test-key')
            self.assertNotIn('unit-test-key', str(error.exception))
            self.assertIn('Download failed', str(error.exception))


if __name__ == '__main__':
    unittest.main()
