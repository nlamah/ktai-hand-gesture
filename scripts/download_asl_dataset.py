"""Download the supplied Roboflow ASL dataset as a standalone COCO export."""
import argparse
import getpass
import os
from pathlib import Path

WORKSPACE = 'david-lee-d0rhs'
PROJECT = 'american-sign-language-letters'
DEFAULT_OUTPUT = 'data/external/asl_letters_v1'


def download_dataset(destination, api_key=None, version=1):
    """Download images and annotations without importing or training on them."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Output directory already exists. Choose a new --out directory.')
    if version < 1:
        raise ValueError('Dataset version must be at least 1.')
    from roboflow import Roboflow
    key = api_key or os.environ.get('ROBOFLOW_API_KEY') or getpass.getpass('Roboflow API key (hidden): ')
    if not key.strip():
        raise ValueError('A Roboflow API key is required.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = Roboflow(api_key=key.strip())
        project = client.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(version).download('coco', location=str(destination), overwrite=False)
    except Exception as error:
        raise RuntimeError('Download failed. Check your API key, dataset access and connection. '
                           'If a partial output directory was created, use a new --out directory when retrying.') from error
    return Path(dataset.location)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUTPUT, help='New directory for the extracted COCO dataset.')
    parser.add_argument('--version', type=int, default=1)
    args = parser.parse_args()
    try:
        location = download_dataset(args.out, version=args.version)
    except ImportError:
        parser.exit(1, 'Install the optional downloader first: python -m pip install roboflow\n')
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')
    print(f'COCO dataset downloaded to: {location}')
    print('Images and annotations are saved locally. No dataset import or model training has run.')


if __name__ == '__main__':
    main()
