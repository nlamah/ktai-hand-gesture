# Download an ASL dataset locally

This optional step downloads the dataset from the supplied Roboflow snippet:

- Workspace: `david-lee-d0rhs`
- Project: `american-sign-language-letters`
- Version: `1`
- Export format: `coco`

The downloader saves the images and annotations on your own computer. It does
not import them into the project's dataset format, replace the NGT reference
images, train a model or start the webcam.

## Download from Terminal

Run these commands from the project root. The `!pip` syntax in the supplied
snippet is for notebooks; use `python -m pip` in Terminal.

A separate download environment keeps the downloader independent of the
recognition project's dependencies:

```bash
conda create -n asl-download python=3.12 pip
conda activate asl-download
python -m pip install roboflow
python -m scripts.download_asl_dataset --out data/external/asl_letters_v1
```

If `asl-download` already exists, activate it instead of creating it again.
The script asks for a Roboflow API key without displaying what you type.
Use a new key if you previously shared one in chat or source code. Keys belong
in the prompt or the `ROBOFLOW_API_KEY` environment variable, not in the script.

The SDK downloads and extracts the COCO export into the selected directory.
The output directory must not already exist; this prevents silently reusing or
overwriting an earlier download. For a retry after an incomplete download,
choose another directory, for example `data/external/asl_letters_v1_retry`.

You can close the terminal after a successful download; the files remain on
your laptop. They can be inspected or copied without downloading them again.
Check the downloaded annotations and any included README/license files to
confirm which letter classes, image splits and reuse terms are provided.

Return to the recognition environment afterward:

```bash
conda activate ktai-hand-gesture
```

## Train after downloading

If the COCO files are already extracted in `data/external/asl_letters_v1`, skip
the download commands and use the recognition environment directly. Downloading
again or installing Roboflow is not needed for local import and training.

Continue with [ASL_TRAINING.md](ASL_TRAINING.md): import the COCO images and
labels, calculate landmarks and train both variants on the same source-grouped
split. The downloaded files are kept separate from the imported dataset.
For the extracted Kaggle alphabet folders, follow
[KAGGLE_TRAINING.md](KAGGLE_TRAINING.md) instead; they use a different importer.

The downloader dependency is optional and is not added to `requirements.txt`.
The download can be tested using mocks without a key or network access.

Roboflow's [`Version.download()` reference](https://roboflow.github.io/roboflow-python/core/version/)
documents the `location` argument and automatic ZIP extraction.
