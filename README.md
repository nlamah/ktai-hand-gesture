# Hand gesture recognition

A Python project for exploring hand gesture recognition in images and live
webcam video. It combines hand landmark detection, gesture classification and
visual feedback, and provides a starting point for experiments with different
recognition methods.

## Current prototype

The prototype described here uses MediaPipe Hands to detect 21 landmarks per
hand. Geometric rules classify five static poses: open palm, fist, pointing,
victory and thumbs up. OpenCV displays the landmarks, a bounding box and the
predicted label.

The classifier currently uses rules rather than a model trained on a custom
dataset. Hand-pose classification covers a limited part of sign language;
recognizing a full sign language also requires movement, grammar and context.

## Setup

### 1. Install Conda (once)

If you already have Anaconda or Miniconda, check it with `conda --version`
and continue to step 2. You do not need to install both.

Otherwise, install Miniconda using the
[official installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install).
On a Mac with Apple Silicon (M1/M2/M3/M4 or later), you can run:

```bash
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o /tmp/miniconda-installer.sh
bash /tmp/miniconda-installer.sh
```

Follow the installer prompts, accept the license if you agree, and answer
**yes** when asked to initialize Conda. Close and reopen Terminal, then run:

```bash
conda --version
```

For Intel Macs, Windows or Linux, choose the matching installer from the
installation page above. On Windows, run the project commands in Anaconda Prompt.
If activation asks you to initialize your shell, run `conda init zsh` for the
default macOS shell, then close and reopen Terminal.

### 2. Open the project folder

Download and extract this repository, or clone it if Git is installed:

```bash
git clone https://github.com/nlamah/ktai-hand-gesture.git
cd ktai-hand-gesture
```

If you already have the project, open Terminal in its existing folder instead.


All remaining commands run from this folder, which contains `app.py`,
`environment.yml` and `requirements.txt`.

### 3. Create the environment and install dependencies

```bash
conda env create -f environment.yml
conda activate hand-gesture
```

Wait for environment creation, including its pip installation step, to finish
successfully before activating it. This installs Python 3.12 and the packages
pinned in `requirements.txt`: MediaPipe 0.10.21, NumPy 1.26.4 and
OpenCV (`opencv-contrib-python`) 4.11.0.86, along with their dependencies.
No separate Python or OpenCV installation is needed.

If the `hand-gesture` environment already exists, update it instead:

```bash
conda env update -n hand-gesture -f environment.yml
conda activate hand-gesture
```

### 4. Verify the installation

```bash
python -c "import sys; print(sys.executable)"
python -c "import cv2, mediapipe as mp, numpy; print('MediaPipe', mp.__version__, 'OpenCV', cv2.__version__, 'NumPy', numpy.__version__); print(mp.solutions.hands.Hands)"
python app.py --help
```

The interpreter path should belong to `hand-gesture`. The import check should
print the versions and the MediaPipe `Hands` class without a traceback.
The package pins preserve the legacy `mp.solutions.hands` API used by this app.

## Run with a webcam

Each time you open a new terminal, enter the project folder and activate the
environment before launching:

```bash
conda activate hand-gesture
python app.py --camera 0
```

Adjust the `cd` path if you saved the project elsewhere. A window opens with
live webcam video. Hold your complete hand in view to see landmarks and a
predicted gesture. Click the video window and press **Q** or **Esc** to stop.
Running `python app.py` also selects camera `0` by default.

On macOS, allow camera access for the terminal or editor when prompted. If
access was denied, enable it in **System Settings > Privacy & Security > Camera**
and restart the terminal or editor. If the wrong camera opens, try:

```bash
python app.py --camera 1
```

In VS Code, use **Python: Select Interpreter** to select the `hand-gesture`
environment, then open `app.py` and use **Run Python File**.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'cv2'`**: activate the environment,
  then install the project requirements from the project folder:

  ```bash
  conda activate hand-gesture
  python -m pip install -r requirements.txt
  ```

  Repeat the import check above. `cv2` is supplied by `opencv-contrib-python`;
  do not install another OpenCV variant alongside it.
- **`python -m app.py` fails**: use `python app.py --camera 0`.
  The `-m` option expects a module name without `.py`; `python -m app --camera 0`
  also works from this folder.
- **`pip check` says “No broken requirements found” but imports fail**:
  that check validates installed package metadata; it does not check that this
  project's requirements have been installed. Use the import check above.
- **`pip check` reports that MediaPipe is not supported on this platform**:
  this warning was observed on the development Mac even though imports and
  `app.py --help` succeeded. Those checks do not verify webcam operation.
  Include the warning, Python version and import-check output when reporting
  an installation problem.
- **The camera does not open**: check camera permissions, close other apps
  using the webcam, and try another camera index such as `--camera 1`.

## Run on an image

From the project folder with the environment activated:

```bash
python app.py --image path/to/hand.jpg --output hand_gesture_result.jpg
```

Replace `path/to/hand.jpg` with an actual image showing the complete hand.

## Run tests

From the project folder with the environment activated:

```bash
python -m unittest discover -s tests -v
```

The tests use Python's built-in `unittest` module.

## How it works

1. Capture an image or webcam frame.
2. Detect hand landmarks with MediaPipe.
3. Use joint angles and distances to estimate which fingers are extended.
4. Map the finger states to a gesture label.
5. Draw the landmarks and prediction with OpenCV.

Recognition can be affected by camera angle, hidden fingers and gestures with
similar finger positions. The current classifier does not recognize movement
across frames.

## Further development

Possible extensions include collecting a labelled dataset, normalizing landmark
coordinates, training a classifier and comparing it with the rule-based
baseline. Evaluation should separate participants between training and test
sets, report per-class precision, recall and F1, and examine performance across
different lighting conditions, distances and viewing angles.

Dynamic gestures require sequences of landmarks rather than individual frames.
Add training or analysis dependencies when those features are implemented.

