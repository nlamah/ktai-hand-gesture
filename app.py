"""Run the hand-gesture proof of concept on an image or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp

if __package__:
    # Started as a package: python -m hand_gesture_poc.app
    from .gesture_classifier import classify_landmarks
else:
    # Started directly, for example with VS Code's "Run Python File" button.
    from gesture_classifier import classify_landmarks


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize five simple static gestures with MediaPipe Hands."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Path to an input image.")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hand_gesture_result.jpg"),
        help="Output path used in image mode.",
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6)
    args = parser.parse_args()
    if args.image is None and args.camera is None:
        args.camera = 0
    return args


def _draw_label(
    frame,
    hand_landmarks,
    label: str,
) -> None:
    height, width = frame.shape[:2]
    xs = [landmark.x for landmark in hand_landmarks.landmark]
    ys = [landmark.y for landmark in hand_landmarks.landmark]
    left = max(0, int(min(xs) * width))
    top = max(0, int(min(ys) * height))
    right = min(width - 1, int(max(xs) * width))
    bottom = min(height - 1, int(max(ys) * height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(width, height) / 900)
    thickness = max(2, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    padding = max(5, round(font_scale * 6))
    text_bottom = (
        top - padding
        if top > text_height + 3 * padding
        else top + text_height + 3 * padding
    )

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness)
    cv2.rectangle(
        frame,
        (left, text_bottom - text_height - 2 * padding),
        (left + text_width + 2 * padding, text_bottom + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (left + padding, text_bottom - padding),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame(frame, hands) -> tuple[object, list[str]]:
    """Detect, classify and annotate every hand in one BGR frame."""

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    labels: list[str] = []

    if not results.multi_hand_landmarks:
        return frame, labels

    handedness_results = results.multi_handedness or []
    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        prediction = classify_landmarks(hand_landmarks.landmark)
        handedness = "Hand"
        if index < len(handedness_results):
            handedness = handedness_results[index].classification[0].label

        label = f"{prediction.label} ({handedness})"
        labels.append(label)
        _draw_label(frame, hand_landmarks, label)
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        states = ", ".join(
            name for name, extended in prediction.extended_fingers.items() if extended
        ) or "none"
        print(f"{label}; extended fingers: {states}")

    return frame, labels


def run_image(args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
    ) as hands:
        annotated, labels = annotate_frame(image, hands)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"Could not write output image: {args.output}")

    if labels:
        print(f"Recognized: {', '.join(labels)}")
    else:
        print("No hand detected. Try a clearer image with the complete hand visible.")
    print(f"Annotated image saved to {args.output}")
    return 0


def run_camera(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check macOS camera permissions."
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a camera frame; stopping.")
                break

            frame = cv2.flip(frame, 1)
            annotated, _ = annotate_frame(frame, hands)
            cv2.imshow("Hand gesture POC - press Q or Esc to stop", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    capture.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_arguments()
    if args.image is not None:
        return run_image(args)
    return run_camera(args)


if __name__ == "__main__":
    raise SystemExit(main())
