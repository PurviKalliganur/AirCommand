import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.features.landmarks import (
    create_hand_detector,
    extract_landmarks,
)
from src.inference.predictor import GesturePredictor
from src.inference.preprocessing import (
    create_frame_features,
    prepare_sequence,
)
from src.inference.sequence_buffer import SequenceBuffer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "final_motion_gru.pt"

BUFFER_LENGTH = 90
PREDICTION_INTERVAL = 5
MIN_FRAMES_FOR_PREDICTION = 15


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model checkpoint not found: {}".format(MODEL_PATH)
        )

    device = torch.device("cpu")
    predictor = GesturePredictor(MODEL_PATH, device)

    detector = create_hand_detector()
    buffer = SequenceBuffer(BUFFER_LENGTH)

    previous_position = None
    frame_count = 0
    last_prediction = "Show your hand"
    prediction_confidence_text = ""
    start_time = time.monotonic()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        detector.close()
        raise RuntimeError(
            "Could not open webcam. Close other apps using the camera and try again."
        )

    print("AirCommand webcam started.")
    print("Press Q to quit. Press C to clear the sequence.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = extract_landmarks(
                frame,
                detector,
                timestamp_ms,
            )

            if result.hand_detected:
                frame_features, previous_position = create_frame_features(
                    result.landmarks,
                    previous_position,
                )

                buffer.append(frame_features)

                if (
                    len(buffer) >= MIN_FRAMES_FOR_PREDICTION
                    and frame_count % PREDICTION_INTERVAL == 0
                ):
                    sequence = buffer.get_sequence()
                    sequence = prepare_sequence(sequence)

                    _, predicted_label = predictor.predict(sequence)
                    last_prediction = predicted_label
                    prediction_confidence_text = ""

                cv2.putText(
                    frame,
                    "Hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2,
                )

            else:
                previous_position = None
                buffer.clear()
                last_prediction = "Show your hand"
                prediction_confidence_text = ""

            cv2.putText(
                frame,
                "Gesture: " + last_prediction,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 180, 0),
                2,
            )

            cv2.putText(
                frame,
                "Frames: {}".format(len(buffer)),
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Q: Quit | C: Clear",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("AirCommand - Real-Time Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                buffer.clear()
                previous_position = None
                last_prediction = "Sequence cleared"

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
