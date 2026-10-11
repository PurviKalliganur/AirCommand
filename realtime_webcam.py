
import time
from dataclasses import dataclass
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

GESTURE_NAMES = {
    "B0A": "Pointing with one Finger",
    "B0B": "Pointing with two Fingers",
    "G01": "Click with one Finger",
    "G02": "Click with two Fingers",
    "G03": "Thumb Gesture",
    "G04": "Pointing down",
    "G05": "Left",
    "G06": "Right",
    "G07": "Open Hand",
    "G08": "Double click with one Finger",
    "G09": "Double click with two Fingers",
    "G10": "Hand Gesture 10",
}

PROJECT_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "configs" / "config.yaml").is_file()
    ),
    None,
)

if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate the AirCommand project root.")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "final_motion_gru.pt"
)

BUFFER_LENGTH = 192
PREDICTION_INTERVAL = 5
MIN_FRAMES_FOR_PREDICTION = 30
MAX_MISSED_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.0
PREDICTION_REQUIRED_CONSECUTIVE = 2
PREDICTION_COOLDOWN_FRAMES = 8


@dataclass
class PredictionState:
    raw_label: str | None = None
    last_raw_label: str | None = None
    confirmed_label: str | None = None
    consecutive_count: int = 0
    last_emitted_frame: int = -1


def reset_prediction_state(state: PredictionState | None = None) -> PredictionState:
    """Reset the stability tracker when tracking is lost or the buffer is cleared."""
    if state is None:
        return PredictionState()

    state.raw_label = None
    state.last_raw_label = None
    state.confirmed_label = None
    state.consecutive_count = 0
    state.last_emitted_frame = -1
    return state


def is_valid_feature_frame(frame: np.ndarray) -> bool:
    """Reject all-zero or non-finite feature vectors before model inference."""
    if frame is None:
        return False

    array = np.asarray(frame, dtype=np.float32)

    if array.ndim != 1 or array.shape[0] != 126:
        return False

    if not np.isfinite(array).all():
        return False

    if np.allclose(array, 0.0):
        return False

    return True


def update_prediction_state(
    raw_label: str | None,
    confidence: float | None,
    state: PredictionState,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    required_consecutive: int = PREDICTION_REQUIRED_CONSECUTIVE,
    cooldown_frames: int = PREDICTION_COOLDOWN_FRAMES,
    frame_index: int = 0,
):
    """Return a confirmed label only after a stable sequence of consistent predictions."""
    if raw_label is None:
        return None, reset_prediction_state(state)

    state.last_raw_label = raw_label

    if confidence is not None and confidence < confidence_threshold:
        state.raw_label = raw_label
        state.consecutive_count = 0
        return None, state

    if state.raw_label != raw_label:
        state.raw_label = raw_label
        state.consecutive_count = 1
    else:
        state.consecutive_count += 1

    if state.consecutive_count < required_consecutive:
        return None, state

    if (
        state.confirmed_label == raw_label
        and state.last_emitted_frame >= 0
        and frame_index - state.last_emitted_frame < cooldown_frames
    ):
        return None, state

    state.confirmed_label = raw_label
    state.last_emitted_frame = frame_index
    return raw_label, state


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model checkpoint not found: {}".format(MODEL_PATH)
        )

    device = torch.device("cpu")
    predictor = GesturePredictor(MODEL_PATH, device)

    detector = create_hand_detector()
    camera = None

    buffer = SequenceBuffer(BUFFER_LENGTH)

    previous_position = None
    frame_count = 0
    missed_frames = 0
    prediction_state = PredictionState()

    last_prediction = "Show your hand"
    start_time = time.monotonic()

    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open webcam. Close other apps using "
                "the camera and try again."
            )

        print("AirCommand webcam started.")
        print("Press Q to quit. Press C to clear the sequence.")

        window_name = "AirCommand - Real-Time Gesture Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
                missed_frames = 0

                frame_features, current_position = create_frame_features(
                    result.landmarks,
                    previous_position,
                )

                if not is_valid_feature_frame(frame_features):
                    previous_position = None
                    buffer.clear()
                    prediction_state = reset_prediction_state(prediction_state)
                    last_prediction = "Show your hand"
                    status_text = "Invalid hand input"
                    status_color = (0, 0, 255)
                else:
                    previous_position = current_position
                    buffer.append(frame_features)

                    if (
                        len(buffer) >= MIN_FRAMES_FOR_PREDICTION
                        and frame_count % PREDICTION_INTERVAL == 0
                    ):
                        sequence = buffer.get_sequence()
                        sequence = prepare_sequence(sequence)

                        _, predicted_label, confidence = (
                            predictor.predict_with_confidence(sequence)
                        )

                        confirmed_label, prediction_state = update_prediction_state(
                            raw_label=predicted_label,
                            confidence=confidence,
                            state=prediction_state,
                            confidence_threshold=CONFIDENCE_THRESHOLD,
                            required_consecutive=PREDICTION_REQUIRED_CONSECUTIVE,
                            cooldown_frames=PREDICTION_COOLDOWN_FRAMES,
                            frame_index=frame_count,
                        )

                        print(
                            f"Raw: {predicted_label} | "
                            f"Confidence: {confidence:.3f} | "
                            f"Buffered frames: {len(buffer)}"
                        )

                        if confirmed_label is not None:
                            last_prediction = GESTURE_NAMES.get(confirmed_label, confirmed_label)

                        status_text = f"Hand detected | Frames: {len(buffer)}"
                        status_color = (0, 200, 0)
                    else:
                        if (
                            len(buffer) >= MIN_FRAMES_FOR_PREDICTION
                            and frame_count % PREDICTION_INTERVAL == 0
                        ):
                            sequence = buffer.get_sequence()
                            sequence = prepare_sequence(sequence)

                            try:
                                _, predicted_label, confidence = predictor.predict_with_confidence(sequence)
                                raw_prediction = predicted_label
                            except ValueError:
                                raw_prediction = None
                                confidence = None

                            confirmed_label, prediction_state = update_prediction_state(
                                raw_label=raw_prediction,
                                confidence=confidence,
                                state=prediction_state,
                                confidence_threshold=CONFIDENCE_THRESHOLD,
                                required_consecutive=PREDICTION_REQUIRED_CONSECUTIVE,
                                cooldown_frames=PREDICTION_COOLDOWN_FRAMES,
                                frame_index=frame_count,
                            )

                            if confirmed_label is not None:
                                if confirmed_label != last_prediction:
                                    print(
                                        "Prediction changed: {} -> {}".format(
                                            last_prediction,
                                            confirmed_label,
                                        )
                                    )
                                last_prediction = GESTURE_NAMES.get(confirmed_label, confirmed_label)

                        status_text = "Hand detected"
                        status_color = (0, 200, 0)

            else:
                missed_frames += 1
                previous_position = None

                if missed_frames > MAX_MISSED_FRAMES:
                    buffer.clear()
                    prediction_state = reset_prediction_state(prediction_state)
                    last_prediction = "Show your hand"

                status_text = "Hand not detected"
                status_color = (0, 0, 255)

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                "Gesture: {}".format(last_prediction),
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

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                buffer.clear()
                previous_position = None
                missed_frames = 0
                prediction_state = reset_prediction_state(prediction_state)
                last_prediction = "Sequence cleared"
                print("Sequence cleared.")

    finally:
        if camera is not None:
            camera.release()

        detector.close()
        cv2.destroyAllWindows()
        print("AirCommand webcam stopped.")


if __name__ == "__main__":
    main()