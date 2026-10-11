import numpy as np

from realtime_webcam import (
    PredictionState,
    reset_prediction_state,
    update_prediction_state,
)
from src.inference.sequence_buffer import SequenceBuffer


def test_sequence_buffer_rejects_missing_hand_features():
    buffer = SequenceBuffer(10)

    valid = np.zeros(126, dtype=np.float32)
    valid[:63] = np.linspace(0.0, 1.0, 63, dtype=np.float32)

    assert buffer.append(valid) is True
    assert len(buffer) == 1

    zero_frame = np.zeros(126, dtype=np.float32)
    assert buffer.append(zero_frame) is False
    assert len(buffer) == 1


def test_prediction_state_requires_consecutive_predictions_and_cooldown():
    state = PredictionState()

    result, state = update_prediction_state(
        raw_label="G01",
        confidence=0.91,
        state=state,
        confidence_threshold=0.5,
        required_consecutive=2,
        cooldown_frames=3,
        frame_index=5,
    )
    assert result is None

    result, state = update_prediction_state(
        raw_label="G01",
        confidence=0.92,
        state=state,
        confidence_threshold=0.5,
        required_consecutive=2,
        cooldown_frames=3,
        frame_index=6,
    )
    assert result == "G01"

    result, state = update_prediction_state(
        raw_label="G01",
        confidence=0.93,
        state=state,
        confidence_threshold=0.5,
        required_consecutive=2,
        cooldown_frames=3,
        frame_index=8,
    )
    assert result is None


def test_tracking_loss_resets_prediction_state():
    state = PredictionState()
    state = reset_prediction_state(state)

    result, state = update_prediction_state(
        raw_label="G02",
        confidence=0.8,
        state=state,
        confidence_threshold=0.5,
        required_consecutive=1,
        cooldown_frames=2,
        frame_index=1,
    )
    assert result == "G02"

    state = reset_prediction_state(state)
    assert state.confirmed_label is None
    assert state.last_raw_label is None
    assert state.consecutive_count == 0
