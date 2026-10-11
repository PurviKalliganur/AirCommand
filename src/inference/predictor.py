from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

class GestureGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.0,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        final_hidden = hidden[-1]

        return self.classifier(final_hidden)

class GesturePredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
    ):
        self.device = device

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model_config = checkpoint["model_config"]

        self.model = GestureGRU(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.class_mapping = checkpoint["class_mapping"]

    def predict(self, sequence):
        """Predict the gesture class for one preprocessed sequence."""

        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.float32,
            device=self.device,
        )

        sequence_length = sequence_tensor.shape[0]

        sequence_tensor = sequence_tensor.unsqueeze(0)

        lengths = torch.tensor(
            [sequence_length],
            dtype=torch.long,
        )

        with torch.no_grad():
            outputs = self.model(
                sequence_tensor,
                lengths,
            )

        predicted_index = outputs.argmax(dim=1).item()

        predicted_label = self.class_mapping[predicted_index]

        return predicted_index, predicted_label

    
    def predict_with_confidence(self, sequence):
        """Predict a gesture and return its confidence."""

        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.ndim != 2 or sequence.shape[1] != 126:
            raise ValueError(
                f"Expected sequence shape (T, 126), got {sequence.shape}"
            )

        if sequence.shape[0] == 0:
            raise ValueError("Cannot predict an empty sequence.")

        if not np.isfinite(sequence).all():
            raise ValueError("Sequence contains NaN or infinite values.")

        sequence_tensor = torch.as_tensor(
            sequence,
            dtype=torch.float32,
            device=self.device,
        )

        sequence_tensor = sequence_tensor.unsqueeze(0)

        lengths = torch.tensor(
            [sequence.shape[0]],
            dtype=torch.long,
        )

        with torch.no_grad():
            logits = self.model(sequence_tensor, lengths)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)

        index = predicted_index.item()

        # Fallback prevents a missing mapping entry from crashing inference.
        label = self.class_mapping.get(index, f"Class {index}")

        return index, label, confidence.item()