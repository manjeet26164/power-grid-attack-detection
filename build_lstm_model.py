from __future__ import annotations

from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: torch is required to run build_lstm_model.py. Install it and try again."
    ) from exc


MODELS_DIR = Path("models")


class SharedLSTMBackbone(nn.Module):
    """Shared recurrent stack used by all three tasks.

    The first LSTM returns the full sequence so the second LSTM can learn
    temporal refinements on top of the intermediate representation.
    Dropout is used after each recurrent block to reduce overfitting.
    Mirrors the Keras backbone: LSTM(128, return_sequences=True) -> Dropout(0.2)
    -> LSTM(64, return_sequences=False) -> Dropout(0.2)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lstm_128 = nn.LSTM(input_size=input_dim, hidden_size=128, batch_first=True)
        self.dropout_1 = nn.Dropout(0.2)
        self.lstm_64 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout_2 = nn.Dropout(0.2)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm_128(x)
        out = self.dropout_1(out)
        out, (h_n, _) = self.lstm_64(out)
        # h_n[-1] is the final hidden state of lstm_64, equivalent to
        # Keras' return_sequences=False output (last timestep).
        out = h_n[-1]
        out = self.dropout_2(out)
        return out


class AttackOccurrenceModel(nn.Module):
    """MODEL 1 - binary attack occurrence. Outputs a raw logit;
    use BCEWithLogitsLoss (sigmoid is applied internally by the loss / at inference)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = SharedLSTMBackbone(input_dim)
        self.dense_16 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.2)
        self.occurrence_output = nn.Linear(16, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.backbone(x)
        out = self.relu(self.dense_16(out))
        out = self.dropout_3(out)
        out = self.occurrence_output(out)
        return out.squeeze(-1)  # raw logit, shape (batch,)


class AttackLocationModel(nn.Module):
    """MODEL 2 - 21-way location classification. Outputs raw logits;
    use CrossEntropyLoss (softmax applied internally by the loss)."""

    def __init__(self, input_dim: int, num_classes: int = 21) -> None:
        super().__init__()
        self.backbone = SharedLSTMBackbone(input_dim)
        self.dense_64 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.2)
        self.location_output = nn.Linear(64, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.backbone(x)
        out = self.relu(self.dense_64(out))
        out = self.dropout_3(out)
        out = self.location_output(out)
        return out  # raw logits, shape (batch, num_classes)


class StateEstimationModel(nn.Module):
    """MODEL 3 - regression back to the full 20-line capacity vector.
    Linear output head, trained with MSELoss."""

    def __init__(self, input_dim: int, output_dim: int = 20) -> None:
        super().__init__()
        self.backbone = SharedLSTMBackbone(input_dim)
        self.dense_64 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.2)
        self.state_output = nn.Linear(64, output_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.backbone(x)
        out = self.relu(self.dense_64(out))
        out = self.dropout_3(out)
        out = self.state_output(out)
        return out  # shape (batch, output_dim)


def build_attack_occurrence_model(input_dim: int) -> AttackOccurrenceModel:
    return AttackOccurrenceModel(input_dim)


def build_attack_location_model(input_dim: int, num_classes: int = 21) -> AttackLocationModel:
    return AttackLocationModel(input_dim, num_classes)


def build_state_estimation_model(input_dim: int, output_dim: int = 20) -> StateEstimationModel:
    return StateEstimationModel(input_dim, output_dim)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    try:
        # Default input_dim matches the repo's default selected_lines=6 (Po=0.3 over 20 lines).
        input_dim = 6

        occurrence_model = build_attack_occurrence_model(input_dim)
        location_model = build_attack_location_model(input_dim)
        state_model = build_state_estimation_model(input_dim)

        print("MODEL 1 - Attack Occurrence Detection")
        print(occurrence_model)
        print(f"Trainable parameters: {count_params(occurrence_model):,}")
        print()

        print("MODEL 2 - Attack Location Detection")
        print(location_model)
        print(f"Trainable parameters: {count_params(location_model):,}")
        print()

        print("MODEL 3 - State Estimation")
        print(state_model)
        print(f"Trainable parameters: {count_params(state_model):,}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nModel classes are defined in build_lstm_model.py.")
        print("They are instantiated directly (no separate architecture JSON needed,")
        print("unlike the Keras version) - train_models.py imports these classes.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
