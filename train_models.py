from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: numpy is required to run train_models.py. Install it and try again.") from exc

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: torch is required to run train_models.py. Install it and try again.") from exc

from build_lstm_model import (
    build_attack_occurrence_model,
    build_attack_location_model,
    build_state_estimation_model,
)
from train_utils import (
    EarlyStopping,
    get_device,
    iterate_minibatches,
    plot_training_history,
    report_best_scores,
    save_checkpoint,
)

DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

MODEL_CHECKPOINTS = {
    "occurrence": MODELS_DIR / "best_occurrence_model.pt",
    "location": MODELS_DIR / "best_location_model.pt",
    "state": MODELS_DIR / "best_state_model.pt",
}


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required preprocessed file: {path}")
    return np.load(path, allow_pickle=False)


def compute_binary_class_weight(labels: np.ndarray) -> dict[int, float]:
    flattened = np.asarray(labels).reshape(-1).astype(int)
    if flattened.size == 0:
        raise ValueError("Cannot compute class weights from an empty label array.")

    unique_classes, counts = np.unique(flattened, return_counts=True)
    total = flattened.size
    class_weight: dict[int, float] = {}
    for class_value, count in zip(unique_classes, counts):
        class_weight[int(class_value)] = float(total / (len(unique_classes) * count))
    for class_value in (0, 1):
        class_weight.setdefault(class_value, 1.0)
    return class_weight


def train_model(
    model_name: str,
    model: nn.Module,
    x_train: "torch.Tensor",
    y_train: "torch.Tensor",
    x_val: "torch.Tensor",
    y_val: "torch.Tensor",
    epochs: int,
    batch_size: int,
    metric_name: str,
    loss_fn,
    metric_fn,
    device: "torch.device",
    checkpoint_path: Path,
    input_dim: int,
) -> dict[str, list[float]]:
    print(f"\n=== Training {model_name.replace('_', ' ').title()} ===")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=10)

    history: dict[str, list[float]] = {"loss": [], "val_loss": [], metric_name: [], f"val_{metric_name}": []}

    x_val_dev = x_val.to(device)
    y_val_dev = y_val.to(device)

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_metric = 0.0
        n_batches = 0

        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_metric += metric_fn(preds.detach(), yb)
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        train_metric = running_metric / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_dev)
            val_loss = loss_fn(val_preds, y_val_dev).item()
            val_metric = metric_fn(val_preds, y_val_dev)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history[metric_name].append(train_metric)
        history[f"val_{metric_name}"].append(val_metric)

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - {metric_name}: {train_metric:.4f} "
            f"- val_loss: {val_loss:.4f} - val_{metric_name}: {val_metric:.4f}"
        )

        if early_stopping.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch} (best val_loss={early_stopping.best_loss:.4f})")
            break

    early_stopping.restore_best_weights(model)
    elapsed_seconds = time.perf_counter() - start_time
    print(f"Training time for {model_name.replace('_', ' ').title()}: {elapsed_seconds:.2f} seconds")

    save_checkpoint(model, input_dim, {"model_type": model_name}, checkpoint_path)
    print(f"Saved best checkpoint to: {checkpoint_path}")

    best_epoch, best_val_loss, best_secondary = report_best_scores(history, metric_name)
    print(f"Best epoch for {model_name.replace('_', ' ').title()}: {best_epoch}")
    print(f"Best validation loss for {model_name.replace('_', ' ').title()}: {best_val_loss:.6f}")
    if best_secondary is not None:
        print(f"Best validation {metric_name} for {model_name.replace('_', ' ').title()}: {best_secondary:.6f}")

    plot_path = plot_training_history(history, model_name, metric_name, PLOTS_DIR)
    print(f"Saved training curves to: {plot_path}")

    return history


# ---- metric functions (numpy-free, operate on torch tensors, return python floats) ----

def binary_accuracy(logits: "torch.Tensor", targets: "torch.Tensor") -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return float((preds == targets).float().mean().item())


def sparse_categorical_accuracy(logits: "torch.Tensor", targets: "torch.Tensor") -> float:
    preds = torch.argmax(logits, dim=-1)
    return float((preds == targets).float().mean().item())


def mean_absolute_error(preds: "torch.Tensor", targets: "torch.Tensor") -> float:
    return float(torch.mean(torch.abs(preds - targets)).item())


def main() -> None:
    try:
        device = get_device()
        print(f"Using device: {device}")

        print("Loading preprocessed arrays...")
        x_train = load_numpy_array(DATA_DIR / "X_train.npy")
        x_val = load_numpy_array(DATA_DIR / "X_val.npy")

        y_train_occur = load_numpy_array(DATA_DIR / "y_train_occur.npy")
        y_val_occur = load_numpy_array(DATA_DIR / "y_val_occur.npy")

        y_train_loc = load_numpy_array(DATA_DIR / "y_train_loc.npy")
        y_val_loc = load_numpy_array(DATA_DIR / "y_val_loc.npy")

        y_train_state = load_numpy_array(DATA_DIR / "y_train_state.npy")
        y_val_state = load_numpy_array(DATA_DIR / "y_val_state.npy")

        print(f"X_train shape: {x_train.shape}")
        print(f"X_val shape: {x_val.shape}")

        input_dim = x_train.shape[2]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        x_train_t = torch.from_numpy(x_train).float()
        x_val_t = torch.from_numpy(x_val).float()

        # --- Model 1: Attack Occurrence Detection ---
        print("Model 1: Attack Occurrence Detection")
        occurrence_model = build_attack_occurrence_model(input_dim)
        print(occurrence_model)

        class_weight = compute_binary_class_weight(y_train_occur)
        print(f"Class weights: {json.dumps(class_weight, sort_keys=True)}")
        # pos_weight rebalances the positive class the same way Keras' class_weight does.
        pos_weight = torch.tensor([class_weight[1] / class_weight[0]], dtype=torch.float32).to(device)
        occurrence_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        y_train_occur_t = torch.from_numpy(y_train_occur).float().reshape(-1)
        y_val_occur_t = torch.from_numpy(y_val_occur).float().reshape(-1)

        train_model(
            model_name="occurrence",
            model=occurrence_model,
            x_train=x_train_t,
            y_train=y_train_occur_t,
            x_val=x_val_t,
            y_val=y_val_occur_t,
            epochs=60,
            batch_size=512,
            metric_name="accuracy",
            loss_fn=occurrence_loss,
            metric_fn=binary_accuracy,
            device=device,
            checkpoint_path=MODEL_CHECKPOINTS["occurrence"],
            input_dim=input_dim,
        )

        # --- Model 2: Attack Location Detection ---
        print("Model 2: Attack Location Detection")
        location_model = build_attack_location_model(input_dim)
        print(location_model)

        location_loss = nn.CrossEntropyLoss()
        y_train_loc_t = torch.from_numpy(y_train_loc).long().reshape(-1)
        y_val_loc_t = torch.from_numpy(y_val_loc).long().reshape(-1)

        train_model(
            model_name="location",
            model=location_model,
            x_train=x_train_t,
            y_train=y_train_loc_t,
            x_val=x_val_t,
            y_val=y_val_loc_t,
            epochs=60,
            batch_size=512,
            metric_name="accuracy",
            loss_fn=location_loss,
            metric_fn=sparse_categorical_accuracy,
            device=device,
            checkpoint_path=MODEL_CHECKPOINTS["location"],
            input_dim=input_dim,
        )

        # --- Model 3: State Estimation ---
        print("Model 3: State Estimation")
        state_model = build_state_estimation_model(input_dim, output_dim=y_train_state.shape[1])
        print(state_model)

        state_loss = nn.MSELoss()
        y_train_state_t = torch.from_numpy(y_train_state).float()
        y_val_state_t = torch.from_numpy(y_val_state).float()

        train_model(
            model_name="state",
            model=state_model,
            x_train=x_train_t,
            y_train=y_train_state_t,
            x_val=x_val_t,
            y_val=y_val_state_t,
            epochs=60,
            batch_size=512,
            metric_name="mae",
            loss_fn=state_loss,
            metric_fn=mean_absolute_error,
            device=device,
            checkpoint_path=MODEL_CHECKPOINTS["state"],
            input_dim=input_dim,
        )

        print("\nTraining completed successfully.")
        print("Available saved artifacts:")
        print(f"- Best occurrence model: {MODEL_CHECKPOINTS['occurrence']}")
        print(f"- Best location model: {MODEL_CHECKPOINTS['location']}")
        print(f"- Best state model: {MODEL_CHECKPOINTS['state']}")
        print(f"- Training plots: {PLOTS_DIR}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
