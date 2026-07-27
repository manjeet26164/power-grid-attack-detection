from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: numpy is required. Install it and try again.") from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: torch is required. Install it and try again.") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: matplotlib is required. Install it and try again.") from exc


def get_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Mirrors tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: dict[str, Any] | None = None
        self.counter = 0
        self.stopped_epoch = 0

    def step(self, val_loss: float, model: "torch.nn.Module", epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def restore_best_weights(self, model: "torch.nn.Module") -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def save_checkpoint(model: "torch.nn.Module", input_dim: int, extra: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            **extra,
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def iterate_minibatches(
    x: "torch.Tensor", y: "torch.Tensor", batch_size: int, shuffle: bool = True
):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def plot_training_history(
    history: dict[str, list[float]], model_name: str, metric_name: str, plots_dir: Path
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plots_dir / f"{model_name}_training_curves.png"

    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["loss"], label="Training loss", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss", color="tab:orange")
    axes[0].set_title(f"{model_name.replace('_', ' ').title()} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    metric_key = metric_name
    val_metric_key = f"val_{metric_name}"
    if metric_key in history and val_metric_key in history:
        axes[1].plot(epochs, history[metric_key], label=f"Training {metric_name}", color="tab:green")
        axes[1].plot(epochs, history[val_metric_key], label=f"Validation {metric_name}", color="tab:red")
        axes[1].set_title(f"{model_name.replace('_', ' ').title()} {metric_name.upper()}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name.upper())
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, f"Metric '{metric_name}' not available", ha="center", va="center")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def report_best_scores(history: dict[str, list[float]], metric_name: str) -> tuple[int, float, float | None]:
    val_loss = history.get("val_loss")
    if not val_loss:
        raise ValueError("Training history does not contain validation loss values.")

    best_epoch_index = int(np.argmin(val_loss))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(val_loss[best_epoch_index])

    secondary_metric = history.get(f"val_{metric_name}")
    best_secondary = None
    if secondary_metric:
        best_secondary = float(np.min(secondary_metric)) if metric_name == "mae" else float(np.max(secondary_metric))

    return best_epoch, best_val_loss, best_secondary
