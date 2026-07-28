from __future__ import annotations
from joblib import parallel_backend

import pickle
import time
from pathlib import Path
import os

os.makedirs("plots", exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: matplotlib is required to run comparison_models.py. Install it and try again.") from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: numpy is required to run comparison_models.py. Install it and try again.") from exc

try:
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: seaborn is required to run comparison_models.py. Install it and try again.") from exc

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: torch is required to run comparison_models.py. Install it and try again.") from exc

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.utils.class_weight import compute_class_weight
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: scikit-learn is required to run comparison_models.py. Install it and try again.") from exc

from build_lstm_model import AttackOccurrenceModel
from train_utils import EarlyStopping, get_device, iterate_minibatches, save_checkpoint

DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

LSTM_MODEL_PATH = MODELS_DIR / "best_occurrence_model.pt"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
FNN_MODEL_PATH = MODELS_DIR / "fnn_model.pt"
COMPARISON_PLOT_PATH = PLOTS_DIR / "model_comparison.png"


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required preprocessed file: {path}")
    return np.load(path, allow_pickle=False)


def ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    if sequences.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {sequences.shape}")
    return sequences.reshape(sequences.shape[0], -1).astype(np.float32)


def load_lstm_model(device: "torch.device") -> "torch.nn.Module":
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM checkpoint: {LSTM_MODEL_PATH}. Train the LSTM models first.")
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
    model = AttackOccurrenceModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


class FNNOccurrenceModel(nn.Module):
    """Fully-connected baseline on flattened sequences (mirrors the Keras FNN)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


def compute_binary_class_weight(labels: np.ndarray) -> dict[int, float]:
    flattened = np.asarray(labels).reshape(-1).astype(int)
    classes = np.unique(flattened)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=flattened)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=1.0, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_comparison_bar_chart(scores: dict[str, float], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    models = list(scores.keys())
    values = list(scores.values())
    colors = ["tab:blue", "tab:orange", "tab:green"]

    bars = axis.bar(models, values, color=colors[: len(models)], width=0.55)
    axis.set_ylim(0.0, max(1.0, max(values) * 1.15))
    axis.set_ylabel("F1 Score")
    axis.set_title("Model Comparison - Attack Detection")
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=10)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_lstm_model(model: "torch.nn.Module", x_test: np.ndarray, y_test: np.ndarray, device: "torch.device") -> float:
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    y_true = np.asarray(y_test).reshape(-1).astype(int)

    f1 = float(f1_score(y_true, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_true, predictions))
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))

    print("\nLSTM Results")
    print(f"F1 score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1], normalize="true")
    plot_confusion_matrix(matrix, ["No Attack", "Attack"], "LSTM Attack Detection Results", PLOTS_DIR / "confusion_lstm.png")
    print(classification_report(y_true, predictions, target_names=["No Attack", "Attack"], zero_division=0))
    return f1


def evaluate_random_forest(x_train_flat: np.ndarray, y_train: np.ndarray, x_test_flat: np.ndarray, y_test: np.ndarray) -> float:
    print("\nBuilding Random Forest with RandomizedSearchCV...")
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=4,
        cv=2,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    start_time = time.perf_counter()
    with parallel_backend('threading', n_jobs=-1):
        search.fit(x_train_flat, y_train)
    elapsed = time.perf_counter() - start_time
    print(f"Random Forest search completed in {elapsed:.2f} seconds")
    print(f"Best RF parameters: {search.best_params_}")

    best_model: RandomForestClassifier = search.best_estimator_
    with RANDOM_FOREST_PATH.open("wb") as file_handle:
        pickle.dump(best_model, file_handle)

    predictions = best_model.predict(x_test_flat)
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))

    print("\nRandom Forest Results")
    print(f"F1 score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1], normalize="true")
    plot_confusion_matrix(matrix, ["No Attack", "Attack"], "Random Forest Attack Detection Results", PLOTS_DIR / "confusion_random_forest.png")
    print(classification_report(y_test, predictions, target_names=["No Attack", "Attack"], zero_division=0))
    return f1


def train_fnn(
    x_train_flat: np.ndarray,
    y_train: np.ndarray,
    x_val_flat: np.ndarray,
    y_val: np.ndarray,
    x_test_flat: np.ndarray,
    y_test: np.ndarray,
    device: "torch.device",
) -> float:
    print("\nTraining Feedforward Neural Network...")
    model = FNNOccurrenceModel(x_train_flat.shape[1]).to(device)
    print(model)

    class_weight = compute_binary_class_weight(y_train)
    print(f"Class weights: {class_weight}")
    pos_weight = torch.tensor([class_weight[1] / class_weight[0]], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=10)

    x_train_t = torch.from_numpy(x_train_flat).float()
    y_train_t = torch.from_numpy(y_train).float().reshape(-1)
    x_val_t = torch.from_numpy(x_val_flat).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().reshape(-1).to(device)

    start_time = time.perf_counter()
    for epoch in range(1, 61):
        model.train()
        for xb, yb in iterate_minibatches(x_train_t, y_train_t, 1024, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/60 - val_loss: {val_loss:.4f}")

        if early_stopping.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    early_stopping.restore_best_weights(model)
    elapsed = time.perf_counter() - start_time
    print(f"FNN training completed in {elapsed:.2f} seconds")

    save_checkpoint(model, x_train_flat.shape[1], {"model_type": "fnn"}, FNN_MODEL_PATH)

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(x_test_flat).float().to(device))).cpu().numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))

    print("\nFNN Results")
    print(f"F1 score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1], normalize="true")
    plot_confusion_matrix(matrix, ["No Attack", "Attack"], "FNN Attack Detection Results", PLOTS_DIR / "confusion_fnn.png")
    print(classification_report(y_test, predictions, target_names=["No Attack", "Attack"], zero_division=0))
    return f1


def print_comparison_table(lstm_f1: float, rf_f1: float, fnn_f1: float) -> None:
    print("\nFinal comparison table:")
    header = f"{'Model':<18} | {'Metric':<8} | {'Score':<10} | {'Paper Score'}"
    print(header)
    print("-" * len(header))
    print(f"{'LSTM':<18} | {'F1':<8} | {lstm_f1:<10.6f} | 0.92-0.99")
    print(f"{'Random Forest':<18} | {'F1':<8} | {rf_f1:<10.6f} | 0.92-0.99")
    print(f"{'FNN':<18} | {'F1':<8} | {fnn_f1:<10.6f} | 0.92-0.99")


def main() -> None:
    try:
        ensure_output_dirs()
        device = get_device()
        print(f"Using device: {device}")

        print("Loading preprocessed arrays...")
        x_train = load_numpy_array(DATA_DIR / "X_train.npy")
        x_val = load_numpy_array(DATA_DIR / "X_val.npy")
        x_test = load_numpy_array(DATA_DIR / "X_test.npy")
        y_train_occur = load_numpy_array(DATA_DIR / "y_train_occur.npy")
        y_val_occur = load_numpy_array(DATA_DIR / "y_val_occur.npy")
        y_test_occur = load_numpy_array(DATA_DIR / "y_test_occur.npy")

        print("Flattening sequence inputs for Random Forest and FNN...")
        x_train_flat = flatten_sequences(x_train)
        x_val_flat = flatten_sequences(x_val)
        x_test_flat = flatten_sequences(x_test)
        print(f"Flattened training shape: {x_train_flat.shape}")

        print("Loading best LSTM model...")
        lstm_model = load_lstm_model(device)
        lstm_f1 = evaluate_lstm_model(lstm_model, x_test, y_test_occur.reshape(-1), device)

        rf_f1 = evaluate_random_forest(x_train_flat, y_train_occur.reshape(-1), x_test_flat, y_test_occur.reshape(-1))

        fnn_f1 = train_fnn(
            x_train_flat, y_train_occur.reshape(-1), x_val_flat, y_val_occur.reshape(-1), x_test_flat, y_test_occur.reshape(-1), device
        )

        plot_comparison_bar_chart({"LSTM": lstm_f1, "Random Forest": rf_f1, "FNN": fnn_f1}, COMPARISON_PLOT_PATH)
        print_comparison_table(lstm_f1, rf_f1, fnn_f1)
        print(f"\nSaved comparison chart to: {COMPARISON_PLOT_PATH}")
        print(f"Saved Random Forest model to: {RANDOM_FOREST_PATH}")
        print(f"Saved FNN model to: {FNN_MODEL_PATH}")

        # === DASHBOARD DYNAMIC METRICS COUPLING SYSTEM ===
        metrics_to_save = {"lstm_f1": lstm_f1, "rf_f1": rf_f1, "fnn_f1": fnn_f1}
        base_path = Path(__file__).resolve().parent
        backup_file = base_path / "plots" / "metrics_backup.pkl"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_file, "wb") as f:
            pickle.dump(metrics_to_save, f)
        print("-> Dashboard dynamic metrics backup saved successfully with live metrics!")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
