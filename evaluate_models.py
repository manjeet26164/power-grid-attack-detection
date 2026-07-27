from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: matplotlib is required to run evaluate_models.py. Install it and try again.") from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: numpy is required to run evaluate_models.py. Install it and try again.") from exc

try:
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: seaborn is required to run evaluate_models.py. Install it and try again.") from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: torch is required to run evaluate_models.py. Install it and try again.") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_squared_error,
        precision_score,
        recall_score,
    )
except ModuleNotFoundError as exc:
    raise SystemExit("ERROR: scikit-learn is required to run evaluate_models.py. Install it and try again.") from exc

from build_lstm_model import AttackOccurrenceModel, AttackLocationModel, StateEstimationModel
from train_utils import get_device

DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

BEST_MODELS = {
    "occurrence": MODELS_DIR / "best_occurrence_model.pt",
    "location": MODELS_DIR / "best_location_model.pt",
    "state": MODELS_DIR / "best_state_model.pt",
}


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required test file: {path}")
    return np.load(path, allow_pickle=False)


def load_occurrence_model(path: Path, device) -> "torch.nn.Module":
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {path}. Run train_models.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = AttackOccurrenceModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def load_location_model(path: Path, device) -> "torch.nn.Module":
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {path}. Run train_models.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = AttackLocationModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def load_state_model(path: Path, output_dim: int, device) -> "torch.nn.Module":
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {path}. Run train_models.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = StateEstimationModel(checkpoint["input_dim"], output_dim=output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(matrix, labels, title, output_path, fmt=".2f", cmap="Blues") -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, vmin=0.0,
                vmax=1.0 if np.max(matrix) <= 1.0 else None, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_occurrence_model(model, x_test: np.ndarray, y_test: np.ndarray, device) -> dict[str, float]:
    print("\nEVALUATE MODEL 1 - Occurrence Detection")
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    y_true = np.asarray(y_test).reshape(-1).astype(int)

    f1 = float(f1_score(y_true, predictions))
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_true, predictions))

    print(f"F1 score: {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1], normalize="true")
    plot_confusion_matrix(matrix, ["No Attack", "Attack"], "Attack Occurrence Detection Results", PLOTS_DIR / "confusion_occurrence.png")

    print("\nClassification report:")
    print(classification_report(y_true, predictions, target_names=["No Attack", "Attack"], zero_division=0))
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def evaluate_location_model(model, x_test: np.ndarray, y_test: np.ndarray, device) -> dict[str, float]:
    print("\nEVALUATE MODEL 2 - Location Detection")
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).float().to(device))
        probabilities = logits.cpu().numpy()
    predictions = np.argmax(probabilities, axis=1).astype(int)
    y_true = np.asarray(y_test).reshape(-1).astype(int)

    weighted_f1 = float(f1_score(y_true, predictions, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, predictions))

    print(f"Weighted F1 score: {weighted_f1:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_true, predictions, labels=list(range(21)), normalize="true")
    labels = ["0"] + [str(index) for index in range(1, 21)]
    plot_confusion_matrix(matrix, labels, "Attack Location Detection Results", PLOTS_DIR / "confusion_location.png", cmap="Greens")

    print("\nAccuracy per class:")
    for class_index in range(21):
        class_mask = y_true == class_index
        class_total = int(np.count_nonzero(class_mask))
        if class_total == 0:
            print(f"  Class {class_index:2d}: n/a (no samples)")
            continue
        class_correct = int(np.count_nonzero(predictions[class_mask] == class_index))
        print(f"  Class {class_index:2d}: {class_correct / class_total:.6f} ({class_correct}/{class_total})")

    return {"weighted_f1": weighted_f1, "accuracy": accuracy}


def evaluate_state_model(model, x_test: np.ndarray, y_test: np.ndarray, device) -> dict[str, float]:
    print("\nEVALUATE MODEL 3 - State Estimation")
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test).float().to(device)).cpu().numpy()
    y_true = np.asarray(y_test)

    if predictions.shape != y_true.shape:
        raise ValueError(f"State prediction shape mismatch: predictions {predictions.shape} vs targets {y_true.shape}")

    mse_per_line = np.mean(np.square(predictions - y_true), axis=0)
    overall_mse = float(mean_squared_error(y_true.reshape(-1), predictions.reshape(-1)))

    print(f"Overall MSE: {overall_mse:.10f}")
    print("MSE per line:")
    for line_index, line_mse in enumerate(mse_per_line, start=1):
        print(f"  Line {line_index:2d}: {float(line_mse):.10f}")

    example_lines = [0, 1, 2]
    preview_length = min(300, y_true.shape[0])
    timesteps = np.arange(preview_length)

    figure, axes = plt.subplots(len(example_lines), 1, figsize=(14, 10), sharex=True)
    for axis, line_index in zip(axes, example_lines):
        axis.plot(timesteps, y_true[:preview_length, line_index], label=f"Real Line {line_index + 1}", linewidth=1.8)
        axis.plot(timesteps, predictions[:preview_length, line_index], label=f"Predicted Line {line_index + 1}", linewidth=1.8, linestyle="--")
        axis.set_ylabel("Capacity")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Timestep")
    figure.suptitle("State Estimation Results")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "state_estimation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    scatter_line_index = 0
    real_values = y_true[:preview_length, scatter_line_index].reshape(-1)
    predicted_values = predictions[:preview_length, scatter_line_index].reshape(-1)
    deviation = np.abs(predicted_values - real_values)
    scatter_mse = float(mean_squared_error(real_values, predicted_values))

    figure, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(real_values, predicted_values, c=deviation, cmap="viridis", alpha=0.75, edgecolors="none")
    min_value = float(min(real_values.min(), predicted_values.min()))
    max_value = float(max(real_values.max(), predicted_values.max()))
    axis.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=1.5)
    axis.set_xlabel("Real values")
    axis.set_ylabel("Predicted values")
    axis.set_title(f"Predicted vs Real Values for Line {scatter_line_index + 1}\nMSE = {scatter_mse:.10f}")
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Absolute deviation")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "state_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    return {"overall_mse": overall_mse, "scatter_mse": scatter_mse}


def print_summary_table(occurrence_metrics, location_metrics, state_metrics) -> None:
    print("\nFinal summary table:")
    header = f"{'Model':<15} | {'Metric':<8} | {'Score':<12} | {'Paper Score'}"
    print(header)
    print("-" * len(header))
    print(f"{'Occurrence':<15} | {'F1':<8} | {occurrence_metrics['f1']:<12.6f} | 0.92-0.99")
    print(f"{'Location':<15} | {'F1':<8} | {location_metrics['weighted_f1']:<12.6f} | 0.85-0.95")
    print(f"{'State (rho)':<15} | {'MSE':<8} | {state_metrics['overall_mse']:<12.10f} | 0.000242")


def main() -> None:
    try:
        ensure_output_dir()
        device = get_device()
        print(f"Using device: {device}")

        print("Loading test data...")
        x_test = load_numpy_array(DATA_DIR / "X_test.npy")
        y_test_occur = load_numpy_array(DATA_DIR / "y_test_occur.npy")
        y_test_loc = load_numpy_array(DATA_DIR / "y_test_loc.npy")
        y_test_state = load_numpy_array(DATA_DIR / "y_test_state.npy")

        print("Loading best trained models...")
        occurrence_model = load_occurrence_model(BEST_MODELS["occurrence"], device)
        location_model = load_location_model(BEST_MODELS["location"], device)
        state_model = load_state_model(BEST_MODELS["state"], output_dim=y_test_state.shape[1], device=device)

        occurrence_metrics = evaluate_occurrence_model(occurrence_model, x_test, y_test_occur, device)
        location_metrics = evaluate_location_model(location_model, x_test, y_test_loc, device)
        state_metrics = evaluate_state_model(state_model, x_test, y_test_state, device)

        print_summary_table(occurrence_metrics, location_metrics, state_metrics)
        print("\nEvaluation completed successfully.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
