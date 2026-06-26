from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run evaluate_models.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run evaluate_models.py. Install it and try again."
    ) from exc

try:
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: seaborn is required to run evaluate_models.py. Install it and try again."
    ) from exc

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run evaluate_models.py. Install it and try again."
    ) from exc

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
    raise SystemExit(
        "ERROR: scikit-learn is required to run evaluate_models.py. Install it and try again."
    ) from exc


DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

BEST_MODELS = {
    "occurrence": MODELS_DIR / "best_occurrence_model.h5",
    "location": MODELS_DIR / "best_location_model.h5",
    "state": MODELS_DIR / "best_state_model.h5",
}


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required test file: {path}")
    return np.load(path, allow_pickle=False)


def load_best_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained model checkpoint: {path}. Run train_models.py first."
        )
    return tf.keras.models.load_model(path, compile=False)


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    fmt: str = ".2f",
    cmap: str = "Blues",
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0 if np.max(matrix) <= 1.0 else None,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_occurrence_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    print("\nEVALUATE MODEL 1 - Occurrence Detection")
    probabilities = np.asarray(model.predict(x_test, verbose=0)).reshape(-1)
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
    plot_confusion_matrix(
        matrix,
        labels=["No Attack", "Attack"],
        title="Attack Occurrence Detection Results",
        output_path=PLOTS_DIR / "confusion_occurrence.png",
        fmt=".2f",
        cmap="Blues",
    )

    print("\nClassification report:")
    print(classification_report(y_true, predictions, target_names=["No Attack", "Attack"], zero_division=0))

    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def evaluate_location_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    print("\nEVALUATE MODEL 2 - Location Detection")
    probabilities = np.asarray(model.predict(x_test, verbose=0))
    predictions = np.argmax(probabilities, axis=1).astype(int)
    y_true = np.asarray(y_test).reshape(-1).astype(int)

    weighted_f1 = float(f1_score(y_true, predictions, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, predictions))

    print(f"Weighted F1 score: {weighted_f1:.6f}")
    print(f"Accuracy: {accuracy:.6f}")

    matrix = confusion_matrix(y_true, predictions, labels=list(range(21)), normalize="true")
    labels = ["0"] + [str(index) for index in range(1, 21)]
    plot_confusion_matrix(
        matrix,
        labels=labels,
        title="Attack Location Detection Results",
        output_path=PLOTS_DIR / "confusion_location.png",
        fmt=".2f",
        cmap="Greens",
    )

    print("\nAccuracy per class:")
    per_class_accuracy = []
    for class_index in range(21):
        class_mask = y_true == class_index
        class_total = int(np.count_nonzero(class_mask))
        if class_total == 0:
            print(f"  Class {class_index:2d}: n/a (no samples)")
            per_class_accuracy.append(np.nan)
            continue
        class_correct = int(np.count_nonzero(predictions[class_mask] == class_index))
        class_accuracy = class_correct / class_total
        per_class_accuracy.append(class_accuracy)
        print(f"  Class {class_index:2d}: {class_accuracy:.6f} ({class_correct}/{class_total})")

    return {"weighted_f1": weighted_f1, "accuracy": accuracy}


def evaluate_state_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    print("\nEVALUATE MODEL 3 - State Estimation")
    predictions = np.asarray(model.predict(x_test, verbose=0))
    y_true = np.asarray(y_test)

    if predictions.shape != y_true.shape:
        raise ValueError(
            f"State prediction shape mismatch: predictions {predictions.shape} vs targets {y_true.shape}"
        )

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
    if len(example_lines) == 1:
        axes = [axes]

    for axis, line_index in zip(axes, example_lines, strict=False):
        axis.plot(timesteps, y_true[:preview_length, line_index], label=f"Real Line {line_index + 1}", linewidth=1.8)
        axis.plot(
            timesteps,
            predictions[:preview_length, line_index],
            label=f"Predicted Line {line_index + 1}",
            linewidth=1.8,
            linestyle="--",
        )
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
    scatter = axis.scatter(
        real_values,
        predicted_values,
        c=deviation,
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
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


def print_summary_table(
    occurrence_metrics: dict[str, float],
    location_metrics: dict[str, float],
    state_metrics: dict[str, float],
) -> None:
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

        print("Loading test data...")
        x_test = load_numpy_array(DATA_DIR / "X_test.npy")
        y_test_occur = load_numpy_array(DATA_DIR / "y_test_occur.npy")
        y_test_loc = load_numpy_array(DATA_DIR / "y_test_loc.npy")
        y_test_state = load_numpy_array(DATA_DIR / "y_test_state.npy")

        print("Loading best trained models...")
        occurrence_model = load_best_model(BEST_MODELS["occurrence"])
        location_model = load_best_model(BEST_MODELS["location"])
        state_model = load_best_model(BEST_MODELS["state"])

        occurrence_metrics = evaluate_occurrence_model(occurrence_model, x_test, y_test_occur)
        location_metrics = evaluate_location_model(location_model, x_test, y_test_loc)
        state_metrics = evaluate_state_model(state_model, x_test, y_test_state)

        print_summary_table(occurrence_metrics, location_metrics, state_metrics)
        print("\nEvaluation completed successfully.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()