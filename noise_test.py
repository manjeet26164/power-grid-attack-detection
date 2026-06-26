from __future__ import annotations

import pickle
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run noise_test.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run noise_test.py. Install it and try again."
    ) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: pandas is required to run noise_test.py. Install it and try again."
    ) from exc

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run noise_test.py. Install it and try again."
    ) from exc

try:
    from sklearn.metrics import f1_score
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: scikit-learn is required to run noise_test.py. Install it and try again."
    ) from exc


DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

LSTM_MODEL_PATH = MODELS_DIR / "best_occurrence_model.h5"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
FNN_MODEL_PATH = MODELS_DIR / "fnn_model.h5"

NOISE_LEVELS = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required preprocessed file: {path}")
    return np.load(path, allow_pickle=False)


def load_lstm_model() -> tf.keras.Model:
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM model checkpoint: {LSTM_MODEL_PATH}")
    return tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)


def load_random_forest_model():
    if not RANDOM_FOREST_PATH.exists():
        raise FileNotFoundError(f"Missing Random Forest model: {RANDOM_FOREST_PATH}")
    with RANDOM_FOREST_PATH.open("rb") as file_handle:
        return pickle.load(file_handle)


def load_fnn_model() -> tf.keras.Model:
    if not FNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing FNN model checkpoint: {FNN_MODEL_PATH}")
    return tf.keras.models.load_model(FNN_MODEL_PATH, compile=False)


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    if sequences.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {sequences.shape}")
    return sequences.reshape(sequences.shape[0], -1).astype(np.float32)


def add_noise(rng: np.random.Generator, x_test: np.ndarray, sigma: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sigma, size=x_test.shape)
    return (x_test + noise).astype(np.float32)


def predict_binary_probabilities(model, inputs: np.ndarray) -> np.ndarray:
    predictions = np.asarray(model.predict(inputs, verbose=0))
    return predictions.reshape(-1)


def compute_f1(y_true: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(y_true.reshape(-1).astype(int), predictions.astype(int), zero_division=0))


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(results["noise_level"], results["LSTM"], marker="o", color="green", label="LSTM")
    axis.plot(results["noise_level"], results["RF"], marker="o", color="orange", label="RF")
    axis.plot(results["noise_level"], results["FNN"], marker="o", color="blue", label="FNN")

    axis.set_xscale("log")
    axis.set_xlabel("Noise level (sigma)")
    axis.set_ylabel("F1 Score")
    axis.set_title("Model Robustness to Sensor Noise")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    ensure_output_dir()

    x_test = load_numpy_array(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test_occur = load_numpy_array(DATA_DIR / "y_test_occur.npy").reshape(-1).astype(int)

    if x_test.ndim != 3:
        raise ValueError(f"Expected X_test to be 3D, got shape {x_test.shape}")

    lstm_model = load_lstm_model()
    random_forest = load_random_forest_model()
    fnn_model = load_fnn_model()

    rng = np.random.default_rng(seed=42)
    x_test_flat = flatten_sequences(x_test)

    rows: list[dict[str, float]] = []

    for sigma in NOISE_LEVELS:
        x_noisy = add_noise(rng, x_test, sigma)
        x_noisy_flat = flatten_sequences(x_noisy)

        lstm_probabilities = predict_binary_probabilities(lstm_model, x_noisy)
        lstm_predictions = (lstm_probabilities >= 0.5).astype(int)

        rf_predictions = random_forest.predict(x_noisy_flat)
        fnn_probabilities = predict_binary_probabilities(fnn_model, x_noisy_flat)
        fnn_predictions = (fnn_probabilities >= 0.5).astype(int)

        row = {
            "noise_level": sigma,
            "LSTM": compute_f1(y_test_occur, lstm_predictions),
            "RF": compute_f1(y_test_occur, rf_predictions),
            "FNN": compute_f1(y_test_occur, fnn_predictions),
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results[["noise_level", "LSTM", "RF", "FNN"]]

    pd.set_option("display.float_format", lambda value: f"{value:.6f}")
    print("\nNoise Robustness Results")
    print(results.to_string(index=False))

    plot_results(results, PLOTS_DIR / "noise_robustness.png")
    print(f"\nSaved plot to {PLOTS_DIR / 'noise_robustness.png'}")


if __name__ == "__main__":
    main()