from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run po_analysis.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run po_analysis.py. Install it and try again."
    ) from exc

try:
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MinMaxScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: scikit-learn is required to run po_analysis.py. Install it and try again."
    ) from exc

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run po_analysis.py. Install it and try again."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

TRAIN_PATH = BASE_DIR / "data_case14_train.pkl"
TEST_PATH = BASE_DIR / "data_case14_test.pkl"

PO_VALUES = [0.1, 0.3, 0.5, 0.6, 0.7, 1.0]
TOTAL_LINES = 20
SEQUENCE_LENGTH = 5
RUNS_PER_PO = 1
PAPER_REFERENCE_F1 = 0.95


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    import pickle

    with path.open("rb") as file_handle:
        data = pickle.load(file_handle)
    return np.asarray(data)


def choose_capacity_channel(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        channel = np.asarray(tensor[:, :, index])
        flattened = channel.reshape(-1)
        unique_values = np.unique(flattened)
        if np.all(np.isin(unique_values, [0, 1, False, True])):
            continue
        if np.allclose(channel, 0):
            continue
        if float(np.nanmin(channel)) >= 0 and float(np.nanmean(channel)) >= 1:
            return index

    for index in range(tensor.shape[2]):
        channel = np.asarray(tensor[:, :, index])
        if not np.allclose(channel, 0):
            return index

    raise ValueError("Could not identify a usable capacity channel in the tensor.")


def choose_attack_channel(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        channel = np.asarray(tensor[:, :, index])
        unique_values = np.unique(channel.reshape(-1))
        if np.all(np.isin(unique_values, [0, 1, False, True])):
            return index

    raise ValueError("Could not identify a binary attack channel in the tensor.")


def extract_observations_and_labels(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    if tensor.shape[1] != TOTAL_LINES:
        raise ValueError(f"Expected {TOTAL_LINES} transmission lines, got shape {tensor.shape}")

    capacity_channel = choose_capacity_channel(tensor)
    attack_channel = choose_attack_channel(tensor)

    observations = np.asarray(tensor[:, :, capacity_channel], dtype=np.float32)
    attack_binary = np.asarray(tensor[:, :, attack_channel]) != 0
    occurrence = np.any(attack_binary, axis=1).astype(np.int64)
    return observations, occurrence


def select_observed_lines(observations: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
    if observations.ndim != 2:
        raise ValueError(f"Observations must be 2D, got shape {observations.shape}")
    return observations[:, selected_indices]


def fit_and_transform_scaler(train_obs: np.ndarray, test_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaler.fit(train_obs)
    return scaler.transform(train_obs), scaler.transform(test_obs), scaler


def create_sliding_windows(features: np.ndarray, occurrence: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("Sequence length must be positive.")

    if features.shape[0] != occurrence.shape[0]:
        raise ValueError("Features and labels must have the same number of timesteps.")

    if features.shape[0] < sequence_length:
        raise ValueError(
            f"Not enough timesteps to build sequences of length {sequence_length}: got {features.shape[0]}."
        )

    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for current_index in range(sequence_length - 1, features.shape[0]):
        start_index = current_index - sequence_length + 1
        sequences.append(features[start_index : current_index + 1])
        targets.append(int(occurrence[current_index]))

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def build_occurrence_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input_sequence"),
            tf.keras.layers.LSTM(128, return_sequences=True, name="lstm_128"),
            tf.keras.layers.Dropout(0.2, name="dropout_1"),
            tf.keras.layers.LSTM(64, return_sequences=False, name="lstm_64"),
            tf.keras.layers.Dropout(0.2, name="dropout_2"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
            tf.keras.layers.Dropout(0.2, name="dropout_3"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="occurrence_output"),
        ],
        name="po_occurrence_lstm",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def evaluate_po_level(
    train_obs: np.ndarray,
    train_labels: np.ndarray,
    test_obs: np.ndarray,
    test_labels: np.ndarray,
    po_value: float,
    run_index: int,
) -> float:
    n_observed = int(TOTAL_LINES * po_value)
    n_observed = max(1, min(TOTAL_LINES, n_observed))

    seed = 42 + int(po_value * 1000) * 10 + run_index
    rng = np.random.default_rng(seed)
    observed_indices = np.sort(rng.choice(TOTAL_LINES, size=n_observed, replace=False))

    train_selected = select_observed_lines(train_obs, observed_indices)
    test_selected = select_observed_lines(test_obs, observed_indices)

    train_scaled, test_scaled, _ = fit_and_transform_scaler(train_selected, test_selected)

    x_train, y_train = create_sliding_windows(train_scaled, train_labels, SEQUENCE_LENGTH)
    x_test, y_test = create_sliding_windows(test_scaled, test_labels, SEQUENCE_LENGTH)

    model = build_occurrence_lstm((SEQUENCE_LENGTH, n_observed))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    validation_size = max(1, int(0.2 * x_train.shape[0]))
    x_val = x_train[-validation_size:]
    y_val = y_train[-validation_size:]
    x_fit = x_train[:-validation_size] if x_train.shape[0] > validation_size else x_train
    y_fit = y_train[:-validation_size] if y_train.shape[0] > validation_size else y_train

    model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = np.asarray(model.predict(x_test, verbose=0)).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return float(f1_score(y_test, predictions, zero_division=0))


def plot_results(po_values: list[float], mean_scores: list[float], min_scores: list[float], max_scores: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(po_values, mean_scores, marker="o", linewidth=2.0, color="tab:blue", label="Mean F1")
    axis.fill_between(po_values, min_scores, max_scores, color="tab:blue", alpha=0.18, label="Min-Max Range")
    axis.axhline(PAPER_REFERENCE_F1, color="tab:red", linestyle="--", linewidth=1.8, label="Paper Result")

    axis.set_xlabel("Po")
    axis.set_ylabel("F1 Score")
    axis.set_title("Effect of Partial Observations on Performance")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()

    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "po_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    ensure_output_dir()

    train_tensor = load_pickle_array(TRAIN_PATH)
    test_tensor = load_pickle_array(TEST_PATH)

    train_obs, train_labels = extract_observations_and_labels(train_tensor)
    test_obs, test_labels = extract_observations_and_labels(test_tensor)

    results: list[dict[str, float]] = []
    mean_scores: list[float] = []
    min_scores: list[float] = []
    max_scores: list[float] = []

    for po_value in PO_VALUES:
        run_scores: list[float] = []
        for run_index in range(RUNS_PER_PO):
            score = evaluate_po_level(
                train_obs=train_obs,
                train_labels=train_labels,
                test_obs=test_obs,
                test_labels=test_labels,
                po_value=po_value,
                run_index=run_index,
            )
            run_scores.append(score)

        result_row = {
            "Po": float(po_value),
            "mean_f1": float(np.mean(run_scores)),
            "min_f1": float(np.min(run_scores)),
            "max_f1": float(np.max(run_scores)),
        }
        results.append(result_row)
        mean_scores.append(result_row["mean_f1"])
        min_scores.append(result_row["min_f1"])
        max_scores.append(result_row["max_f1"])

    print("\nPartial Observation Results")
    print(f"{'Po':>8} {'Mean F1':>12} {'Min F1':>12} {'Max F1':>12}")
    for row in results:
        print(f"{row['Po']:8.2f} {row['mean_f1']:12.6f} {row['min_f1']:12.6f} {row['max_f1']:12.6f}")

    plot_results(PO_VALUES, mean_scores, min_scores, max_scores)
    print(f"\nSaved plot to {PLOTS_DIR / 'po_analysis.png'}")


if __name__ == "__main__":
    main()