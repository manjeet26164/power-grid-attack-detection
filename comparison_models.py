from __future__ import annotations

import pickle
import time
from pathlib import Path
import os

os.makedirs('plots', exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run comparison_models.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run comparison_models.py. Install it and try again."
    ) from exc

try:
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: seaborn is required to run comparison_models.py. Install it and try again."
    ) from exc

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run comparison_models.py. Install it and try again."
    ) from exc

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
    raise SystemExit(
        "ERROR: scikit-learn is required to run comparison_models.py. Install it and try again."
    ) from exc


DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

LSTM_MODEL_PATH = MODELS_DIR / "best_occurrence_model.h5"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
FNN_MODEL_PATH = MODELS_DIR / "fnn_model.h5"
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


def load_lstm_model() -> tf.keras.Model:
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing LSTM checkpoint: {LSTM_MODEL_PATH}. Train the LSTM models first."
        )
    return tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)


def build_fnn_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="flattened_sequence_input"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(0.2, name="dropout_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
            tf.keras.layers.Dropout(0.2, name="dropout_2"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
            tf.keras.layers.Dropout(0.2, name="dropout_3"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="occurrence_output"),
        ],
        name="fnn_occurrence_model",
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


def compute_binary_class_weight(labels: np.ndarray) -> dict[int, float]:
    flattened = np.asarray(labels).reshape(-1).astype(int)
    classes = np.unique(flattened)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=flattened)
    return {int(class_value): float(weight) for class_value, weight in zip(classes, weights, strict=False)}


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
        ax=axis,
    )
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

    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_lstm_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    probabilities = np.asarray(model.predict(x_test, verbose=0)).reshape(-1)
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


def evaluate_random_forest(
    x_train_flat: np.ndarray,
    y_train: np.ndarray,
    x_test_flat: np.ndarray,
    y_test: np.ndarray,
) -> float:
    print("\nBuilding Random Forest with RandomizedSearchCV...")
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=2,
        cv=2,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    start_time = time.perf_counter()
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
) -> float:
    print("\nTraining Feedforward Neural Network...")
    model = build_fnn_model(x_train_flat.shape[1])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(FNN_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    class_weight = compute_binary_class_weight(y_train)
    print(f"Class weights: {class_weight}")

    start_time = time.perf_counter()
    model.fit(
        x_train_flat,
        y_train,
        validation_data=(x_val_flat, y_val),
        epochs=3,
        batch_size=1024,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    elapsed = time.perf_counter() - start_time
    print(f"FNN training completed in {elapsed:.2f} seconds")

    if FNN_MODEL_PATH.exists():
        model = tf.keras.models.load_model(FNN_MODEL_PATH, compile=False)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    probabilities = np.asarray(model.predict(x_test_flat, verbose=0)).reshape(-1)
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
        lstm_model = load_lstm_model()
        lstm_f1 = evaluate_lstm_model(lstm_model, x_test, y_test_occur.reshape(-1))

        rf_f1 = evaluate_random_forest(
            x_train_flat,
            y_train_occur.reshape(-1),
            x_test_flat,
            y_test_occur.reshape(-1),
        )

        fnn_f1 = train_fnn(
            x_train_flat,
            y_train_occur.reshape(-1),
            x_val_flat,
            y_val_occur.reshape(-1),
            x_test_flat,
            y_test_occur.reshape(-1),
        )

        plot_comparison_bar_chart(
            {"LSTM": lstm_f1, "Random Forest": rf_f1, "FNN": fnn_f1},
            COMPARISON_PLOT_PATH,
        )
        print_comparison_table(lstm_f1, rf_f1, fnn_f1)
        print(f"\nSaved comparison chart to: {COMPARISON_PLOT_PATH}")
        print(f"Saved Random Forest model to: {RANDOM_FOREST_PATH}")
        print(f"Saved FNN model to: {FNN_MODEL_PATH}")
        
        # === DASHBOARD DYNAMIC METRICS COUPLING SYSTEM ===
        import pickle
        from pathlib import Path
        
        metrics_to_save = {
            'lstm_f1': lstm_f1,
            'rf_f1': rf_f1,
            'fnn_f1': fnn_f1
        }
        
        # === DASHBOARD DYNAMIC METRICS COUPLING SYSTEM ===
        import pickle
        from pathlib import Path
        
        # Hardcoded numbers hata kar real calculated variables map kar diye hain
        metrics_to_save = {
            'lstm_f1': lstm_f1,
            'rf_f1': rf_f1,
            'fnn_f1': fnn_f1
        }
        
        base_path = Path(__file__).resolve().parent
        backup_file = base_path / "plots" / "metrics_backup.pkl"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_file, "wb") as f:
            pickle.dump(metrics_to_save, f)
            
        print("-> Dashboard dynamic metrics backup saved successfully with live metrics!")
        # =================================================
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()