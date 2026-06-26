from __future__ import annotations

import json
import pickle
import importlib
import time
from pathlib import Path
from typing import Any, Callable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run train_models.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run train_models.py. Install it and try again."
    ) from exc

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run train_models.py. Install it and try again."
    ) from exc


DATA_DIR = Path("data/preprocessed")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

MODEL_JSONS = {
    "occurrence": MODELS_DIR / "attack_occurrence_model.json",
    "location": MODELS_DIR / "attack_location_model.json",
    "state": MODELS_DIR / "state_estimation_model.json",
}

MODEL_CHECKPOINTS = {
    "occurrence": MODELS_DIR / "best_occurrence_model.h5",
    "location": MODELS_DIR / "best_location_model.h5",
    "state": MODELS_DIR / "best_state_model.h5",
}


def load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required preprocessed file: {path}")
    return np.load(path, allow_pickle=False)


def load_pickle_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required pickle file: {path}")
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


def ensure_model_files_exist() -> None:
    missing = [str(path) for path in MODEL_JSONS.values() if not path.exists()]
    if missing:
        print("Model architecture files are missing. Generating them now from build_lstm_model.py...")
        build_module = importlib.import_module("build_lstm_model")

        generators: dict[str, Callable[[], tf.keras.Model]] = {
            "occurrence": build_module.build_attack_occurrence_model,
            "location": build_module.build_attack_location_model,
            "state": build_module.build_state_estimation_model,
        }

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for model_name, json_path in MODEL_JSONS.items():
            model = generators[model_name]()
            json_path.write_text(model.to_json(indent=2), encoding="utf-8")
            print(f"Saved model architecture to: {json_path}")

    remaining_missing = [str(path) for path in MODEL_JSONS.values() if not path.exists()]
    if remaining_missing:
        raise FileNotFoundError(
            "Missing model architecture file(s): "
            + ", ".join(remaining_missing)
            + ". Model generation failed."
        )


def load_model_from_json(json_path: Path) -> tf.keras.Model:
    with json_path.open("r", encoding="utf-8") as file_handle:
        model_json = file_handle.read()
    return tf.keras.models.model_from_json(model_json)


def compile_occurrence_model(model: tf.keras.Model) -> tf.keras.Model:
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


def compile_location_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def compile_state_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def compute_binary_class_weight(labels: np.ndarray) -> dict[int, float]:
    flattened = np.asarray(labels).reshape(-1).astype(int)
    if flattened.size == 0:
        raise ValueError("Cannot compute class weights from an empty label array.")

    unique_classes, counts = np.unique(flattened, return_counts=True)
    total = flattened.size
    class_weight: dict[int, float] = {}
    for class_value, count in zip(unique_classes, counts, strict=False):
        if count <= 0:
            raise ValueError(f"Invalid class count for label {class_value}: {count}")
        class_weight[int(class_value)] = float(total / (len(unique_classes) * count))

    for class_value in (0, 1):
        class_weight.setdefault(class_value, 1.0)

    return class_weight


def build_callbacks(model_name: str) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path = MODEL_CHECKPOINTS[model_name]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
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


def plot_training_history(history: tf.keras.callbacks.History, model_name: str, metric_name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = PLOTS_DIR / f"{model_name}_training_curves.png"

    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_dict["loss"], label="Training loss", color="tab:blue")
    axes[0].plot(epochs, history_dict["val_loss"], label="Validation loss", color="tab:orange")
    axes[0].set_title(f"{model_name.replace('_', ' ').title()} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    metric_key = metric_name
    val_metric_key = f"val_{metric_name}"
    if metric_key in history_dict and val_metric_key in history_dict:
        axes[1].plot(epochs, history_dict[metric_key], label=f"Training {metric_name}", color="tab:green")
        axes[1].plot(epochs, history_dict[val_metric_key], label=f"Validation {metric_name}", color="tab:red")
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


def report_best_scores(history: tf.keras.callbacks.History, metric_name: str) -> tuple[int, float, float | None]:
    history_dict = history.history
    val_loss = history_dict.get("val_loss")
    if not val_loss:
        raise ValueError("Training history does not contain validation loss values.")

    best_epoch_index = int(np.argmin(val_loss))
    best_epoch = best_epoch_index + 1
    best_val_loss = float(val_loss[best_epoch_index])

    secondary_metric = history_dict.get(f"val_{metric_name}")
    best_secondary = None
    if secondary_metric:
        if metric_name == "mae":
            best_secondary = float(np.min(secondary_metric))
        else:
            best_secondary = float(np.max(secondary_metric))

    return best_epoch, best_val_loss, best_secondary


def train_model(
    model_name: str,
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    metric_name: str,
    class_weight: dict[int, float] | None = None,
) -> tf.keras.callbacks.History:
    print(f"\n=== Training {model_name.replace('_', ' ').title()} ===")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    if class_weight is not None:
        print(f"Class weights: {json.dumps(class_weight, sort_keys=True)}")

    callbacks = build_callbacks(model_name)
    start_time = time.perf_counter()

    fit_kwargs: dict[str, Any] = {
        "x": x_train,
        "y": y_train,
        "validation_data": (x_val, y_val),
        "epochs": epochs,
        "batch_size": batch_size,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(**fit_kwargs)

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Training time for {model_name.replace('_', ' ').title()}: {elapsed_seconds:.2f} seconds")

    best_epoch, best_val_loss, best_secondary = report_best_scores(history, metric_name)
    print(f"Best epoch for {model_name.replace('_', ' ').title()}: {best_epoch}")
    print(f"Best validation loss for {model_name.replace('_', ' ').title()}: {best_val_loss:.6f}")
    if best_secondary is not None:
        print(f"Best validation {metric_name} for {model_name.replace('_', ' ').title()}: {best_secondary:.6f}")

    plot_path = plot_training_history(history, model_name, metric_name)
    print(f"Saved training curves to: {plot_path}")

    return history


def main() -> None:
    try:
        print("Loading preprocessed arrays...")
        x_train = load_numpy_array(DATA_DIR / "X_train.npy")
        x_val = load_numpy_array(DATA_DIR / "X_val.npy")
        x_test = load_numpy_array(DATA_DIR / "X_test.npy")

        y_train_occur = load_numpy_array(DATA_DIR / "y_train_occur.npy")
        y_val_occur = load_numpy_array(DATA_DIR / "y_val_occur.npy")
        y_test_occur = load_numpy_array(DATA_DIR / "y_test_occur.npy")

        y_train_loc = load_numpy_array(DATA_DIR / "y_train_loc.npy")
        y_val_loc = load_numpy_array(DATA_DIR / "y_val_loc.npy")
        y_test_loc = load_numpy_array(DATA_DIR / "y_test_loc.npy")

        y_train_state = load_numpy_array(DATA_DIR / "y_train_state.npy")
        y_val_state = load_numpy_array(DATA_DIR / "y_val_state.npy")
        y_test_state = load_numpy_array(DATA_DIR / "y_test_state.npy")

        print(f"X_train shape: {x_train.shape}")
        print(f"X_val shape: {x_val.shape}")
        print(f"X_test shape: {x_test.shape}")

        print("Checking model architecture files...")
        ensure_model_files_exist()

        print("Loading model architectures from JSON files...")
        occurrence_model = compile_occurrence_model(load_model_from_json(MODEL_JSONS["occurrence"]))
        location_model = compile_location_model(load_model_from_json(MODEL_JSONS["location"]))
        state_model = compile_state_model(load_model_from_json(MODEL_JSONS["state"]))

        print("Model 1: Attack Occurrence Detection")
        occurrence_model.summary()
        occurrence_class_weight = compute_binary_class_weight(y_train_occur)
        train_model(
            model_name="occurrence",
            model=occurrence_model,
            x_train=x_train,
            y_train=y_train_occur,
            x_val=x_val,
            y_val=y_val_occur,
            epochs=3,
            batch_size=1024,
            metric_name="accuracy",
            class_weight=occurrence_class_weight,
        )

        print("Model 2: Attack Location Detection")
        location_model.summary()
        train_model(
            model_name="location",
            model=location_model,
            x_train=x_train,
            y_train=y_train_loc,
            x_val=x_val,
            y_val=y_val_loc,
            epochs=3,
            batch_size=1024,
            metric_name="accuracy",
        )

        print("Model 3: State Estimation")
        state_model.summary()
        train_model(
            model_name="state",
            model=state_model,
            x_train=x_train,
            y_train=y_train_state,
            x_val=x_val,
            y_val=y_val_state,
            epochs=3,
            batch_size=1024,
            metric_name="mae",
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