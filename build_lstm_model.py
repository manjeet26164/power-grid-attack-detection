from __future__ import annotations

from pathlib import Path

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tensorflow is required to run build_lstm_model.py. Install it and try again."
    ) from exc


INPUT_SHAPE = (5, 6)
MODELS_DIR = Path("models")


def build_shared_lstm_backbone() -> tf.keras.Sequential:
    """Create the shared recurrent stack used by all three tasks.

    The first LSTM returns the full sequence so the second LSTM can learn
    temporal refinements on top of the intermediate representation.
    Dropout is used after each recurrent block to reduce overfitting.
    """

    model = tf.keras.Sequential(name="shared_lstm_backbone")

    # The input sees 5 timesteps from 6 observed transmission lines.
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_sequence"))

    # Learn short-term and medium-term temporal patterns while preserving the sequence.
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, name="lstm_128"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_1"))

    # Compress the temporal representation into a single latent vector.
    model.add(tf.keras.layers.LSTM(64, return_sequences=False, name="lstm_64"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_2"))

    return model


def build_attack_occurrence_model() -> tf.keras.Model:
    backbone = build_shared_lstm_backbone()

    # Compact dense projection before the binary decision boundary.
    backbone.add(tf.keras.layers.Dense(16, activation="relu", name="dense_16"))
    backbone.add(tf.keras.layers.Dropout(0.2, name="dropout_3"))

    # Sigmoid outputs the probability that an attack is occurring.
    backbone.add(tf.keras.layers.Dense(1, activation="sigmoid", name="occurrence_output"))

    backbone.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return backbone


def build_attack_location_model() -> tf.keras.Model:
    backbone = build_shared_lstm_backbone()

    # Expand the latent vector a bit more because this task has 21 classes.
    backbone.add(tf.keras.layers.Dense(64, activation="relu", name="dense_64"))
    backbone.add(tf.keras.layers.Dropout(0.2, name="dropout_3"))

    # Softmax predicts the no-attack class plus the 20 possible attacked lines.
    backbone.add(tf.keras.layers.Dense(21, activation="softmax", name="location_output"))

    backbone.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return backbone


def build_state_estimation_model() -> tf.keras.Model:
    backbone = build_shared_lstm_backbone()

    # Regression head maps the latent state back to the full 20-line capacity vector.
    backbone.add(tf.keras.layers.Dense(64, activation="relu", name="dense_64"))
    backbone.add(tf.keras.layers.Dropout(0.2, name="dropout_3"))

    # Linear activation is appropriate for continuous-valued outputs.
    backbone.add(tf.keras.layers.Dense(20, activation="linear", name="state_output"))

    backbone.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return backbone


def save_model_architecture(model: tf.keras.Model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model.to_json(indent=2), encoding="utf-8")


def main() -> None:
    try:
        occurrence_model = build_attack_occurrence_model()
        location_model = build_attack_location_model()
        state_model = build_state_estimation_model()

        print("MODEL 1 - Attack Occurrence Detection")
        occurrence_model.summary()
        print()

        print("MODEL 2 - Attack Location Detection")
        location_model.summary()
        print()

        print("MODEL 3 - State Estimation")
        state_model.summary()

        save_model_architecture(occurrence_model, MODELS_DIR / "attack_occurrence_model.json")
        save_model_architecture(location_model, MODELS_DIR / "attack_location_model.json")
        save_model_architecture(state_model, MODELS_DIR / "state_estimation_model.json")

        print(f"\nSaved model architectures to: {MODELS_DIR}")
        print("- attack_occurrence_model.json")
        print("- attack_location_model.json")
        print("- state_estimation_model.json")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()