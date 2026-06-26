from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run preprocess_data.py. Install it and try again."
    ) from exc

try:
    from sklearn.preprocessing import MinMaxScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: scikit-learn is required to run preprocess_data.py. Install it and try again."
    ) from exc


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data/preprocessed")
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_PO = 0.3
DEFAULT_SELECTED_LINES = 6
DEFAULT_SEED = 42

TRAIN_FILENAME = "data_case14_train.pkl"
VAL_FILENAME = "data_case14_val.pkl"
TEST_FILENAME = "data_case14_test.pkl"


@dataclass
class LocatedArray:
    path: str
    array: np.ndarray


def load_pickle_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    try:
        with path.open("rb") as file_handle:
            return pickle.load(file_handle)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(f"Could not unpickle {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def resolve_input_file(data_dir: Path, filename: str) -> Path:
    candidates = [
        data_dir / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "data" / filename,
    ]

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the input pickle file. Tried: "
        + ", ".join(str(candidate) for candidate in unique_candidates)
    )


def iter_arrays(value: Any, path: str = "root", seen: set[int] | None = None) -> list[LocatedArray]:
    if seen is None:
        seen = set()

    object_id = id(value)
    if object_id in seen:
        return []
    seen.add(object_id)

    located: list[LocatedArray] = []

    if isinstance(value, np.ndarray):
        located.append(LocatedArray(path=path, array=value))
        return located

    if isinstance(value, dict):
        for key, child in value.items():
            located.extend(iter_arrays(child, f"{path}.{key}", seen))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            located.extend(iter_arrays(child, f"{path}[{index}]", seen))
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for key, child in vars(value).items():
            located.extend(iter_arrays(child, f"{path}.{key}", seen))

    return located


def as_float_array(array: Any) -> np.ndarray:
    numeric = np.asarray(array)
    if numeric.dtype == object:
        numeric = numeric.astype(float)
    return numeric.astype(np.float32, copy=False)


def is_binary_like(array: np.ndarray) -> bool:
    values = np.asarray(array).reshape(-1)
    if values.size == 0:
        return False

    if np.issubdtype(values.dtype, np.floating):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return False

    if np.issubdtype(values.dtype, np.bool_):
        return True

    unique_values = np.unique(values)
    return np.all(np.isin(unique_values, [0, 1, False, True]))


def is_location_like(array: np.ndarray) -> bool:
    values = np.asarray(array).reshape(-1)
    if values.size == 0:
        return False

    if np.issubdtype(values.dtype, np.floating):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return False

    try:
        rounded = np.rint(values).astype(int)
    except (TypeError, ValueError):
        return False

    return np.all((rounded >= 0) & (rounded <= 20))


def choose_capacity_channel(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        channel = np.asarray(tensor[:, :, index])
        if is_binary_like(channel):
            continue
        if np.allclose(channel, 0):
            continue

        channel_min = float(np.nanmin(channel))
        channel_mean = float(np.nanmean(channel))
        if channel_min >= 0 and channel_mean >= 1:
            return index

    for index in range(tensor.shape[2]):
        channel = np.asarray(tensor[:, :, index])
        if not is_binary_like(channel) and not np.allclose(channel, 0):
            return index

    raise ValueError("Could not identify a usable capacity channel in the tensor.")


def choose_attack_channel(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        if is_binary_like(tensor[:, :, index]):
            return index

    raise ValueError("Could not identify a binary attack channel in the tensor.")


def extract_from_tensor(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    if tensor.shape[1] != 20:
        raise ValueError(
            f"Expected 20 transmission lines in axis 1, got shape {tensor.shape}."
        )

    capacity_channel = choose_capacity_channel(tensor)
    attack_channel = choose_attack_channel(tensor)

    # Use one numeric feature channel as the per-line capacity observation matrix.
    observations = as_float_array(tensor[:, :, capacity_channel])
    # The binary channel marks attacked lines at each timestep.
    attack_binary_matrix = np.asarray(tensor[:, :, attack_channel]) != 0
    attack_occurrence = np.any(attack_binary_matrix, axis=1).astype(np.int64)

    # Convert the attacked line indicator into a 1-based line index, or 0 when no attack occurred.
    attack_location = np.zeros(tensor.shape[0], dtype=np.int64)
    attack_rows = np.flatnonzero(attack_occurrence)
    for row_index in attack_rows:
        line_candidates = np.flatnonzero(attack_binary_matrix[row_index])
        if line_candidates.size == 0:
            continue
        attack_location[row_index] = int(line_candidates[0]) + 1

    return observations, attack_occurrence, attack_location


def find_array_by_hints(arrays: list[LocatedArray], hints: tuple[str, ...]) -> LocatedArray | None:
    for located in arrays:
        path = located.path.lower()
        if any(hint in path for hint in hints):
            return LocatedArray(path=located.path, array=np.asarray(located.array))
    return None


def extract_from_generic_structure(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = iter_arrays(data)

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return extract_from_tensor(data)
        raise ValueError(
            f"Top-level pickle contains a numpy array with unsupported shape {data.shape}. Expected a 3D tensor with 20 lines."
        )

    tensor_candidates = [located for located in arrays if located.array.ndim == 3]
    if tensor_candidates:
        return extract_from_tensor(np.asarray(tensor_candidates[0].array))

    observation_candidate = find_array_by_hints(
        arrays,
        ("observation", "observations", "obs", "capacity", "state", "states", "feature", "features", "x", "data"),
    )
    occurrence_candidate = find_array_by_hints(
        arrays,
        ("attack_occurrence", "attack occurrence", "occurrence", "attack", "label", "labels", "y", "target", "targets"),
    )
    location_candidate = find_array_by_hints(
        arrays,
        ("attack_location", "attack location", "location", "loc"),
    )

    if observation_candidate is None:
        raise ValueError(
            "Could not locate observations. Expected a 3D tensor with 20 lines or a 2D observations array."
        )

    observations = as_float_array(observation_candidate.array)
    if observations.ndim != 2 or observations.shape[1] != 20:
        raise ValueError(f"Observations must have shape (timesteps, 20), got {observations.shape}.")

    if occurrence_candidate is None:
        raise ValueError("Could not locate attack_occurrence labels in the pickle file.")
    attack_occurrence = np.asarray(occurrence_candidate.array).reshape(-1).astype(np.int64)
    if not is_binary_like(attack_occurrence):
        raise ValueError("attack_occurrence must contain only 0/1 values.")

    if location_candidate is None:
        raise ValueError("Could not locate attack_location labels in the pickle file.")
    attack_location = np.asarray(location_candidate.array).reshape(-1).astype(np.int64)
    if not is_location_like(attack_location):
        raise ValueError("attack_location must contain integer values between 0 and 20.")

    return observations, attack_occurrence, attack_location


def extract_split_components(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return extract_from_generic_structure(data)
    except ValueError as exc:
        raise ValueError(f"Failed to extract observations and labels: {exc}") from exc


def select_observed_lines(observations: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
    if observations.ndim != 2:
        raise ValueError(f"Observations must be 2D, got shape {observations.shape}")

    if observations.shape[1] < selected_indices.size:
        raise ValueError(
            f"Cannot select {selected_indices.size} lines from observations with only {observations.shape[1]} columns."
        )

    return observations[:, selected_indices]


def fit_and_transform_scaler(
    train_obs: np.ndarray,
    val_obs: np.ndarray,
    test_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaler.fit(train_obs)
    return scaler.transform(train_obs), scaler.transform(val_obs), scaler.transform(test_obs), scaler


def create_sliding_windows(
    features: np.ndarray,
    occurrence: np.ndarray,
    location: np.ndarray,
    state_targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("Sequence length must be a positive integer.")

    if (
        features.shape[0] != occurrence.shape[0]
        or features.shape[0] != location.shape[0]
        or features.shape[0] != state_targets.shape[0]
    ):
        raise ValueError(
            "Features, attack occurrence, attack location, and state targets must have the same number of timesteps."
        )

    if features.shape[0] < sequence_length:
        raise ValueError(
            f"Not enough timesteps to build sequences of length {sequence_length}: got {features.shape[0]}."
        )

    sequences: list[np.ndarray] = []
    occurrence_targets: list[int] = []
    location_targets: list[int] = []
    state_window_targets: list[np.ndarray] = []

    for current_index in range(sequence_length - 1, features.shape[0]):
        start_index = current_index - sequence_length + 1
        sequences.append(features[start_index : current_index + 1])
        occurrence_targets.append(int(occurrence[current_index]))
        location_targets.append(int(location[current_index]))
        state_window_targets.append(np.asarray(state_targets[current_index], dtype=np.float32))

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(occurrence_targets, dtype=np.int64),
        np.asarray(location_targets, dtype=np.int64),
        np.asarray(state_window_targets, dtype=np.float32),
    )


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)


def verify_saved_outputs(output_dir: Path, expected_shapes: dict[str, tuple[int, ...]]) -> None:
    for name, expected_shape in expected_shapes.items():
        array_path = output_dir / f"{name}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"Expected saved file is missing: {array_path}")

        loaded = np.load(array_path, allow_pickle=False)
        if loaded.shape != expected_shape:
            raise ValueError(
                f"Saved file has unexpected shape: {array_path} -> {loaded.shape}, expected {expected_shape}"
            )

    scaler_path = output_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Expected saved scaler is missing: {scaler_path}")

    with scaler_path.open("rb") as file_handle:
        _ = pickle.load(file_handle)


def print_final_shapes(arrays: dict[str, np.ndarray]) -> None:
    print("\nFinal array shapes:")
    for name, array in arrays.items():
        print(f"  - {name}: {array.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess power grid attack detection pickle files for LSTM training."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Folder containing the pickle files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Folder for preprocessed outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Sliding window length (default: {DEFAULT_SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--po",
        type=float,
        default=DEFAULT_PO,
        help=f"Observation ratio, used to derive the number of observed lines (default: {DEFAULT_PO})",
    )
    parser.add_argument(
        "--selected-lines",
        type=int,
        default=DEFAULT_SELECTED_LINES,
        help=f"Number of observed lines to keep (default: {DEFAULT_SELECTED_LINES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for line selection (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    sequence_length = args.sequence_length
    rng = np.random.default_rng(args.seed)

    train_path = resolve_input_file(data_dir, TRAIN_FILENAME)
    val_path = resolve_input_file(data_dir, VAL_FILENAME)
    test_path = resolve_input_file(data_dir, TEST_FILENAME)

    try:
        # Step 1: load the three raw pickle files.
        print("STEP 1 - Loading pickle files")
        train_data = load_pickle_file(train_path)
        val_data = load_pickle_file(val_path)
        test_data = load_pickle_file(test_path)

        # Step 2: extract observations and attack labels from each split.
        print("STEP 2 - Extracting observations, attack_occurrence, and attack_location")
        train_observations, train_occurrence, train_location = extract_split_components(train_data)
        val_observations, val_occurrence, val_location = extract_split_components(val_data)
        test_observations, test_occurrence, test_location = extract_split_components(test_data)

        if train_observations.shape[1] != 20:
            raise ValueError(
                f"Training observations must contain 20 transmission lines, got shape {train_observations.shape}."
            )

        selected_count = int(args.selected_lines)
        expected_count = int(round(args.po * train_observations.shape[1]))
        if selected_count != expected_count:
            print(
                f"WARNING: Po={args.po} suggests {expected_count} lines, but the script will use {selected_count} lines as requested."
            )

        if selected_count <= 0 or selected_count > train_observations.shape[1]:
            raise ValueError(
                f"Invalid number of selected lines: {selected_count}. Must be between 1 and {train_observations.shape[1]}."
            )

        selected_indices = np.sort(
            rng.choice(train_observations.shape[1], size=selected_count, replace=False)
        )
        print(f"Selected observed line indices: {selected_indices.tolist()}")

        # Step 3: keep the same partially observed lines across train, validation, and test.
        train_features = select_observed_lines(train_observations, selected_indices)
        val_features = select_observed_lines(val_observations, selected_indices)
        test_features = select_observed_lines(test_observations, selected_indices)

        # Step 4: normalize using a scaler fit only on the training split.
        print("STEP 3 - Applying MinMaxScaler normalization")
        scaled_train, scaled_val, scaled_test, scaler = fit_and_transform_scaler(
            train_features,
            val_features,
            test_features,
        )

        # Step 5: build sliding windows for the LSTM input/output pairs.
        print("STEP 4 - Creating sliding window sequences")
        X_train, y_train_occur, y_train_loc, y_train_state = create_sliding_windows(
            scaled_train,
            train_occurrence,
            train_location,
            train_observations,
            sequence_length,
        )
        X_val, y_val_occur, y_val_loc, y_val_state = create_sliding_windows(
            scaled_val,
            val_occurrence,
            val_location,
            val_observations,
            sequence_length,
        )
        X_test, y_test_occur, y_test_loc, y_test_state = create_sliding_windows(
            scaled_test,
            test_occurrence,
            test_location,
            test_observations,
            sequence_length,
        )

        # Step 6: save all arrays and the fitted scaler to disk.
        print("STEP 5 - Saving preprocessed data")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_array(output_dir / "X_train.npy", X_train)
        save_array(output_dir / "X_val.npy", X_val)
        save_array(output_dir / "X_test.npy", X_test)
        save_array(output_dir / "y_train_occur.npy", y_train_occur)
        save_array(output_dir / "y_val_occur.npy", y_val_occur)
        save_array(output_dir / "y_test_occur.npy", y_test_occur)
        save_array(output_dir / "y_train_loc.npy", y_train_loc)
        save_array(output_dir / "y_val_loc.npy", y_val_loc)
        save_array(output_dir / "y_test_loc.npy", y_test_loc)
        save_array(output_dir / "y_train_state.npy", y_train_state)
        save_array(output_dir / "y_val_state.npy", y_val_state)
        save_array(output_dir / "y_test_state.npy", y_test_state)
        save_array(output_dir / "selected_lines.npy", selected_indices.astype(np.int64))
        save_pickle(output_dir / "scaler.pkl", scaler)

        verify_saved_outputs(
            output_dir,
            {
                "X_train": X_train.shape,
                "X_val": X_val.shape,
                "X_test": X_test.shape,
                "y_train_occur": y_train_occur.shape,
                "y_val_occur": y_val_occur.shape,
                "y_test_occur": y_test_occur.shape,
                "y_train_loc": y_train_loc.shape,
                "y_val_loc": y_val_loc.shape,
                "y_test_loc": y_test_loc.shape,
                "y_train_state": y_train_state.shape,
                "y_val_state": y_val_state.shape,
                "y_test_state": y_test_state.shape,
                "selected_lines": selected_indices.shape,
            },
        )

        # Step 7: report the final shapes and confirm the save location.
        print("STEP 6 - Final shapes and confirmation")
        print_final_shapes(
            {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train_occur": y_train_occur,
                "y_val_occur": y_val_occur,
                "y_test_occur": y_test_occur,
                "y_train_loc": y_train_loc,
                "y_val_loc": y_val_loc,
                "y_test_loc": y_test_loc,
                "y_train_state": y_train_state,
                "y_val_state": y_val_state,
                "y_test_state": y_test_state,
                "selected_lines": selected_indices,
            }
        )
        print(f"\nSaved preprocessed data to: {output_dir}")
        print(f"Saved scaler to: {output_dir / 'scaler.pkl'}")
        print("Preprocessing completed successfully.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()