from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run explore_data.py. Install it and try again."
    ) from exc


DEFAULT_PICKLE_PATH = Path("data/data_case14_train.pkl")


@dataclass
class FoundArray:
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


def describe_value(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    return type(value).__name__


def walk_structure(
    value: Any,
    path: str = "root",
    max_depth: int = 5,
    seen: set[int] | None = None,
) -> Iterable[str]:
    if seen is None:
        seen = set()

    object_id = id(value)
    if object_id in seen:
        yield f"{path}: <circular reference>"
        return
    seen.add(object_id)

    yield f"{path}: {describe_value(value)}"

    if max_depth <= 0:
        return

    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_structure(child, f"{path}.{key}", max_depth - 1, seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_structure(child, f"{path}[{index}]", max_depth - 1, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for key, child in vars(value).items():
            yield from walk_structure(child, f"{path}.{key}", max_depth - 1, seen)


def find_numpy_arrays(value: Any, path: str = "root", seen: set[int] | None = None) -> list[FoundArray]:
    if seen is None:
        seen = set()

    object_id = id(value)
    if object_id in seen:
        return []
    seen.add(object_id)

    arrays: list[FoundArray] = []

    if isinstance(value, np.ndarray):
        arrays.append(FoundArray(path=path, array=value))
        return arrays

    if isinstance(value, dict):
        for key, child in value.items():
            arrays.extend(find_numpy_arrays(child, f"{path}.{key}", seen))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            arrays.extend(find_numpy_arrays(child, f"{path}[{index}]", seen))
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for key, child in vars(value).items():
            arrays.extend(find_numpy_arrays(child, f"{path}.{key}", seen))

    return arrays


def print_structure_summary(data: Any) -> None:
    print(f"Top-level type: {type(data).__name__}")
    print(f"Top-level structure: {describe_value(data)}")

    if isinstance(data, dict):
        print("Dictionary keys:")
        for key in data.keys():
            print(f"  - {key}")

    print("\nStructure preview:")
    for line in walk_structure(data):
        print(line)


def print_array_shapes(data: Any) -> list[FoundArray]:
    arrays = find_numpy_arrays(data)
    if not arrays:
        print("\nNo numpy arrays found inside the loaded object.")
        return []

    print("\nNumpy arrays found:")
    for found in arrays:
        print(f"  - {found.path}: shape={found.array.shape}, dtype={found.array.dtype}")
    return arrays


def choose_observation_arrays(data: Any, arrays: list[FoundArray]) -> list[FoundArray]:
    observation_names = ("observation", "observations", "obs", "x", "features", "feature", "state", "states", "data", "capacity")

    if isinstance(data, dict):
        matches = [
            FoundArray(path=f"root.{key}", array=np.asarray(value))
            for key, value in data.items()
            if isinstance(value, np.ndarray) and any(name in str(key).lower() for name in observation_names)
        ]
        if matches:
            return matches

    arrays_by_name = [
        found
        for found in arrays
        if any(name in found.path.lower() for name in observation_names)
    ]
    if arrays_by_name:
        return arrays_by_name

    return [found for found in arrays if np.issubdtype(found.array.dtype, np.number)]


def print_observation_stats(data: Any, arrays: list[FoundArray]) -> None:
    observation_arrays = choose_observation_arrays(data, arrays)
    if not observation_arrays:
        print("\nNo numeric observation arrays found for min/max/mean statistics.")
        return

    print("\nObservation statistics:")
    for found in observation_arrays:
        array = np.asarray(found.array)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            print(f"  - {found.path}: skipped (non-numeric or empty)")
            continue

        print(
            f"  - {found.path}: min={np.min(array):.6g}, max={np.max(array):.6g}, mean={np.mean(array):.6g}"
        )


def choose_label_arrays(data: Any, arrays: list[FoundArray]) -> list[FoundArray]:
    label_names = ("attack", "attacks", "label", "labels", "y", "target", "targets", "attack_label")

    if isinstance(data, dict):
        matches = [
            FoundArray(path=f"root.{key}", array=np.asarray(value))
            for key, value in data.items()
            if isinstance(value, np.ndarray) and any(name in str(key).lower() for name in label_names)
        ]
        if matches:
            return matches

    arrays_by_name = [
        found
        for found in arrays
        if any(name in found.path.lower() for name in label_names)
    ]
    if arrays_by_name:
        return arrays_by_name

    binary_candidates = []
    for found in arrays:
        array = np.asarray(found.array)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            continue

        unique_values = np.unique(array)
        if unique_values.size <= 5 and np.all(np.isin(unique_values, [0, 1, False, True])):
            binary_candidates.append(found)

    return binary_candidates


def print_attack_counts(data: Any, arrays: list[FoundArray]) -> None:
    label_arrays = choose_label_arrays(data, arrays)
    if not label_arrays:
        print("\nNo binary attack label arrays found for attack/no-attack counts.")
        return

    print("\nAttack vs no-attack counts:")
    for found in label_arrays:
        labels = np.asarray(found.array).reshape(-1)
        labels = labels[~np.isnan(labels)] if np.issubdtype(labels.dtype, np.floating) else labels

        if labels.size == 0:
            print(f"  - {found.path}: skipped (empty)")
            continue

        if np.issubdtype(labels.dtype, np.bool_):
            attack_count = int(np.count_nonzero(labels))
            no_attack_count = int(labels.size - attack_count)
        else:
            attack_count = int(np.count_nonzero(labels != 0))
            no_attack_count = int(np.count_nonzero(labels == 0))

        print(f"  - {found.path}: attack={attack_count}, no_attack={no_attack_count}, total={labels.size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a pickle file containing power grid simulation data."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PICKLE_PATH),
        help=f"Path to pickle file (default: {DEFAULT_PICKLE_PATH})",
    )
    args = parser.parse_args()

    path = Path(args.path)

    try:
        data = load_pickle_file(path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print_structure_summary(data)
    arrays = print_array_shapes(data)
    print_observation_stats(data, arrays)
    print_attack_counts(data, arrays)

    print("\nSETUP COMPLETE")


if __name__ == "__main__":
    main()