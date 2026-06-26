from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: matplotlib is required to run visualize_data.py. Install it and try again."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: numpy is required to run visualize_data.py. Install it and try again."
    ) from exc


DEFAULT_PICKLE_PATH = Path("data_case14_train.pkl")
DEFAULT_OUTPUT_PATH = Path("plots/data_overview.png")


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


def resolve_existing_path(path: Path) -> Path:
    candidates: list[Path] = [path]

    if not path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                Path.cwd() / path,
                Path.cwd() / path.name,
                script_dir / path,
                script_dir / path.name,
            ]
        )

        if path.parts and path.parts[0] == "data":
            candidates.extend([Path.cwd() / path.name, script_dir / path.name])
        else:
            candidates.extend([Path.cwd() / "data" / path.name, script_dir / "data" / path.name])

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.resolve(strict=False))
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Pickle file not found. Tried: " + ", ".join(str(candidate) for candidate in unique_candidates)
    )


def choose_feature_index(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        if is_binary_like(tensor[:, :, index]):
            continue

        channel = np.asarray(tensor[:, :, index])
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

    return 0


def choose_attack_index(tensor: np.ndarray) -> int:
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tensor.shape}")

    for index in range(tensor.shape[2] - 1, -1, -1):
        if is_binary_like(tensor[:, :, index]):
            return index

    return tensor.shape[2] - 1


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


def normalize_numeric_array(array: np.ndarray) -> np.ndarray:
    numeric = np.asarray(array)
    if numeric.dtype == object:
        numeric = numeric.astype(float)
    return numeric


def choose_observation_array(data: Any, arrays: list[LocatedArray]) -> LocatedArray:
    name_hints = ("observation", "observations", "obs", "capacity", "state", "states", "feature", "features", "x", "data")

    if isinstance(data, np.ndarray) and data.ndim == 3:
        feature_index = choose_feature_index(data)
        return LocatedArray(path=f"root[:, :, {feature_index}]", array=normalize_numeric_array(data[:, :, feature_index]))

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-1] == 20:
                return LocatedArray(path=f"root.{key}", array=normalize_numeric_array(value))

    shape_matches = [
        located
        for located in arrays
        if located.array.ndim >= 2 and located.array.shape[-1] == 20
    ]
    if shape_matches:
        return shape_matches[0]

    name_matches = [
        located
        for located in arrays
        if any(hint in located.path.lower() for hint in name_hints) and located.array.ndim >= 2
    ]
    if name_matches:
        return name_matches[0]

    raise ValueError("Could not locate an observations array with shape timesteps x 20 lines.")


def choose_label_array(data: Any, arrays: list[LocatedArray], timesteps: int) -> LocatedArray:
    name_hints = ("attack", "attacks", "label", "labels", "target", "targets", "y")

    if isinstance(data, np.ndarray) and data.ndim == 3:
        attack_index = choose_attack_index(data)
        labels = np.any(np.asarray(data[:, :, attack_index]) != 0, axis=1).astype(int)
        return LocatedArray(path=f"root[:, :, {attack_index}]", array=labels)

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, np.ndarray):
                continue
            flattened = np.asarray(value).reshape(-1)
            if flattened.size == timesteps and is_binary_like(flattened):
                return LocatedArray(path=f"root.{key}", array=flattened)

    name_matches = [
        located
        for located in arrays
        if any(hint in located.path.lower() for hint in name_hints)
    ]
    for located in name_matches:
        flattened = np.asarray(located.array).reshape(-1)
        if flattened.size == timesteps and is_binary_like(flattened):
            return LocatedArray(path=located.path, array=flattened)

    binary_matches = []
    for located in arrays:
        flattened = np.asarray(located.array).reshape(-1)
        if flattened.size == timesteps and is_binary_like(flattened):
            binary_matches.append(LocatedArray(path=located.path, array=flattened))

    if binary_matches:
        return binary_matches[0]

    raise ValueError("Could not locate an attack labels array matching the number of timesteps.")


def is_binary_like(array: np.ndarray) -> bool:
    cleaned = np.asarray(array)
    if cleaned.size == 0:
        return False

    if np.issubdtype(cleaned.dtype, np.bool_):
        return True

    if np.issubdtype(cleaned.dtype, np.floating):
        cleaned = cleaned[~np.isnan(cleaned)]
        if cleaned.size == 0:
            return False

    unique_values = np.unique(cleaned)
    return np.all(np.isin(unique_values, [0, 1, False, True]))


def summarize_counts(labels: np.ndarray) -> tuple[int, int, int, float]:
    flattened = np.asarray(labels).reshape(-1)
    if np.issubdtype(flattened.dtype, np.floating):
        flattened = flattened[~np.isnan(flattened)]

    attack_timesteps = int(np.count_nonzero(flattened != 0))
    normal_timesteps = int(np.count_nonzero(flattened == 0))
    total_timesteps = int(flattened.size)
    attack_percentage = (attack_timesteps / total_timesteps * 100.0) if total_timesteps else 0.0
    return total_timesteps, attack_timesteps, normal_timesteps, attack_percentage


def prepare_plot_data(observations: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = normalize_numeric_array(observations)
    if obs.ndim != 2:
        raise ValueError(f"Observations array must be 2D, got shape {obs.shape}")

    if obs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Observations and labels must have the same number of timesteps: {obs.shape[0]} vs {labels.shape[0]}"
        )

    return obs, np.asarray(labels).reshape(-1)


def create_figure(observations: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    total_timesteps, attack_timesteps, normal_timesteps, attack_percentage = summarize_counts(labels)
    obs, labels_flat = prepare_plot_data(observations, labels)

    timesteps = np.arange(obs.shape[0])
    preview_limit = min(500, obs.shape[0])
    preview_timesteps = timesteps[:preview_limit]
    preview_obs = obs[:preview_limit]
    preview_labels = labels_flat[:preview_limit]

    attack_mask = labels_flat.astype(bool) if labels_flat.dtype == bool else labels_flat != 0

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), constrained_layout=True)

    axes[0].set_title("Power Line Capacity During Attack")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Capacity")

    line_indices = np.linspace(0, obs.shape[1] - 1, 4, dtype=int) if obs.shape[1] >= 4 else np.arange(obs.shape[1])
    line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for color, line_index in zip(line_colors, line_indices, strict=False):
        axes[0].plot(
            preview_timesteps,
            preview_obs[:, line_index],
            color=color,
            linewidth=1.8,
            label=f"Line {line_index + 1}",
        )

    attack_regions = np.where(preview_labels != 0)[0]
    if attack_regions.size:
        start = attack_regions[0]
        prev = attack_regions[0]
        for index in attack_regions[1:]:
            if index == prev + 1:
                prev = index
                continue
            axes[0].axvspan(start, prev, color="red", alpha=0.12)
            start = prev = index
        axes[0].axvspan(start, prev, color="red", alpha=0.12)

    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Attack Distribution")
    axes[1].pie(
        [attack_timesteps, normal_timesteps],
        labels=["Attack", "Normal"],
        colors=["red", "green"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].axis("equal")

    axes[2].set_title("Histogram of Capacity Values")
    attack_values = obs[attack_mask]
    normal_values = obs[~attack_mask]

    if normal_values.size:
        axes[2].hist(
            normal_values.reshape(-1),
            bins=50,
            alpha=0.55,
            color="green",
            label="Normal periods",
        )
    if attack_values.size:
        axes[2].hist(
            attack_values.reshape(-1),
            bins=50,
            alpha=0.55,
            color="red",
            label="Attack periods",
        )

    axes[2].set_xlabel("Capacity")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Total timesteps: {total_timesteps}")
    print(f"Attack timesteps: {attack_timesteps}")
    print(f"Normal timesteps: {normal_timesteps}")
    print(f"Attack percentage: {attack_percentage:.2f}%")
    print(f"Saved figure to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize power grid attack detection data.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PICKLE_PATH),
        help=f"Path to pickle file (default: {DEFAULT_PICKLE_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output image path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    input_path = resolve_existing_path(Path(args.path))
    output_path = Path(args.output)

    try:
        data = load_pickle_file(input_path)
        arrays = iter_arrays(data)
        observations_location = choose_observation_array(data, arrays)
        labels_location = choose_label_array(data, arrays, observations_location.array.shape[0])
        create_figure(observations_location.array, labels_location.array, output_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()