from __future__ import annotations

import importlib
import sys


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "torch",
    "streamlit",
]


def main() -> None:
    print(f"Python version: {sys.version}")
    print()

    missing = []
    for package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package_name} ({version})")
        except ModuleNotFoundError:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)

    print()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU. For GPU acceleration on Windows, install the CUDA build of")
            print("PyTorch, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ModuleNotFoundError:
        pass

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        raise SystemExit(1)

    print("\nAll required packages are installed.")


if __name__ == "__main__":
    main()
