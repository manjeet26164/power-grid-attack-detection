from importlib import import_module


def print_version(name, version):
    print(f"{name}: {version}")


def load_package(name):
    try:
        module = import_module(name)
        version = getattr(module, "__version__", "unknown")
        return module, version
    except ModuleNotFoundError:
        return None, "not installed"


def main():
    tensorflow, tensorflow_version = load_package("tensorflow")
    numpy, numpy_version = load_package("numpy")
    pandas, pandas_version = load_package("pandas")
    matplotlib, matplotlib_version = load_package("matplotlib")
    seaborn, seaborn_version = load_package("seaborn")
    sklearn, sklearn_version = load_package("sklearn")

    print_version("tensorflow", tensorflow_version)
    print_version("numpy", numpy_version)
    print_version("pandas", pandas_version)
    print_version("matplotlib", matplotlib_version)
    print_version("seaborn", seaborn_version)
    print_version("sklearn", sklearn_version)

    if tensorflow is not None:
        gpus = tensorflow.config.list_physical_devices("GPU")
        if gpus:
            print("GPU available: Yes")
            print(f"GPU Name: {gpus[0].name}")
        else:
            print("GPU available: No")
    else:
        print("GPU available: Unknown (tensorflow not installed)")

    print("SETUP COMPLETE")


if __name__ == "__main__":
    main()