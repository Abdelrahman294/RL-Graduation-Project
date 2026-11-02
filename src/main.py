"""Check dependencies, then train and evaluate the temporal DQN."""

from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
import argparse


DEPENDENCIES = (
    ("NumPy", "numpy", "numpy"),
    ("Pandas", "pandas", "pandas"),
    ("scikit-learn", "sklearn", "scikit-learn"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("PyYAML", "yaml", "PyYAML"),
    ("joblib", "joblib", "joblib"),
    ("Gymnasium", "gymnasium", "gymnasium"),
    ("Stable-Baselines3", "stable_baselines3", "stable-baselines3"),
    ("PyTorch", "torch", "torch"),
)


def check_dependencies() -> bool:
    print("Checking temporal project dependencies:", flush=True)
    missing = []
    for label, module_name, distribution_name in DEPENDENCIES:
        try:
            if find_spec(module_name) is None:
                raise ModuleNotFoundError(f"No importable module named {module_name}")
            installed_version = version(distribution_name)
            print(f"  [OK] {label} {installed_version}", flush=True)
        except (ImportError, PackageNotFoundError, OSError) as exc:
            print(f"  [MISSING] {label}: {exc}", flush=True)
            missing.append(label)
    if missing:
        print("\nInstall missing dependencies with: python -m pip install -r requirements.txt", flush=True)
        return False
    print("All project dependencies are available.\n", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the temporal GNSS DQN")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if not check_dependencies():
        raise SystemExit(1)

    # Import project modules only after the dependency report is printed.
    from gnss_temporal.config import load_config
    from gnss_temporal.evaluation import evaluate_model
    from gnss_temporal.training import train_model

    config, _ = load_config(args.config)
    train_model(config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
