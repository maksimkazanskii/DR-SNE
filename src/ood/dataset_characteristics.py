import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets import (
    load_shuttle,
    load_spiral,
    load_fashion_dino,
    load_fashion_anomaly,
    load_kddcup99_small,
    load_resnet_cifar10,
    load_pbmc,
    load_tumor,
    load_swiss_density
)

# =========================================================
# DATASETS
# =========================================================

DATASETS = {
    "cifar": load_resnet_cifar10,
    "thyroid": load_tumor,
    "pbmc": load_pbmc,
    "shuttle": load_shuttle,
    "fashion_dino": load_fashion_dino,
    "swiss": load_swiss_density,
    "fashion": load_fashion_anomaly,
    "synthetic": load_spiral,
}

# =========================================================
# MAIN
# =========================================================

def main():
    print("\n=== DATASET STATISTICS ===\n")

    for name, loader in DATASETS.items():
        X, y = loader()

        X = np.asarray(X)
        y = np.asarray(y)

        assert len(X) == len(y), f"{name}: X/y mismatch"
        assert y.ndim == 1, f"{name}: y must be 1D"

        n_samples = X.shape[0]
        n_features = X.shape[1] if X.ndim > 1 else 1
        anomaly_fraction = float(np.mean(y))

        print(f"{name}")
        print(f"  samples: {n_samples}")
        print(f"  features: {n_features}")
        print(f"  anomaly_fraction: {anomaly_fraction:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()