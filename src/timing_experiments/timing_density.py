import os

# =========================================================
# HARD CONTROL OF THREADING (CRITICAL)
# MUST be set BEFORE numpy / sklearn import
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import csv
import random
import gc

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comparison import compute_knn_density, compute_P
from density_tsne import run_density_tsne
from run_all_datasets import load_fashion_mnist


# =========================================================
# CONFIG
# =========================================================
N_RUNS = 7
LAMBDA_DENSITY = 0.01
SEED = 42

os.makedirs("output/timing", exist_ok=True)


# =========================================================
# FULL REPRODUCIBILITY
# =========================================================
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


# =========================================================
# CORE TIMING
# =========================================================
def run_once(X: np.ndarray) -> float:
    start = time.perf_counter()

    rho_high, knn_indices = compute_knn_density(X)
    P = compute_P(X)

    _, _ = run_density_tsne(
        X,
        P,
        knn_indices,
        rho_high,
        lambda_density=LAMBDA_DENSITY,
        seed=SEED
    )

    return time.perf_counter() - start


def run_repeated(X: np.ndarray):
    set_global_seed(SEED)

    times = []

    # ---- warm-up (not measured) ----
    print("  Warm-up run...")
    _ = run_once(X)

    # ---- measured runs ----
    for i in range(N_RUNS):
        print(f"  Run {i + 1}/{N_RUNS}")
        t = run_once(X)
        times.append(t)

    times = np.array(times, dtype=float)

    print(f"  Raw times: {times}")
    print(f"  Sorted: {np.sort(times)}")

    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "times": times
    }


# =========================================================
# SAMPLE SCALING
# =========================================================
def experiment_samples():
    print("\n=== SAMPLE SCALING EXPERIMENT (FASHION-MNIST) ===")

    sample_sizes = [1000, 3000, 5000, 7000, 9000, 11000, 13000]
    results = []

    # ---- LOAD ONCE (IMPORTANT) ----
    X_full, y_full = load_fashion_mnist(n_samples=max(sample_sizes))
    X_full = StandardScaler().fit_transform(X_full)

    for n in sample_sizes:
        print(f"\nSamples: {n}")

        X = X_full[:n]

        stats = run_repeated(X)

        print(
            f"Time: median={stats['median']:.3f}, "
            f"mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}"
        )

        row = {
            "n_samples": int(n),
            "median_time": stats["median"],
            "mean_time": stats["mean"],
            "std_time": stats["std"],
        }

        for j, t in enumerate(stats["times"], start=1):
            row[f"run_{j}"] = float(t)

        results.append(row)

    fieldnames = ["n_samples", "median_time", "mean_time", "std_time"] + \
                 [f"run_{i}" for i in range(1, N_RUNS + 1)]

    with open("output/timing/n.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved → output/timing/n.csv")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n==============================")
    print("Density t-SNE Timing Script")
    print("==============================")
    print("\nParameters:")
    print(f"Runs per config: {N_RUNS}")
    print(f"Lambda density: {LAMBDA_DENSITY}")
    print(f"Seed: {SEED}")
    gc.disable()
    experiment_samples()


if __name__ == "__main__":
    main()