import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

# Your implementation
from dtsne import (
    compute_knn_density,
    compute_P,
    run_density_tsne,
    continuity,
    density_correlation,
    timed_run
)

# =========================================================
# DATA LOADERS
# =========================================================

def load_digits_data(n_samples=1500):
    X, y = load_digits(return_X_y=True)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=2000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=int)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_fashion_mnist(n_samples=2000):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)

    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=int)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_pbmc():
    # ⚠️ Replace with your real RNA loader
    # Must return X (numpy), y (labels or clusters)
    raise NotImplementedError("Implement your RNA loader here")


# =========================================================
# PLOTTING
# =========================================================

def plot_lambda_progression(Z_list, lambdas, y, dataset_name):
    os.makedirs("output/images", exist_ok=True)

    fig, axes = plt.subplots(1, len(Z_list), figsize=(4 * len(Z_list), 4))

    if len(Z_list) == 1:
        axes = [axes]

    for ax, Z, lam in zip(axes, Z_list, lambdas):
        ax.scatter(Z[:, 0], Z[:, 1], c=y.astype(int), cmap="tab10", s=6)
        ax.set_title(f"λ={lam:.0e}")
        ax.axis("off")

    plt.tight_layout()
    filename = f"output/images/{dataset_name}_lambda_progression.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved progression image → {filename}")


def plot_lambda_metrics(results, dataset_name):
    os.makedirs("output/images", exist_ok=True)

    lambdas = np.array([r["lambda"] for r in results])
    tw = [r["trustworthiness"] for r in results]
    dens = [r["density_corr"] for r in results]

    # ignore λ=0 for log-scale
    lambdas_plot = np.array([l if l > 0 else 1e-6 for l in lambdas])

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas_plot, tw, marker='o', label="Trustworthiness")
    plt.plot(lambdas_plot, dens, marker='o', label="Density Corr")

    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Metric value")
    plt.title(dataset_name)
    plt.legend()

    filename = f"output/images/{dataset_name}_lambda_metrics.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved metric plot → {filename}")


# =========================================================
# ABLATION CORE
# =========================================================

def run_lambda_ablation(X, y, dataset_name):

    print(f"\n=== λ Ablation: {dataset_name} ===")

    print("Computing density...")
    rho_high, knn_indices = compute_knn_density(X)

    print("Computing P...")
    P = compute_P(X)

    # 🔥 LOG-SCALE LAMBDAS
    lambdas = [0] + list(np.logspace(-4.3, -1, 8))

    results = []
    Z_list = []

    for lam in lambdas:
        print(f"\nRunning λ = {lam:.5f}")

        (Z, _), runtime = timed_run(
            f"{dataset_name} (λ={lam:.5f})",
            run_density_tsne,
            X, P, knn_indices, rho_high,
            lambda_density=lam
        )

        Z_list.append(Z)

        tw = trustworthiness(X, Z, n_neighbors=10)
        cont = continuity(X, Z, n_neighbors=10)
        dens = density_correlation(Z, knn_indices, rho_high)

        print(
            f"{dataset_name} | λ={lam:.5f} | "
            f"TW={tw:.4f} | CONT={cont:.4f} | "
            f"DENS={dens:.4f} | TIME={runtime:.2f}"
        )

        results.append({
            "lambda": lam,
            "trustworthiness": tw,
            "continuity": cont,
            "density_corr": dens,
            "time_sec": runtime
        })

    # =========================
    # SAVE CSV
    # =========================
    os.makedirs("output", exist_ok=True)

    csv_path = f"output/{dataset_name}_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV → {csv_path}")

    # =========================
    # SAVE FIGURES
    # =========================
    plot_lambda_progression(Z_list, lambdas, y, dataset_name)
    plot_lambda_metrics(results, dataset_name)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    np.random.seed(42)

    # DIGITS
    X, y = load_digits_data()
    run_lambda_ablation(X, y, "digits")

    # MNIST
    X, y = load_mnist()
    run_lambda_ablation(X, y, "mnist")

    # FASHION-MNIST
    X, y = load_fashion_mnist()
    run_lambda_ablation(X, y, "fashion_mnist")

    # RNA (PBMC)
    # X, y = load_pbmc()
    # run_lambda_ablation(X, y, "rna")