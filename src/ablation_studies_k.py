import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from dtsne import (
    compute_knn_density,
    compute_P,
    run_density_tsne,
    continuity,
    density_correlation,
    timed_run
)

import scanpy as sc


# =========================================================
# DATA LOADERS
# =========================================================

def load_digits_data(n_samples=1500):
    X, y = load_digits(return_X_y=True)
    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=3000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    return X, y


def load_fashion_mnist(n_samples=3000):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    idx = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    return X, y


def load_pbmc():
    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)

    X = adata.X.toarray()
    y = adata.obs["louvain"].astype(int).values

    X = np.log1p(X)
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    return X, y


# =========================================================
# PLOTTING
# =========================================================

def plot_k_metrics(results, dataset_name):
    os.makedirs("output/images", exist_ok=True)

    ks = np.array([r["k"] for r in results])
    tw = np.array([r["trustworthiness"] for r in results])
    dens = np.array([r["density_corr"] for r in results])

    plt.figure(figsize=(6, 4))
    plt.plot(ks, tw, marker='o', label="Trustworthiness")
    plt.plot(ks, dens, marker='o', label="Density Corr")

    plt.xlabel("k (neighbors)")
    plt.ylabel("Metric value")
    plt.title(f"{dataset_name} (k ablation)")
    plt.legend()
    plt.grid(True)

    filename = f"output/images/{dataset_name}_k_metrics.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_k_tradeoff(results, dataset_name):
    os.makedirs("output/images", exist_ok=True)

    tw = np.array([r["trustworthiness"] for r in results])
    dens = np.array([r["density_corr"] for r in results])
    ks = [r["k"] for r in results]

    plt.figure(figsize=(6, 5))
    plt.plot(tw, dens, marker='o')

    for i, k in enumerate(ks):
        plt.text(tw[i], dens[i], str(k), fontsize=7)

    plt.xlabel("Trustworthiness")
    plt.ylabel("Density Correlation")
    plt.title(f"{dataset_name} trade-off (k)")
    plt.grid(True)

    filename = f"output/images/{dataset_name}_k_tradeoff.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_k_progression(Z_list, ks, y, dataset_name):
    os.makedirs("output/images", exist_ok=True)

    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4))

    for ax, Z, k in zip(axes, Z_list, ks):
        ax.scatter(Z[:, 0], Z[:, 1], c=y.astype(int), cmap="tab10", s=6)
        ax.set_title(f"k={k}")
        ax.axis("off")

    plt.tight_layout()
    filename = f"output/images/{dataset_name}_k_progression.png"
    plt.savefig(filename, dpi=300)
    plt.close()


# =========================================================
# ABLATION
# =========================================================

def run_k_ablation(X, y, dataset_name, lambda_density):

    print(f"\n=== k Ablation: {dataset_name} ===")

    # 🔥 ONLY CHANGE: dense k grid
    ks = list(range(5, 55, 5))

    results = []
    Z_list = []

    for k in ks:
        print(f"\nRunning k = {k}")

        rho_high, knn_indices = compute_knn_density(X, k=k)
        P = compute_P(X)

        (Z, _), runtime = timed_run(
            f"{dataset_name} (k={k})",
            run_density_tsne,
            X, P, knn_indices, rho_high,
            lambda_density=lambda_density
        )

        Z_list.append(Z)

        tw = trustworthiness(X, Z, n_neighbors=10)
        cont = continuity(X, Z, n_neighbors=10)
        dens = density_correlation(Z, knn_indices, rho_high)

        results.append({
            "k": k,
            "trustworthiness": tw,
            "continuity": cont,
            "density_corr": dens,
            "time_sec": runtime
        })

        print(
            f"k={k} | TW={tw:.4f} | CONT={cont:.4f} | "
            f"DENS={dens:.4f} | TIME={runtime:.2f}"
        )

    # SAVE CSV
    os.makedirs("output", exist_ok=True)
    csv_path = f"output/{dataset_name}_k_ablation.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # SAVE FIGURES
    plot_k_metrics(results, dataset_name)
    plot_k_tradeoff(results, dataset_name)
    plot_k_progression(Z_list, ks, y, dataset_name)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    np.random.seed(42)

    X, y = load_digits_data()
    run_k_ablation(X, y, "digits", lambda_density=0.001)

    X, y = load_mnist()
    run_k_ablation(X, y, "mnist", lambda_density=0.002)

    X, y = load_fashion_mnist()
    run_k_ablation(X, y, "fashion_mnist", lambda_density=0.003)

    X, y = load_pbmc()
    run_k_ablation(X, y, "rna", lambda_density=0.003)