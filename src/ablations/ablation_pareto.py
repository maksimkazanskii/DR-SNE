import os
import csv
import time
import numpy as np
import scanpy as sc

from sklearn.datasets import load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from density_tsne import run_density_tsne

# =========================================================
# SETTINGS
# =========================================================
LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
SEED = 42

OUT_DIR = "output/ablation_pareto"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# UTILS
# =========================================================
def timed_run(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


# =========================================================
# METRICS
# =========================================================
def compute_knn_density(X, k=300):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    volume = distances[:, 1:].sum(axis=1) + 1e-8
    density = (k - 1) / volume
    density /= density.mean() + 1e-8

    return density, indices


def density_correlation(Z, knn_indices, rho_high):
    rho_low = []

    for i in range(len(Z)):
        neighbors = knn_indices[i][1:]
        dists = np.linalg.norm(Z[i] - Z[neighbors], axis=1)
        volume = dists.sum() + 1e-8
        rho_low.append(len(neighbors) / volume)

    rho_low = np.array(rho_low)
    rho_low /= rho_low.mean() + 1e-8

    return np.corrcoef(
        np.log(rho_high + 1e-8),
        np.log(rho_low + 1e-8)
    )[0, 1]


def compute_P(X, perplexity=30.0):
    import torch

    n = X.shape[0]
    D = pairwise_distances(X, squared=True)

    P = np.zeros((n, n), dtype=np.float32)
    log_perp = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        Di = np.delete(D[i], i)

        for _ in range(50):
            Pi = np.exp(-Di * beta)
            Pi /= np.sum(Pi) + 1e-8

            H = -np.sum(Pi * np.log(Pi + 1e-8))

            if abs(H - log_perp) < 1e-5:
                break

            beta *= 2 if H > log_perp else 0.5

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi

    P = (P + P.T) / (2 * n)
    return torch.tensor(P, dtype=torch.float32)


# =========================================================
# DATASETS
# =========================================================
def load_digits_data():
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=5000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_fashion_mnist(n_samples=5000):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_pbmc():
    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30)

    sc.pp.neighbors(adata, n_neighbors=300, n_pcs=20)
    sc.tl.louvain(adata, resolution=1.0)

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values

    return X, y


def load_synthetic_density(n_samples=5000, n_clusters=4, dim=50):
    np.random.seed(42)

    X_list, y_list = [], []

    variances = np.linspace(0.2, 2.0, n_clusters)
    samples_per_cluster = n_samples // n_clusters

    for i, var in enumerate(variances):
        mean = np.random.randn(dim) * 5
        cov = np.eye(dim) * var

        X_cluster = np.random.multivariate_normal(mean, cov, size=samples_per_cluster)

        X_list.append(X_cluster)
        y_list.append(np.full(samples_per_cluster, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    return X, y


def load_spiral_density(n_samples=5000):
    t = np.linspace(0, 4 * np.pi, n_samples)

    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)

    X = np.stack([x, y], axis=1)

    return X, np.zeros(n_samples)


# =========================================================
# CORE
# =========================================================
def run_pareto_ablation(X, dataset_name, lambdas, seed=42):
    print(f"\n=== {dataset_name} ===")

    np.random.seed(seed)

    rho_high, knn_indices = compute_knn_density(X)
    P = compute_P(X)

    results = []

    for lam in lambdas:
        print(f"λ = {lam}")

        (Z, _), _ = timed_run(
            run_density_tsne,
            X=X,
            P=P,
            knn_indices=knn_indices,
            rho_high=rho_high,
            lambda_density=lam,
            seed=seed,
        )

        tw = trustworthiness(X, Z, n_neighbors=10)
        dens = density_correlation(Z, knn_indices, rho_high)

        print(f"TW={tw:.4f} | DENS={dens:.4f}")

        results.append({
            "lambda": lam,
            "trustworthiness": tw,
            "density_corr": dens,
        })

    return results


def save_results(dataset_name, results):
    path = os.path.join(OUT_DIR, f"{dataset_name}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lambda", "trustworthiness", "density_corr"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved -> {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    datasets = [
        ("pbmc", load_pbmc),
        ("digits", load_digits_data),
        ("mnist", lambda: load_mnist(5000)),
        ("fashion_mnist", lambda: load_fashion_mnist(5000)),
        ("synthetic_density", load_synthetic_density),
        ("spiral_density", load_spiral_density),
    ]

    for name, loader in datasets:
        print("\n========================")
        print(f"DATASET: {name}")
        print("========================")

        X, y = loader()

        results = run_pareto_ablation(
            X,
            dataset_name=name,
            lambdas=LAMBDAS,
            seed=SEED
        )

        save_results(name, results)


if __name__ == "__main__":
    main()