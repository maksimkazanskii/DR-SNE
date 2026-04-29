import numpy as np
import scanpy as sc
import torch
import random
import time
import os
import csv

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import umap
import pacmap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from density_tsne import run_density_tsne


# =========================================================
# SEED
# =========================================================
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# LOAD PBMC
# =========================================================
def load_pbmc(seed=42):
    sc.settings.seed = seed
    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(adata, resolution=1.0)

    X = adata.obsm["X_pca"]
    y = adata.obs["leiden"].astype(int).values

    return X, y


# =========================================================
# DEFINE ANOMALIES (RARE CLUSTERS)
# =========================================================
def anomaly_pbmc(y, threshold=0.05):
    unique, counts = np.unique(y, return_counts=True)
    freq = counts / len(y)

    rare_clusters = unique[freq < threshold]
    labels = np.isin(y, rare_clusters).astype(int)

    print(f"Rare clusters: {rare_clusters}")
    print(f"Anomaly %: {labels.mean() * 100:.2f}%")

    return labels


# =========================================================
# SPLIT: TRAIN ONLY NORMAL, TEST FULL DATASET
# =========================================================
def split_pbmc(X, anomaly_labels):
    normal_idx = np.where(anomaly_labels == 0)[0]

    X_train = X[normal_idx]
    X_test = X
    y_test = anomaly_labels

    return X_train, X_test, y_test


# =========================================================
# ESTIMATE SIGMA FOR KERNEL PROJECTION
# =========================================================
def estimate_sigma(X, n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    D = pairwise_distances(X[idx])
    sigma = np.median(D)
    return max(sigma, 1e-6)


# =========================================================
# STABLE KERNEL PROJECTION
# =========================================================
def kernel_project(X_train, Z_train, X_test, sigma):
    K = rbf_kernel(X_test, X_train, gamma=1.0 / (2.0 * sigma**2))

    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8
    K = K / row_sums

    return K @ Z_train


# =========================================================
# DENSITY SCORE
# =========================================================
def compute_density_scores(Z_train, Z_test, k=30):
    k_eff = min(k, len(Z_train))
    if k_eff < 2:
        raise ValueError("Need at least 2 training points for density estimation.")

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Z_train)

    dist_train, _ = nbrs.kneighbors(Z_train)
    rho_train = (k_eff - 1) / (dist_train[:, 1:].sum(axis=1) + 1e-8)

    dist_test, _ = nbrs.kneighbors(Z_test)
    rho_test = (k_eff - 1) / (dist_test.sum(axis=1) + 1e-8)

    mean_rho = rho_train.mean() + 1e-8
    scores = -np.log(rho_test / mean_rho + 1e-8)

    return scores


# =========================================================
# HIGH-D P MATRIX FOR DENSITY t-SNE
# =========================================================
def compute_P(X, perplexity=30.0, tol=1e-5):
    n = X.shape[0]
    D = pairwise_distances(X, squared=True)

    P = np.zeros((n, n), dtype=np.float32)
    log_perp = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf

        Di = np.delete(D[i], i)

        for _ in range(50):
            Pi = np.exp(-Di * beta)
            sumPi = np.sum(Pi) + 1e-8
            Pi = Pi / sumPi

            H = -np.sum(Pi * np.log(Pi + 1e-8))
            Hdiff = H - log_perp

            if abs(Hdiff) < tol:
                break

            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2

        inds = np.concatenate((np.r_[0:i], np.r_[i + 1:n]))
        P[i, inds] = Pi

    P = (P + P.T) / (2 * n)
    return torch.tensor(P, dtype=torch.float32)


# =========================================================
# KNN DENSITY IN HIGH-D SPACE FOR DENSITY t-SNE
# =========================================================
def compute_knn_density(X, k=30):
    k_eff = min(k, len(X))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="brute").fit(X)
    distances, indices = nbrs.kneighbors(X)

    volume = distances[:, 1:].sum(axis=1) + 1e-8
    density = (k_eff - 1) / volume
    density = density / (density.mean() + 1e-8)

    return density, indices


# =========================================================
# EMBEDDING METHODS
# =========================================================
def run_pca_2d(X, seed=42):
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def run_tsne_2d(X, seed=42, perplexity=30):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        random_state=seed
    )
    return tsne.fit_transform(X)


def run_umap_2d(X, seed=42, n_neighbors=30):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=seed
    )
    return reducer.fit_transform(X)


def run_pacmap_2d(X, seed=42, n_neighbors=10):
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed
    )
    return reducer.fit_transform(X)


def run_densmap_2d(X, seed=42, dens_lambda=2.0, n_neighbors=15):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        densmap=True,
        dens_lambda=dens_lambda,
        dens_frac=0.3,
        output_dens=True,
        random_state=seed
    )

    result = reducer.fit_transform(X, return_dens=True)
    if isinstance(result, tuple):
        return result[0]
    return result


# =========================================================
# SAVE RESULTS
# =========================================================
def save_results_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "param_name", "param_value", "auroc", "auprc", "time_sec"]
        )
        writer.writeheader()
        writer.writerows(results)


# =========================================================
# EVALUATION WRAPPER
# =========================================================
def evaluate_method(
        method_name,
        embed_fn,
        X_train,
        X_test,
        y_test,
        sigma,
        param_name="-",
        param_value="-"
):
    start = time.time()

    Z_train = embed_fn(X_train)
    Z_test = kernel_project(X_train, Z_train, X_test, sigma=sigma)

    scores = compute_density_scores(Z_train, Z_test)

    elapsed = time.time() - start

    auroc = roc_auc_score(y_test, scores)
    auprc = average_precision_score(y_test, scores)

    print(
        f"{method_name:15s} | "
        f"{param_name}={param_value} | "
        f"AUROC={auroc:.4f} | "
        f"AUPRC={auprc:.4f} | "
        f"Time={elapsed:.2f}s"
    )

    return {
        "method": method_name,
        "param_name": param_name,
        "param_value": param_value,
        "auroc": auroc,
        "auprc": auprc,
        "time_sec": elapsed
    }


# =========================================================
# DENSITY t-SNE SWEEP
# =========================================================
def run_density_tsne_sweep(X_train, X_test, y_test, sigma, seed, results):
    print("\n==============================")
    print("Density t-SNE FULL SWEEP")
    print("==============================")

    lambda_list = [
        1e-4, 2e-4, 5e-4,
        1e-3, 2e-3, 5e-3,
        1e-2, 2e-2, 5e-2,
        1e-1
    ]
    k_list = [30, 70, 150, 300]

    P = compute_P(X_train, perplexity=30.0)

    for k in k_list:
        print(f"\n[Density t-SNE] k = {k}")

        rho_high, knn_indices = compute_knn_density(X_train, k=k)

        # Stabilize density targets
        rho_high = np.log(rho_high + 1e-8)
        rho_high = rho_high - rho_high.mean()

        for lam in lambda_list:
            start = time.time()

            Z_train, _ = run_density_tsne(
                X_train,
                P,
                knn_indices,
                rho_high,
                lambda_density=lam,
                seed=seed
            )

            Z_test = kernel_project(X_train, Z_train, X_test, sigma=sigma)
            scores = compute_density_scores(Z_train, Z_test)

            elapsed = time.time() - start
            auroc = roc_auc_score(y_test, scores)
            auprc = average_precision_score(y_test, scores)

            print(
                f"{'Density t-SNE':15s} | "
                f"k={k:3d} | lambda={lam:.4g} | "
                f"AUROC={auroc:.4f} | "
                f"AUPRC={auprc:.4f} | "
                f"Time={elapsed:.2f}s"
            )

            results.append({
                "method": "Density t-SNE",
                "param_name": "k_lambda",
                "param_value": f"{k}_{lam:.4g}",
                "auroc": auroc,
                "auprc": auprc,
                "time_sec": elapsed
            })

    return results


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pbmc_experiment(seed=42):
    set_global_seed(seed)

    print("Loading PBMC...")
    X, y = load_pbmc(seed=seed)

    print("\nDefining anomalies...")
    anomaly_labels = anomaly_pbmc(y)

    print("\nSplitting...")
    X_train, X_test, y_test = split_pbmc(X, anomaly_labels)

    print(f"Train size: {len(X_train)} (normal only)")
    print(f"Test size:  {len(X_test)} (full dataset)")

    sigma = estimate_sigma(X_train, seed=seed)
    print(f"\nEstimated sigma: {sigma:.4f}")

    results = []

    # -----------------------------------------------------
    # RAW baseline
    # -----------------------------------------------------
    print("\nRAW SPACE")
    start = time.time()
    scores = compute_density_scores(X_train, X_test)
    elapsed = time.time() - start
    auroc = roc_auc_score(y_test, scores)
    auprc = average_precision_score(y_test, scores)

    print(f"{'RAW':15s} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} | Time={elapsed:.2f}s")

    results.append({
        "method": "RAW",
        "param_name": "-",
        "param_value": "-",
        "auroc": auroc,
        "auprc": auprc,
        "time_sec": elapsed
    })

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------
    results.append(
        evaluate_method(
            "PCA",
            lambda X_: run_pca_2d(X_, seed=seed),
            X_train, X_test, y_test, sigma,
            "-", "-"
        )
    )

    # -----------------------------------------------------
    # t-SNE
    # -----------------------------------------------------
    for perp in [10, 30, 50]:
        results.append(
            evaluate_method(
                "t-SNE",
                lambda X_, p=perp: run_tsne_2d(X_, seed=seed, perplexity=p),
                X_train, X_test, y_test, sigma,
                "perplexity", perp
            )
        )

    # -----------------------------------------------------
    # UMAP
    # -----------------------------------------------------
    for k in [10, 30, 50]:
        results.append(
            evaluate_method(
                "UMAP",
                lambda X_, kk=k: run_umap_2d(X_, seed=seed, n_neighbors=kk),
                X_train, X_test, y_test, sigma,
                "n_neighbors", k
            )
        )

    # -----------------------------------------------------
    # PaCMAP
    # -----------------------------------------------------
    for k in [10, 15, 30]:
        results.append(
            evaluate_method(
                "PaCMAP",
                lambda X_, kk=k: run_pacmap_2d(X_, seed=seed, n_neighbors=kk),
                X_train, X_test, y_test, sigma,
                "n_neighbors", k
            )
        )

    # -----------------------------------------------------
    # DensMAP
    # -----------------------------------------------------
    for lam in [0.5, 1.0, 2.0, 4.0]:
        results.append(
            evaluate_method(
                "DensMAP",
                lambda X_, ll=lam: run_densmap_2d(X_, seed=seed, dens_lambda=ll, n_neighbors=15),
                X_train, X_test, y_test, sigma,
                "dens_lambda", lam
            )
        )

    # -----------------------------------------------------
    # Density t-SNE FULL SWEEP
    # -----------------------------------------------------
    results = run_density_tsne_sweep(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sigma=sigma,
        seed=seed,
        results=results
    )

    print("\nFINAL RESULTS:")
    for r in results:
        print(r)

    save_results_csv(results, "output/anomaly/pbmc_results.csv")
    print("\nSaved to output/anomaly/pbmc_results.csv")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    run_pbmc_experiment(seed=42)