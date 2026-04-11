import os
import csv
import time
import numpy as np
import scanpy as sc

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from density_tsne import run_density_tsne

# =========================================================
# SETTINGS
# =========================================================
N_RUNS = 3
BASE_SEED = 42
LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

OUT_DIR = "output/ablation_lambda/3run"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# TIMING
# =========================================================
def timed_run(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


# =========================================================
# DATA (PBMC)
# =========================================================
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


# =========================================================
# METRICS
# =========================================================
def continuity(X, Z, n_neighbors=10):
    return trustworthiness(Z, X, n_neighbors=n_neighbors)


def stress_metric(X, Z):
    D_high = pairwise_distances(X)
    D_low = pairwise_distances(Z)

    num = np.sum((D_high - D_low) ** 2)
    den = np.sum(D_high ** 2) + 1e-8
    return np.sqrt(num / den)


def compute_silhouette(Z, y):
    if y is None:
        return np.nan

    _, y_encoded = np.unique(y, return_inverse=True)

    if len(np.unique(y_encoded)) < 2:
        return np.nan

    counts = np.bincount(y_encoded)
    if np.min(counts) < 2:
        return np.nan

    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        return np.nan

    if np.allclose(Z, Z[0]):
        return np.nan

    try:
        return float(silhouette_score(Z, y_encoded))
    except:
        return np.nan


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
    from sklearn.metrics import pairwise_distances
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
# CORE RUN
# =========================================================
def run_single_ablation(X, y, seed):
    np.random.seed(seed)

    rho_high, knn_indices = compute_knn_density(X, k=300)
    P = compute_P(X)

    results = []

    for lam in LAMBDAS:
        print(f"\nSeed={seed} | λ={lam}")

        (Z, _), runtime = timed_run(
            run_density_tsne,
            X=X,
            P=P,
            knn_indices=knn_indices,
            rho_high=rho_high,
            lambda_density=lam,
            seed=seed,
        )

        tw = trustworthiness(X, Z, n_neighbors=10)
        cont = continuity(X, Z)
        dens = density_correlation(Z, knn_indices, rho_high)
        stress = stress_metric(X, Z)
        sil = compute_silhouette(Z, y)

        results.append({
            "lambda": lam,
            "trustworthiness": tw,
            "continuity": cont,
            "density_corr": dens,
            "silhouette": sil,
            "stress": stress,
            "time_sec": runtime
        })

    return results


# =========================================================
# AGGREGATION
# =========================================================
def aggregate_results(all_runs):
    agg = {}

    for run in all_runs:
        for row in run:
            lam = row["lambda"]

            if lam not in agg:
                agg[lam] = {k: [] for k in row if k != "lambda"}

            for k, v in row.items():
                if k != "lambda":
                    agg[lam][k].append(v)

    final = []

    for lam, metrics in agg.items():
        row = {"lambda": lam}

        for k, values in metrics.items():
            row[f"{k}_mean"] = np.nanmean(values)
            row[f"{k}_std"] = np.nanstd(values)

        final.append(row)

    return final


# =========================================================
# SAVE
# =========================================================
def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    X, y = load_pbmc()

    all_runs = []

    for i in range(N_RUNS):
        seed = BASE_SEED + i
        print(f"\n======== RUN {i+1} / {N_RUNS} ========")

        run_results = run_single_ablation(X, y, seed)
        all_runs.append(run_results)

        save_csv(
            f"{OUT_DIR}/run_{i+1}.csv",
            run_results
        )

    final = aggregate_results(all_runs)

    save_csv(
        f"{OUT_DIR}/summary_mean_std.csv",
        final
    )

    print("\n✅ Done. Results saved to:", OUT_DIR)