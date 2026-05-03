import os
import sys
import torch
import umap
import pacmap
import trimap
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DRSNE.drsne import drsne
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


import sys
sys.path.append("densvis/densne")

import densne

def embed_densne_2d(
        X,
        dim,
        seed,
        perplexity,
        dens_lambda,
        theta=0.5,
        dens_frac=0.5,
        initial_dims=None,
        use_pca=False,
        max_iter=800,
):
    if dim != 2:
        raise ValueError("DenSNE here is configured for 2D output only.")

    # -----------------------------------
    # 🔑 CRITICAL FIX: random initialization
    # -----------------------------------
    rng = np.random.RandomState(seed)

    init_emb = 1e-4 * rng.normal(size=(X.shape[0], 2)).astype(np.float64)

    # -----------------------------------
    # RUN DenSNE
    # -----------------------------------
    Z = densne.run_densne(
        np.asarray(X, dtype=np.float64),
        no_dims=dim,
        perplexity=float(perplexity),
        theta=float(theta),
        randseed=int(seed),          # may or may not be used internally
        verbose=False,
        initial_dims=initial_dims,
        use_pca=False,               # keep consistent across methods
        max_iter=int(max_iter),
        dens_frac=float(dens_frac),
        dens_lambda=float(dens_lambda),
        final_dens=False,
        initial_emb=init_emb
    )

    return Z

from sklearn.linear_model import LogisticRegression

def linear_scores(Z, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z, y)
    return clf.predict_proba(Z)[:, 1]

# =========================================================
# CONFIG
# =========================================================

LAMBDA_GRID = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1
]


# =========================================================
# DATA
# =========================================================




# =========================================================
# DR-SNE HELPERS
# =========================================================

def compute_knn_density(X, k=50):
    n = len(X)
    k_eff = min(k + 1, n)

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X)
    distances, indices = nbrs.kneighbors(X)

    if k_eff > 1:
        d = distances[:, 1:]
        volume = d.sum(axis=1) + 1e-8
        density = (k_eff - 1) / volume
    else:
        volume = distances.sum(axis=1) + 1e-8
        density = 1.0 / volume

    density /= (density.mean() + 1e-8)
    return density.astype(np.float32), indices.astype(np.int64)


def compute_P(X, perplexity=30.0, tol=1e-5):
    n = X.shape[0]
    D = pairwise_distances(X, squared=True)

    P = np.zeros((n, n), dtype=np.float32)
    log_perp = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        betamin, betamax = -np.inf, np.inf

        Di = np.delete(D[i], i)

        for _ in range(50):
            Pi = np.exp(-Di * beta)
            Pi /= (Pi.sum() + 1e-8)

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
# ANOMALY SCORES
# =========================================================

def knn_anomaly_score(Z, k=50):
    k_eff = min(k + 1, len(Z))
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Z)
    dists, _ = nbrs.kneighbors(Z)

    if k_eff > 1:
        dists = dists[:, 1:]

    return np.sum(dists, axis=1)


def lof_scores(Z, k=50):
    n = len(Z)
    k_eff = min(20, n - 1)

    lof = LocalOutlierFactor(n_neighbors=k_eff)
    lof.fit(Z)
    return -lof.negative_outlier_factor_


def iforest_scores(Z):
    model = IsolationForest(
        n_estimators=500,
        random_state=777,
        contamination="auto"
    )
    model.fit(Z)
    return -model.score_samples(Z)


def centroid_distance(Z):
    c = Z.mean(axis=0)
    return np.linalg.norm(Z - c, axis=1)


# =========================================================
# EMBEDDINGS
# =========================================================



def embed_tsne_2d(X, dim, seed, perplexity, lr):
    return TSNE(
        n_components=dim,
        perplexity=perplexity,
        learning_rate=lr,
        random_state=seed,
        init="random",
        method="barnes_hut" if dim <= 3 else "exact"
    ).fit_transform(X)

def embed_umap_2d(X, dim, seed, n_neighbors, min_dist):
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=dim,
        random_state=seed
    ).fit_transform(X)
def embed_densmap_2d(X, dim, seed, dens_lambda, dens_frac):
    result = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        densmap=True,
        dens_lambda=dens_lambda,
        dens_frac=dens_frac,
        n_components=dim,
        random_state=seed,
        init="random"
    ).fit_transform(X)

    return result[0] if isinstance(result, tuple) else result



def embed_pacmap_2d(X, dim, seed, n_neighbors, fp_ratio):
    return pacmap.PaCMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        FP_ratio=fp_ratio,
        MN_ratio=0.5,
        random_state=seed
    ).fit_transform(X)

def embed_trimap_2d(X, dim, seed, n_inliers, n_outliers):
    np.random.seed(seed)
    return trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=5
    ).fit_transform(X)


def embed_drsne_2d(X, dim, seed, lambda_density, k_density):
    if dim != 2:
        raise ValueError("DR-SNE currently supports dim=2")

    P = compute_P(X, perplexity=30.0)
    rho_high, knn_indices = compute_knn_density(X, k=k_density)
    Z, history = drsne(
        X=X,
        P=P,
        knn_indices=knn_indices,
        rho_high=rho_high,
        n_iter=1200,
        warmup=50,
        lr=2.0,
        lambda_density=lambda_density,
        seed=seed,
        verbose=False
    )

    return Z

def embed_drsne(X, dim, seed, lambda_density=0.0):
    if dim != 2:
        raise ValueError("DR-SNE currently supports dim=2")

    return drsne(
        X,
        n_components=2,
        lambda_density=lambda_density,
        seed=seed,
        verbose=False
    )

def evaluate_2d_sweep(
        method,
        X,
        y,
        dim,
        seed,
        param1_grid,
        param2_grid,
        embed_fn,
        param1_name,
        param2_name
):
    results = []

    for p1 in param1_grid:
        for p2 in param2_grid:
            print(f"\n--- {method} | {param1_name}={p1}, {param2_name}={p2} ---")

            Z = embed_fn(X, dim, seed, p1, p2)

            scores_knn = knn_anomaly_score(Z)
            scores_lof = lof_scores(Z)
            scores_if = iforest_scores(Z)
            scores_centroid = centroid_distance(Z)
            scores_linear = linear_scores(Z, y)

            res = {
                "method": method,
                param1_name: p1,
                param2_name: p2,
                "seed": seed,

                "AUROC_kNN": roc_auc_score(y, scores_knn),
                "AUROC_LOF": roc_auc_score(y, scores_lof),
                "AUROC_IF": roc_auc_score(y, scores_if),
                "AUROC_centroid": roc_auc_score(y, scores_centroid),
                "AUROC_linear": roc_auc_score(y, scores_linear),

                "AUPRC_kNN": average_precision_score(y, scores_knn),
                "AUPRC_LOF": average_precision_score(y, scores_lof),
                "AUPRC_IF": average_precision_score(y, scores_if),
                "AUPRC_centroid": average_precision_score(y, scores_centroid),
                "AUPRC_linear": average_precision_score(y, scores_linear),
            }

            print(
                f"{method} | {param1_name}={p1}, {param2_name}={p2} | seed={seed} | "
                f"kNN={res['AUPRC_kNN']:.3f} | "
                f"LOF={res['AUPRC_LOF']:.3f} | "
                f"IF={res['AUPRC_IF']:.3f} | "
                f"CENT={res['AUPRC_centroid']:.3f} | "
                f"LIN={res['AUPRC_linear']:.3f}"
            )
            results.append(res)

    return results

# =========================================================
# EVALUATION
# =========================================================



def evaluate_drsne_lambda_sweep(X_all, y_all, dataset_name, dim, seed):
    results = []

    for lam in LAMBDA_GRID:
        print(f"\n--- DR-SNE λ={lam} | seed={seed} ---")

        Z = embed_drsne(X_all, dim, seed, lambda_density=lam)

        scores_knn = knn_anomaly_score(Z)
        scores_lof = lof_scores(Z)
        scores_if = iforest_scores(Z)
        scores_centroid = centroid_distance(Z)
        scores_linear = linear_scores(Z, y_all)

        res = {
            "dataset": dataset_name,
            "method": "DR-SNE",
            "lambda": lam,
            "seed": seed,

            "AUROC_kNN": roc_auc_score(y_all, scores_knn),
            "AUROC_LOF": roc_auc_score(y_all, scores_lof),
            "AUROC_IF": roc_auc_score(y_all, scores_if),
            "AUROC_centroid": roc_auc_score(y_all, scores_centroid),
            "AUROC_linear": roc_auc_score(y_all, scores_linear),

            "AUPRC_kNN": average_precision_score(y_all, scores_knn),
            "AUPRC_LOF": average_precision_score(y_all, scores_lof),
            "AUPRC_IF": average_precision_score(y_all, scores_if),
            "AUPRC_centroid": average_precision_score(y_all, scores_centroid),
            "AUPRC_linear": average_precision_score(y_all, scores_linear),
        }

        print(
            f"seed={seed} | "
            f"kNN={res['AUPRC_kNN']:.3f} | "
            f"LOF={res['AUPRC_LOF']:.3f} | "
            f"IF={res['AUPRC_IF']:.3f} | "
            f"CENT={res['AUPRC_centroid']:.3f} | "
            f"LIN={res['AUPRC_linear']:.3f}"
        )

        results.append(res)

    return results

def embed_pacmap(X, dim, seed, n_neighbors=10):
    return pacmap.PaCMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed
    ).fit_transform(X)

def embed_trimap(X, dim, seed, n_inliers=10):
    np.random.seed(seed)  # ✅ control randomness here

    return trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=5,
        n_random=5
    ).fit_transform(X)
# =========================================================
# RUN
# =========================================================

def run_experiment_2d(X, y, dataset_name, dim=2, seeds=[0]):
    import pandas as pd

    all_results = []


    print("X shape:", X.shape)
    print("Anomaly ratio:", y.mean())

    dataset_path = f"results/{dataset_name}_2d.csv"

    # =====================================================
    # CORE EVALUATION (shared)
    # =====================================================
    def evaluate(Z, method, seed, p1_name=None, p1_val=None, p2_name=None, p2_val=None):
        scores_knn = knn_anomaly_score(Z)
        scores_lof = lof_scores(Z)
        scores_if = iforest_scores(Z)
        scores_centroid = centroid_distance(Z)
        scores_linear = linear_scores(Z, y)

        return {
            "dataset": dataset_name,
            "method": method,
            "seed": seed,

            "param1_name": p1_name,
            "param1_value": p1_val,
            "param2_name": p2_name,
            "param2_value": p2_val,

            "AUROC_kNN": roc_auc_score(y, scores_knn),
            "AUROC_LOF": roc_auc_score(y, scores_lof),
            "AUROC_IF": roc_auc_score(y, scores_if),
            "AUROC_centroid": roc_auc_score(y, scores_centroid),
            "AUROC_linear": roc_auc_score(y, scores_linear),

            "AUPRC_kNN": average_precision_score(y, scores_knn),
            "AUPRC_LOF": average_precision_score(y, scores_lof),
            "AUPRC_IF": average_precision_score(y, scores_if),
            "AUPRC_centroid": average_precision_score(y, scores_centroid),
            "AUPRC_linear": average_precision_score(y, scores_linear),
        }

    # =====================================================
    # BASELINE
    # =====================================================
    print("\n=== BASELINE (HIGH-D) ===")
    for seed in seeds:
        res = evaluate(X, "HIGH-D", seed)
        print(
            f"HIGH-D | seed={seed} | "
            f"kNN={res['AUPRC_kNN']:.3f} | "
            f"LOF={res['AUPRC_LOF']:.3f} | "
            f"IF={res['AUPRC_IF']:.3f} | "
            f"CENT={res['AUPRC_centroid']:.3f} | "
            f"LIN={res['AUPRC_linear']:.3f}"
        )
        all_results.append(res)

    # =====================================================
    # GENERIC 2D SWEEP
    # =====================================================
    def run_grid(method, embed_fn, grid1, grid2, name1, name2):
        for seed in seeds:
            for p1 in grid1:
                for p2 in grid2:
                    Z = embed_fn(X, dim, seed, p1, p2)
                    res = evaluate(
                        Z, method, seed,
                        p1_name=name1, p1_val=p1,
                        p2_name=name2, p2_val=p2
                    )

                    print(
                        f"{method} | {name1}={p1}, {name2}={p2} | seed={seed} | "
                        f"kNN={res['AUPRC_kNN']:.3f} | "
                        f"LOF={res['AUPRC_LOF']:.3f} | "
                        f"IF={res['AUPRC_IF']:.3f} | "
                        f"CENT={res['AUPRC_centroid']:.3f} | "
                        f"LIN={res['AUPRC_linear']:.3f}"
                    )

                    all_results.append(res)

    run_grid("DR-SNE", embed_drsne_2d, DRSNE_LAMBDA, DRSNE_K, "lambda", "k_density")
    """
    run_grid("DenSNE", embed_densne_2d, DENSNE_PERP, DENSNE_LAMBDA, "perplexity", "dens_lambda")

    run_grid("t-SNE", embed_tsne_2d, TSNE_PERP, TSNE_LR, "perplexity", "learning_rate")

    run_grid("UMAP", embed_umap_2d, UMAP_NN, UMAP_MIN_DIST, "n_neighbors", "min_dist")

    run_grid("DensMAP", embed_densmap_2d, DENSMAP_LAMBDA, DENSMAP_FRAC, "dens_lambda", "dens_frac")

    run_grid("PaCMAP", embed_pacmap_2d, PACMAP_NN, PACMAP_FP, "n_neighbors", "FP_ratio")
    """


    # =====================================================
    # SAVE
    # =====================================================
    df = pd.DataFrame(all_results)
    df.to_csv(dataset_path, index=False)

    print(f"\nSaved results → {dataset_path}")

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # =========================================================
    # DATASETS
    # =========================================================
    DATASETS = {
        "fashion": load_fashion_anomaly,
        "synthetic": load_spiral,
        "thyroid": load_tumor,
        "pbmc": load_pbmc,
        "shuttle": load_shuttle,
        "fashion_dino": load_fashion_dino,
        "cifar": load_resnet_cifar10,

    }

    # =========================================================
    # 2D PARAM GRIDS (CONSISTENT WITH YOUR ORIGINAL)
    # =========================================================

    DENSNE_PERP = [5, 10, 15, 30, 50, 75, 100]
    DENSNE_LAMBDA = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    TSNE_PERP = [1, 2, 3, 5, 10, 15, 30, 50, 75, 100]
    TSNE_LR = [50, 200, 500]

    UMAP_NN = [5, 8, 10, 15, 20, 30, 40, 60, 80, 120]
    UMAP_MIN_DIST = [0.0, 0.1, 0.5]

    DENSMAP_LAMBDA = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    DENSMAP_FRAC = [0.1, 0.3, 0.5]

    # PaCMAP
    PACMAP_NN = [5, 8, 10, 15, 20, 30, 40, 60, 80, 120]
    PACMAP_FP = [1.0, 2.0, 5.0]

    # TriMap
    TRIMAP_IN = [5, 8, 10, 15, 20, 30, 40, 60, 80, 120]
    TRIMAP_OUT = [2, 5, 10]

    # DR-SNE (FIXED naming)
    """DRSNE_LAMBDA = [
        0.0, 5e-5, 1e-4, 2e-4, 5e-4,
        1e-3, 2e-3, 5e-3,
        1e-2, 2e-2, 5e-2, 1e-1, 0.5, 1.0
    ]"""
    DRSNE_LAMBDA =[
        0.0, 5e-5, 1e-4, 2e-4, 5e-4,
        1e-3, 2e-3, 5e-3, 1e-2, 1e-1, 1.0
    ]

    #DRSNE_K = [10, 20, 40, 80, 300]
    DRSNE_K = [40, 80, 300]
    seeds = [0,1,2,3,4]

    os.makedirs("results", exist_ok=True)
    # =========================================================
    # DATA LOADER
    # =========================================================
    def get_data(name):
        X, y = DATASETS[name]()
        X = np.asarray(X)
        y = np.asarray(y)

        assert len(X) == len(y), f"{name}: X/y mismatch"
        assert y.ndim == 1, f"{name}: y must be 1D"

        return X, y

    # =========================================================
    # RUN
    # =========================================================
    for dataset_name in DATASETS:
        print("\n\n==============================")
        print(f"DATASET: {dataset_name}")
        print("==============================")

        X, y = get_data(dataset_name)

        # IMPORTANT
        X = StandardScaler().fit_transform(X)

        df = run_experiment_2d(
            X,
            y,
            dataset_name=dataset_name,
            dim=2,
            seeds=seeds
        )

        # =====================================================
        # SAFE RESULT PRINTING
        # =====================================================
        print("\nTop configs (by AUPRC_IF):")

        cols = [
            "method",
            "param1_name",
            "param1_value",
            "param2_name",
            "param2_value",
            "AUPRC_IF"
        ]

        # only keep existing columns (robust)
        cols = [c for c in cols if c in df.columns]

        print(
            df.sort_values("AUPRC_IF", ascending=False)
            .head(5)[cols]
        )