import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import umap
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from density_tsne import run_density_tsne
import torch
# 🔥 import your dataset loaders

import warnings

warnings.filterwarnings(
    "ignore",
    message="n_jobs value .* overridden to 1 by setting random_state"
)
from datasets import (
    load_thyroid_easy,
    load_synthetic_anomaly,
    load_shuttle,
)

import numpy as np

def aggregate_results(all_results):
    grouped = {}

    # -------------------------
    # GROUP RESULTS
    # -------------------------
    for res in all_results:
        key = (res["dataset"], res["method"], res["dim"])
        grouped.setdefault(key, []).append(res)

    final = []

    # -------------------------
    # AGGREGATE
    # -------------------------
    for (dataset, method, dim), runs in grouped.items():

        out = {
            "dataset": dataset,
            "method": method,
            "dim": dim,
            "density_tuned": {},
            "distance_tuned": {}
        }

        for mode in ["density_tuned", "distance_tuned"]:

            metrics = {
                "dist_auroc": [],
                "dist_auprc": [],
                "dens_auroc": [],
                "dens_auprc": [],
                "time": []
            }

            counts = {k: 0 for k in metrics}

            # 🔥 NEW: track repeats
            repeat_counts = []

            # -------------------------
            # COLLECT
            # -------------------------
            for r in runs:
                block = r.get(mode, None)
                if block is None:
                    continue

                # 🔥 collect repeats
                if "n_repeats" in block and block["n_repeats"] is not None:
                    repeat_counts.append(block["n_repeats"])

                for m in metrics:
                    val = block.get(m, None)

                    if val is None:
                        continue
                    if isinstance(val, float) and np.isnan(val):
                        continue

                    metrics[m].append(val)
                    counts[m] += 1

            # -------------------------
            # COMPUTE STATS
            # -------------------------
            agg = {}

            for m, values in metrics.items():
                if len(values) == 0:
                    agg[m + "_mean"] = None
                    agg[m + "_std"] = None
                else:
                    agg[m + "_mean"] = float(np.mean(values))
                    agg[m + "_std"] = float(np.std(values))

                agg[m + "_n"] = counts[m]

            # -------------------------
            # GLOBAL COUNTS (metrics)
            # -------------------------
            valid_counts = [c for c in counts.values() if c > 0]

            if len(valid_counts) == 0:
                agg["n_seeds_min"] = 0
                agg["n_seeds_max"] = 0
            else:
                agg["n_seeds_min"] = int(min(valid_counts))
                agg["n_seeds_max"] = int(max(valid_counts))

            # -------------------------
            # 🔥 NEW: REPEAT STATS
            # -------------------------
            if len(repeat_counts) == 0:
                agg["n_repeats_min"] = 0
                agg["n_repeats_max"] = 0
            else:
                agg["n_repeats_min"] = int(min(repeat_counts))
                agg["n_repeats_max"] = int(max(repeat_counts))

            out[mode] = agg

        final.append(out)

    return final

# =========================================================
# KNN MAPPING
# =========================================================
def knn_map(X_train, Z_train, X_query, k=30):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    dist, idx = nbrs.kneighbors(X_query)

    sigma = np.maximum(dist.mean(axis=1, keepdims=True), 1e-3)
    weights = np.exp(-dist**2 / (sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)

    Z_query = (weights[..., None] * Z_train[idx]).sum(axis=1)
    return Z_query


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
# ANOMALY SCORES
# =========================================================


def compute_knn_density(X, k=30):
    k_eff = min(k, len(X))
    if k_eff < 2:
        raise ValueError("Need at least 2 points for density estimation.")

    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="brute").fit(X)
    distances, indices = nbrs.kneighbors(X)

    volume = distances[:, 1:].sum(axis=1) + 1e-8
    density = (k_eff - 1) / volume
    density = density / (density.mean() + 1e-8)

    return density, indices
def anomaly_score_distance(Z_train, Z_query, k=30):
    nbrs = NearestNeighbors(n_neighbors=k).fit(Z_train)
    dist, _ = nbrs.kneighbors(Z_query)
    return dist.mean(axis=1)

def anomaly_score_density(Z_train, Z_query, k=30):
    nbrs = NearestNeighbors(n_neighbors=k).fit(Z_train)
    dist, _ = nbrs.kneighbors(Z_query)
    volume = dist.sum(axis=1)
    rho = k / (volume + 1e-8)
    return -np.log(rho + 1e-8)


# =========================================================
# EMBEDDINGS
# =========================================================
def embed_pca(X_train, dim):
    model = PCA(n_components=dim)
    Z_train = model.fit_transform(X_train)
    return Z_train, model


def embed_tsne(X_train, dim, seed):
    return TSNE(
        n_components=dim,
        perplexity=30,
        init="pca",              # 🔥 FIXED
        learning_rate="auto",    # 🔥 modern default (sklearn >=1.2)
        n_iter=1000,             # explicit for clarity
        random_state=seed,
        method="barnes_hut",     # explicit (default but safer)
        angle=0.5                # standard tradeoff
    ).fit_transform(X_train)


def embed_umap(X_train, dim, seed):
    return umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=dim,
        random_state=seed
    ).fit_transform(X_train)


def embed_density_tsne(X_train, dim, seed, lam):
    P = compute_P(X_train, perplexity=30)
    rho, knn_idx = compute_knn_density(X_train, k=30)

    Z_train, _ = run_density_tsne(
        X_train,
        P,
        knn_idx,
        rho,
        lambda_density=lam,
        seed=seed,
        verbose=False
    )

    return Z_train


# =========================================================
# TUNING (USES VAL PROPERLY)
# =========================================================
def tune_high_dim(X_train, X_val, y_val, k_list=[10, 20, 30, 50], score_type="density"):
    best_score = -np.inf
    best_k = None

    for k in k_list:
        try:
            if score_type == "density":
                scores = anomaly_score_density(X_train, X_val, k)
            else:
                scores = anomaly_score_distance(X_train, X_val, k)

            score = roc_auc_score(y_val, scores)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        print("[WARN] No valid k found for HIGH-D, fallback to k=30")
        best_k = 30

    return best_k

def tune_method(
        name,
        X_train,
        X_val,
        y_val,
        dim,
        seed,
        score_type="density",
        k_map_fixed=30,
        n_repeats=3
):
    best_score = -np.inf
    best_params = None

    k_score_list = [10, 20, 30, 40, 50]

    if name == "Density t-SNE":
        lambda_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        dens_frac_list = [None]

    elif name == "DensMAP":
        lambda_list = [0.1, 0.5, 1.0, 2.0, 5.0]
        dens_frac_list = [0.1, 0.3, 0.5]

    else:
        lambda_list = [None]
        dens_frac_list = [None]

    for lam in lambda_list:
        for dens_frac in dens_frac_list:

            # store results per k_score
            scores_per_k = {k: [] for k in k_score_list}

            # -------------------------
            # REPEAT EMBEDDINGS
            # -------------------------
            for repeat in range(n_repeats):

                try:
                    seed_i = seed + repeat

                    # -------------------------
                    # EMBEDDING (ONCE per repeat)
                    # -------------------------
                    if name == "PCA":
                        Z_train, model = embed_pca(X_train, dim)
                        Z_val = model.transform(X_val)

                    elif name == "t-SNE":
                        Z_train = embed_tsne(X_train, dim, seed_i)
                        Z_val = knn_map(X_train, Z_train, X_val, k_map_fixed)

                    elif name == "UMAP":
                        Z_train = embed_umap(X_train, dim, seed_i)
                        Z_val = knn_map(X_train, Z_train, X_val, k_map_fixed)

                    elif name == "Density t-SNE":
                        Z_train = embed_density_tsne(X_train, dim, seed_i, lam)
                        Z_val = knn_map(X_train, Z_train, X_val, k_map_fixed)

                    elif name == "DensMAP":
                        Z_train = embed_densmap(X_train, dim, seed_i, lam, dens_frac)
                        Z_val = knn_map(X_train, Z_train, X_val, k_map_fixed)

                    else:
                        raise ValueError(f"Unknown method: {name}")

                    # -------------------------
                    # EVALUATE ALL k_score
                    # -------------------------
                    for k_score in k_score_list:
                        if score_type == "density":
                            scores = anomaly_score_density(Z_train, Z_val, k_score)
                        else:
                            scores = anomaly_score_distance(Z_train, Z_val, k_score)

                        auc = roc_auc_score(y_val, scores)
                        scores_per_k[k_score].append(auc)

                except Exception:
                    continue

            # -------------------------
            # AGGREGATE PER k_score
            # -------------------------
            for k_score, values in scores_per_k.items():
                if len(values) == 0:
                    continue

                mean_score = np.mean(values)

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = (k_map_fixed, k_score, lam, dens_frac)

    # -------------------------
    # FALLBACK
    # -------------------------
    if best_params is None:
        print(f"[WARN] fallback params for {name}")
        best_params = (k_map_fixed, 30, None, None)

    return best_params

def embed_densmap(X_train, dim, seed, dens_lambda, dens_frac):
    """
    Density-preserving UMAP (DensMAP) with tunable parameters
    """
    return umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=dim,
        random_state=seed,
        densmap=True,
        dens_lambda=dens_lambda,
        dens_frac=dens_frac
    ).fit_transform(X_train)

# =========================================================
# EVALUATION
# =========================================================
def evaluate_method(
        name,
        train,
        val,
        test,
        dim,
        seed,
        k_map_fixed=30,
        n_repeats=3   # 🔥 same idea as tuning
):
    X_train, _ = train
    X_val, y_val = val
    X_test, y_test = test

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = {}

    for score_type in ["density", "distance"]:

        # -------------------------
        # TUNE (already repeat-aware)
        # -------------------------
        k_map, k_score, lam, dens_frac = tune_method(
            name=name,
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            dim=dim,
            seed=seed,
            score_type=score_type,
            k_map_fixed=k_map_fixed,
        )

        # -------------------------
        # REPEAT FINAL EVALUATION
        # -------------------------
        dist_auroc_list = []
        dist_auprc_list = []
        dens_auroc_list = []
        dens_auprc_list = []
        time_list = []

        for repeat in range(n_repeats):
            try:
                seed_i = seed + repeat

                start = time.time()

                # -------------------------
                # TRAIN FINAL
                # -------------------------
                if name == "PCA":
                    Z_train, model = embed_pca(X_train, dim)
                    Z_test = model.transform(X_test)

                elif name == "t-SNE":
                    Z_train = embed_tsne(X_train, dim, seed_i)
                    Z_test = knn_map(X_train, Z_train, X_test, k_map)

                elif name == "UMAP":
                    Z_train = embed_umap(X_train, dim, seed_i)
                    Z_test = knn_map(X_train, Z_train, X_test, k_map)

                elif name == "Density t-SNE":
                    Z_train = embed_density_tsne(X_train, dim, seed_i, lam)
                    Z_test = knn_map(X_train, Z_train, X_test, k_map)

                elif name == "DensMAP":
                    Z_train = embed_densmap(X_train, dim, seed_i, lam, dens_frac)
                    Z_test = knn_map(X_train, Z_train, X_test, k_map)

                else:
                    raise ValueError(f"Unknown method: {name}")

                # -------------------------
                # TEST SCORES
                # -------------------------
                scores_dist = anomaly_score_distance(Z_train, Z_test, k_score)
                scores_dens = anomaly_score_density(Z_train, Z_test, k_score)

                elapsed = time.time() - start

                dist_auroc_list.append(roc_auc_score(y_test, scores_dist))
                dist_auprc_list.append(average_precision_score(y_test, scores_dist))
                dens_auroc_list.append(roc_auc_score(y_test, scores_dens))
                dens_auprc_list.append(average_precision_score(y_test, scores_dens))
                time_list.append(elapsed)

            except Exception:
                continue

        # -------------------------
        # AGGREGATE REPEATS
        # -------------------------
        def safe_mean(x):
            return float(np.mean(x)) if len(x) > 0 else None

        results[score_type] = {
            "k_map": k_map,
            "k_score": k_score,
            "lambda": lam,
            "dens_frac": dens_frac,
            "dist_auroc": safe_mean(dist_auroc_list),
            "dist_auprc": safe_mean(dist_auprc_list),
            "dens_auroc": safe_mean(dens_auroc_list),
            "dens_auprc": safe_mean(dens_auprc_list),
            "time": safe_mean(time_list),
            "n_repeats": len(dist_auroc_list)
        }

    return {
        "method": name,
        "dim": dim,
        "density_tuned": results["density"],
        "distance_tuned": results["distance"],
    }


# =========================================================
# HIGH-D BASELINE
# =========================================================
def evaluate_high_dim(train, val, test, n_repeats=3):
    X_train, _ = train
    X_val, y_val = val
    X_test, y_test = test

    # -------------------------
    # SCALE
    # -------------------------
    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    results = {}

    for score_type in ["density", "distance"]:

        # -------------------------
        # TUNE k (deterministic)
        # -------------------------
        k = tune_high_dim(
            X_train, X_val, y_val,
            score_type=score_type
        )

        # -------------------------
        # REPEAT EVALUATION (symmetry)
        # -------------------------
        dist_auroc_list = []
        dist_auprc_list = []
        dens_auroc_list = []
        dens_auprc_list = []
        time_list = []

        for _ in range(n_repeats):
            try:
                start = time.time()

                scores_dist = anomaly_score_distance(X_train, X_test, k)
                scores_dens = anomaly_score_density(X_train, X_test, k)

                elapsed = time.time() - start

                dist_auroc_list.append(roc_auc_score(y_test, scores_dist))
                dist_auprc_list.append(average_precision_score(y_test, scores_dist))
                dens_auroc_list.append(roc_auc_score(y_test, scores_dens))
                dens_auprc_list.append(average_precision_score(y_test, scores_dens))
                time_list.append(elapsed)

            except Exception:
                continue

        def safe_mean(x):
            return float(np.mean(x)) if len(x) > 0 else None

        results[score_type] = {
            "k": k,
            "dist_auroc": safe_mean(dist_auroc_list),
            "dist_auprc": safe_mean(dist_auprc_list),
            "dens_auroc": safe_mean(dens_auroc_list),
            "dens_auprc": safe_mean(dens_auprc_list),
            "time": safe_mean(time_list),
            "n_repeats": len(dist_auroc_list)  # 🔥 now consistent
        }

    return {
        "method": "HIGH-D",
        "dim": X_train.shape[1],
        "density_tuned": results["density"],
        "distance_tuned": results["distance"],
    }
# =========================================================
# RUN DATASET
# =========================================================
def run_dataset(name, loader, seed=42):
    print(f"\n==============================")
    print(f"Dataset: {name}")
    print(f"==============================")

    train, val, test = loader(seed=seed)

    methods = ["PCA", "t-SNE", "UMAP", "DensMAP", "Density t-SNE"]
    results = []

    # -------------------------
    # HIGH-D BASELINE
    # -------------------------
    print("\nRunning HIGH-D baseline...")
    hd_res = evaluate_high_dim(train, val, test)
    hd_res["dataset"] = name
    print(hd_res)
    results.append(hd_res)

    # -------------------------
    # EMBEDDING METHODS
    # -------------------------
    for dim in [2, 10]:
        print(f"\n--- dim = {dim} ---")

        for method in methods:
            print(f"Running {method}...")

            res = evaluate_method(
                method,
                train,
                val,
                test,
                dim,
                seed
            )

            res["dataset"] = name
            print(res)
            results.append(res)

    return results


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    all_results = []

    for seed in [0, 1, 2]:
        all_results += run_dataset("thyroid", load_thyroid_easy, seed)
        all_results += run_dataset("synthetic", load_synthetic_anomaly, seed)
        all_results += run_dataset("shuttle", load_shuttle, seed)

    # 🔥 AGGREGATE HERE
    aggregated = aggregate_results(all_results)

    print("\n==============================")
    print("FINAL AGGREGATED RESULTS")
    print("==============================")

    for res in aggregated:
        print(res)