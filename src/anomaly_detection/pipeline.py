import numpy as np
import os
import csv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import sys
import scanpy as sc
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comparison import (
    run_tsne,
    run_umap,
    run_densmap,
    run_pacmap,
    run_density_tsne,
    compute_knn_density,
    compute_P
)

# =========================================================
# OUTPUT DIR
# =========================================================
OUTPUT_DIR = "output/anomaly"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except:
        pass


# =========================================================
# ANOMALY DEFINITION
# =========================================================
def anomaly_pbmc(y, threshold=0.05):
    unique, counts = np.unique(y, return_counts=True)
    freq = counts / len(y)

    rare = unique[freq < threshold]
    labels = np.isin(y, rare).astype(int)

    print(f"Rare clusters: {rare}")
    print(f"Anomaly %: {labels.mean()*100:.2f}%")

    return labels

# =========================================================
# LOAD SHUTTLE
# =========================================================
def load_shuttle(n_samples=50000, seed=42):
    from sklearn.datasets import fetch_openml

    print("Loading Shuttle dataset...")

    X, y = fetch_openml("shuttle", version=1, return_X_y=True, as_frame=False)

    # convert labels to int
    y = y.astype(int)

    # subsample for speed
    if n_samples is not None and len(X) > n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # standardize
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    print(f"Shuttle shape: {X.shape}")
    return X, y

# =========================================================
# ANOMALY: SHUTTLE
# =========================================================
def anomaly_shuttle(y):
    # class 1 = normal, others = anomaly
    labels = (y != 1).astype(int)

    print(f"Anomaly %: {labels.mean()*100:.2f}%")
    return labels

# =========================================================
# SPLIT
# =========================================================
def split_data(anomaly_labels, test_size=0.3, seed=42):
    idx = np.arange(len(anomaly_labels))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=anomaly_labels,
        random_state=seed
    )

    return train_idx, test_idx


# =========================================================
# KNN PROJECTION
# =========================================================
def knn_project(X_train, Z_train, X_test, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    _, idx = nbrs.kneighbors(X_test)
    return Z_train[idx].mean(axis=1)


# =========================================================
# DENSITY SCORE
# =========================================================
def compute_density_scores_train_test(Z_train, Z_test, k=30):
    nbrs = NearestNeighbors(n_neighbors=k).fit(Z_train)

    dist_train, _ = nbrs.kneighbors(Z_train)
    vol_train = dist_train[:, 1:].sum(axis=1) + 1e-8
    rho_train = (k - 1) / vol_train

    dist_test, _ = nbrs.kneighbors(Z_test)
    vol_test = dist_test.sum(axis=1) + 1e-8
    rho_test = (k - 1) / vol_test

    mean_rho = rho_train.mean() + 1e-8
    scores_test = -np.log(rho_test / mean_rho + 1e-8)

    return scores_test


# =========================================================
# SAVE CSV
# =========================================================
def save_results_csv(results, dataset_name, seed):
    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_seed_{seed}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results → {path}")


# =========================================================
# SAVE SUMMARY CSV
# =========================================================
def save_summary_csv(all_results, dataset_name):
    if len(all_results) <= 1:
        return

    grouped = {}

    for results in all_results:
        for row in results:
            key = (row["method"], str(row["param"]), row["tweaked_parameter"])
            if key not in grouped:
                grouped[key] = {"auroc": [], "auprc": []}
            grouped[key]["auroc"].append(row["auroc"])
            grouped[key]["auprc"].append(row["auprc"])

    summary_rows = []
    for (method, param, tweaked_parameter), vals in grouped.items():
        summary_rows.append({
            "method": method,
            "param": param,
            "tweaked_parameter": tweaked_parameter,
            "auroc_mean": float(np.mean(vals["auroc"])),
            "auroc_std": float(np.std(vals["auroc"], ddof=1)) if len(vals["auroc"]) > 1 else 0.0,
            "auprc_mean": float(np.mean(vals["auprc"])),
            "auprc_std": float(np.std(vals["auprc"], ddof=1)) if len(vals["auprc"]) > 1 else 0.0,
            "n_seeds": len(vals["auroc"])
        })

    summary_rows.sort(key=lambda x: (x["method"], x["param"]))

    path = os.path.join(OUTPUT_DIR, f"{dataset_name}_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "param",
                "tweaked_parameter",
                "auroc_mean",
                "auroc_std",
                "auprc_mean",
                "auprc_std",
                "n_seeds"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary → {path}")


# =========================================================
# SAVE EMBEDDINGS
# =========================================================
def save_embedding(Z_train, Z_test, scores_test, labels_test, method, param, dataset, seed):
    fname = f"{dataset}_seed_{seed}_{method}_{param}.npz"
    path = os.path.join(OUTPUT_DIR, fname)

    np.savez_compressed(
        path,
        Z_train=Z_train,
        Z_test=Z_test,
        scores_test=scores_test,
        labels_test=labels_test,
        method=method,
        param=param
    )


# =========================================================
# LOAD DATA
# =========================================================
def load_pbmc(seed=42):
    sc.settings.seed = seed

    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, random_state=seed)
    sc.tl.louvain(adata, resolution=1.0, random_state=seed)

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values
    return X, y


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pbmc_anomaly(seeds):

    tsne_grid = [5, 10, 20, 30, 50, 75, 100]
    umap_grid = [5, 10, 15, 30, 50, 100]
    pacmap_grid = [5, 10, 15, 30, 50]
    densmap_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    lambda_grid = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

    all_results = []

    for seed in seeds:

        set_global_seed(seed)

        print("\n==============================")
        print(f"PBMC ANOMALY | SEED: {seed}")
        print("==============================")

        X, y = load_pbmc(seed=seed)
        anomaly_labels = anomaly_pbmc(y)

        train_idx, test_idx = split_data(anomaly_labels, seed=seed)

        X_train, X_test = X[train_idx], X[test_idx]
        y_test = anomaly_labels[test_idx]

        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        rho_high, knn_indices = compute_knn_density(X_train)
        P = compute_P(X_train)

        results = []

        # =====================================================
        # 🔵 kNN BASELINE
        # =====================================================
        nbrs = NearestNeighbors(n_neighbors=30).fit(X_train)
        dist_test, _ = nbrs.kneighbors(X_test)
        scores_knn = dist_test.mean(axis=1)

        auroc = roc_auc_score(y_test, scores_knn)
        auprc = average_precision_score(y_test, scores_knn)

        results.append({
            "method": "kNN",
            "param": 30,
            "tweaked_parameter": "n_neighbors",
            "auroc": auroc,
            "auprc": auprc
        })

        # =====================================================
        # 🟣 Isolation Forest
        # =====================================================
        iso = IsolationForest(random_state=seed)
        iso.fit(X_train)
        scores_iso = -iso.score_samples(X_test)

        auroc = roc_auc_score(y_test, scores_iso)
        auprc = average_precision_score(y_test, scores_iso)

        results.append({
            "method": "IsolationForest",
            "param": "-",
            "tweaked_parameter": "default",
            "auroc": auroc,
            "auprc": auprc
        })

        # =====================================================
        # t-SNE
        # =====================================================
        for p in tsne_grid:
            Z_train = run_tsne(X_train, seed=seed, perplexity=p)
            Z_test = knn_project(X_train, Z_train, X_test)

            scores_test = compute_density_scores_train_test(Z_train, Z_test)

            auroc = roc_auc_score(y_test, scores_test)
            auprc = average_precision_score(y_test, scores_test)

            results.append({
                "method": "t-SNE",
                "param": p,
                "tweaked_parameter": "perplexity",
                "auroc": auroc,
                "auprc": auprc
            })

        # =====================================================
        # UMAP
        # =====================================================
        for k in umap_grid:
            Z_train = run_umap(X_train, seed=seed, n_neighbors=k)
            Z_test = knn_project(X_train, Z_train, X_test)

            scores_test = compute_density_scores_train_test(Z_train, Z_test)

            auroc = roc_auc_score(y_test, scores_test)
            auprc = average_precision_score(y_test, scores_test)

            results.append({
                "method": "UMAP",
                "param": k,
                "tweaked_parameter": "n_neighbors",
                "auroc": auroc,
                "auprc": auprc
            })

        # =====================================================
        # PaCMAP
        # =====================================================
        for k in pacmap_grid:
            Z_train = run_pacmap(X_train, seed=seed, n_neighbors=k)
            Z_test = knn_project(X_train, Z_train, X_test)

            scores_test = compute_density_scores_train_test(Z_train, Z_test)

            auroc = roc_auc_score(y_test, scores_test)
            auprc = average_precision_score(y_test, scores_test)

            results.append({
                "method": "PaCMAP",
                "param": k,
                "tweaked_parameter": "n_neighbors",
                "auroc": auroc,
                "auprc": auprc
            })

        # =====================================================
        # DensMAP
        # =====================================================
        for lam in densmap_grid:
            Z_train = run_densmap(X_train, seed=seed, dens_lambda=lam)
            Z_test = knn_project(X_train, Z_train, X_test)

            scores_test = compute_density_scores_train_test(Z_train, Z_test)

            auroc = roc_auc_score(y_test, scores_test)
            auprc = average_precision_score(y_test, scores_test)

            results.append({
                "method": "DensMAP",
                "param": lam,
                "tweaked_parameter": "dens_lambda",
                "auroc": auroc,
                "auprc": auprc
            })

        # =====================================================
        # Density t-SNE
        # =====================================================
        for lam in lambda_grid:
            Z_train, _ = run_density_tsne(
                X_train,
                P,
                knn_indices,
                rho_high,
                lambda_density=lam,
                seed=seed
            )

            Z_test = knn_project(X_train, Z_train, X_test)

            scores_test = compute_density_scores_train_test(Z_train, Z_test)

            auroc = roc_auc_score(y_test, scores_test)
            auprc = average_precision_score(y_test, scores_test)

            results.append({
                "method": "Density t-SNE",
                "param": lam,
                "tweaked_parameter": "lambda_density",
                "auroc": auroc,
                "auprc": auprc
            })

        save_results_csv(results, "pbmc", seed)
        all_results.append(results)

        print("\n==============================\n")

    save_summary_csv(all_results, "pbmc")

def run_shuttle_anomaly(seeds):

    tsne_grid = [5, 10, 20, 30, 50, 75, 100]
    umap_grid = [5, 10, 15, 30, 50, 100]
    pacmap_grid = [5, 10, 15, 30, 50]
    densmap_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    lambda_grid = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

    all_results = []

    for seed in seeds:

        set_global_seed(seed)

        print("\n==============================")
        print(f"SHUTTLE ANOMALY | SEED: {seed}")
        print("==============================")

        X, y = load_shuttle(seed=seed)
        anomaly_labels = anomaly_shuttle(y)

        train_idx, test_idx = split_data(anomaly_labels, seed=seed)

        X_train, X_test = X[train_idx], X[test_idx]
        y_test = anomaly_labels[test_idx]

        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        rho_high, knn_indices = compute_knn_density(X_train)
        P = compute_P(X_train)

        results = []

        # =====================================================
        # kNN BASELINE
        # =====================================================
        nbrs = NearestNeighbors(n_neighbors=30).fit(X_train)
        dist_test, _ = nbrs.kneighbors(X_test)
        scores_knn = dist_test.mean(axis=1)

        results.append({
            "method": "kNN",
            "param": 30,
            "tweaked_parameter": "n_neighbors",
            "auroc": roc_auc_score(y_test, scores_knn),
            "auprc": average_precision_score(y_test, scores_knn)
        })

        # =====================================================
        # Isolation Forest
        # =====================================================
        iso = IsolationForest(random_state=seed)
        iso.fit(X_train)
        scores_iso = -iso.score_samples(X_test)

        results.append({
            "method": "IsolationForest",
            "param": "-",
            "tweaked_parameter": "default",
            "auroc": roc_auc_score(y_test, scores_iso),
            "auprc": average_precision_score(y_test, scores_iso)
        })

        # =====================================================
        # EMBEDDING METHODS (UNCHANGED)
        # =====================================================
        def eval_embedding(method_name, Z_train):
            Z_test = knn_project(X_train, Z_train, X_test)
            scores = compute_density_scores_train_test(Z_train, Z_test)

            return roc_auc_score(y_test, scores), average_precision_score(y_test, scores)

        # t-SNE
        for p in tsne_grid:
            Z_train = run_tsne(X_train, seed=seed, perplexity=p)
            auroc, auprc = eval_embedding("t-SNE", Z_train)
            results.append({"method": "t-SNE", "param": p, "tweaked_parameter": "perplexity", "auroc": auroc, "auprc": auprc})

        # UMAP
        for k in umap_grid:
            Z_train = run_umap(X_train, seed=seed, n_neighbors=k)
            auroc, auprc = eval_embedding("UMAP", Z_train)
            results.append({"method": "UMAP", "param": k, "tweaked_parameter": "n_neighbors", "auroc": auroc, "auprc": auprc})

        # PaCMAP
        for k in pacmap_grid:
            Z_train = run_pacmap(X_train, seed=seed, n_neighbors=k)
            auroc, auprc = eval_embedding("PaCMAP", Z_train)
            results.append({"method": "PaCMAP", "param": k, "tweaked_parameter": "n_neighbors", "auroc": auroc, "auprc": auprc})

        # DensMAP
        for lam in densmap_grid:
            Z_train = run_densmap(X_train, seed=seed, dens_lambda=lam)
            auroc, auprc = eval_embedding("DensMAP", Z_train)
            results.append({"method": "DensMAP", "param": lam, "tweaked_parameter": "dens_lambda", "auroc": auroc, "auprc": auprc})

        # Density t-SNE
        for lam in lambda_grid:
            Z_train, _ = run_density_tsne(
                X_train, P, knn_indices, rho_high,
                lambda_density=lam,
                seed=seed
            )
            auroc, auprc = eval_embedding("Density t-SNE", Z_train)
            results.append({"method": "Density t-SNE", "param": lam, "tweaked_parameter": "lambda_density", "auroc": auroc, "auprc": auprc})

        save_results_csv(results, "shuttle", seed)
        all_results.append(results)

    save_summary_csv(all_results, "shuttle")


if __name__ == "__main__":
    seeds = [23,232,2323]
    run_shuttle_anomaly(seeds)
    run_pbmc_anomaly(seeds)
