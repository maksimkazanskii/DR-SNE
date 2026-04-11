import os
import time
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.manifold import trustworthiness

# =========================================================
# GLOBAL DETERMINISM
# =========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
except:
    pass

# =========================================================
# IMPORTS (LOCAL)
# =========================================================
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from density_tsne import run_density_tsne
from comparison import (
    compute_knn_density,
    compute_P,
    continuity,
    density_correlation,
)

# =========================================================
# SETTINGS
# =========================================================
LAMBDA_FIXED = 0.01
NEIGHBOR_LIST = [10, 50, 250]
PCA_DIM = 50
OUT_DIR = "output/ablation_neighbors"


# =========================================================
# HELPERS
# =========================================================
def timed_run(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


# =========================================================
# DATA
# =========================================================
def load_pbmc(n_comps=50, n_neighbors=15, seed=42):
    sc.settings.seed = seed

    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_comps,
        random_state=seed
    )

    sc.tl.louvain(
        adata,
        resolution=1.0,
        random_state=seed
    )

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values

    print(f"Neighbors={n_neighbors} | clusters={len(np.unique(y))}")
    return X, y


# =========================================================
# PLOTTING
# =========================================================
def plot_neighbors_progression(embeddings, neighbor_list, y):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        1, len(embeddings),
        figsize=(4 * len(embeddings), 4),
        gridspec_kw={'wspace': 0.0}
    )

    if len(embeddings) == 1:
        axes = [axes]

    from matplotlib.colors import ListedColormap

    colors = [
        "#1f77b4", "#4fa3d1", "#a6cee3",
        "#d95f02", "#ff7f0e", "#fdd49e",
    ]
    cmap = ListedColormap(colors)

    for ax, Z, k in zip(axes, embeddings, neighbor_list):
        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y % len(colors),
            cmap=cmap,
            s=5,
            alpha=0.4
        )

        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("0.85")

    plt.subplots_adjust(wspace=0.0)

    filename = os.path.join(OUT_DIR, "rna_neighbors_progression.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved figure -> {filename}")


# =========================================================
# CORE
# =========================================================
def run_neighbors_ablation(seed=42):
    print("\n=== Neighbor Ablation (λ = 0.01) ===")

    os.makedirs(OUT_DIR, exist_ok=True)

    embeddings = []
    results = []

    for k in NEIGHBOR_LIST:
        print(f"\n--- Neighbors = {k} ---")

        X, y = load_pbmc(n_comps=PCA_DIM, n_neighbors=k, seed=seed)

        rho_high, knn_indices = compute_knn_density(X, k=k)
        P = compute_P(X)

        try:
            import torch
            if not isinstance(P, torch.Tensor):
                P = torch.tensor(P, dtype=torch.float32)
        except:
            pass

        (Z, _), runtime = timed_run(
            run_density_tsne,
            X=X,
            P=P,
            knn_indices=knn_indices,
            rho_high=rho_high,
            lambda_density=LAMBDA_FIXED,
            seed=seed,
        )

        embeddings.append(Z)

        # metrics
        tw = trustworthiness(X, Z, n_neighbors=10)
        cont = continuity(X, Z, n_neighbors=10)
        dens = density_correlation(Z, knn_indices, rho_high)

        print(
            f"k={k} | "
            f"TW={tw:.4f} | CONT={cont:.4f} | "
            f"DENS={dens:.4f} | TIME={runtime:.2f}"
        )

        results.append({
            "neighbors": k,
            "trustworthiness": tw,
            "continuity": cont,
            "density_corr": dens,
            "time_sec": runtime,
        })

    # =========================
    # SAVE CSV
    # =========================
    csv_path = os.path.join(OUT_DIR, "rna_neighbors_ablation.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV -> {csv_path}")

    # =========================
    # PLOT
    # =========================
    plot_neighbors_progression(embeddings, NEIGHBOR_LIST, y)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_neighbors_ablation(seed=SEED)