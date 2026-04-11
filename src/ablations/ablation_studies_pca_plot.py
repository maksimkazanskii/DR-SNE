import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.manifold import trustworthiness

import os
import random
import numpy as np

SEED = 42

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# OS-level (important for hashing, some libs)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Threading (prevents nondeterministic BLAS behavior)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Torch (if used)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Strict determinism (may slow down / error if unsupported ops)
    torch.use_deterministic_algorithms(True)
except:
    pass
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
PCA_DIMS = [20, 50, 100]
SEED = 42
OUT_DIR = "output/ablation_pca"



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
def load_pbmc(n_comps=50, seed=42):
    sc.settings.seed = seed  # IMPORTANT

    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # deterministic gene selection order
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)

    # PCA (deterministic solver)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')

    # neighbors (force deterministic)
    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        n_pcs=n_comps,
        random_state=seed
    )

    # Louvain (NOT deterministic unless seed set)
    sc.tl.louvain(
        adata,
        resolution=1.0,
        random_state=seed
    )

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values

    print(f"PCA={n_comps} | clusters={len(np.unique(y))}")
    return X, y


# =========================================================
# PLOTTING
# =========================================================
def plot_pca_progression(embeddings, pca_dims, y):
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
        "#1f77b4", "#4fa3d1", "#a6cee3",   # blues
        "#d95f02", "#ff7f0e", "#fdd49e",   # oranges
    ]
    cmap = ListedColormap(colors)

    for ax, Z, n_comps in zip(axes, embeddings, pca_dims):
        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y % len(colors),
            cmap=cmap,
            s=5,
            alpha=0.4
        )

        ax.set_title(f"PCA = {n_comps}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("0.85")

    plt.subplots_adjust(wspace=0.0)

    filename = os.path.join(OUT_DIR, "rna_pca_progression.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved figure -> {filename}")


# =========================================================
# CORE
# =========================================================
def run_pca_ablation(seed=42):
    print("\n=== PCA Ablation (λ = 0.01) ===")

    os.makedirs(OUT_DIR, exist_ok=True)

    embeddings = []
    results = []

    for n_comps in PCA_DIMS:
        print(f"\n--- PCA = {n_comps} ---")

        X, y = load_pbmc(n_comps=n_comps)

        rho_high, knn_indices = compute_knn_density(X)
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
            f"PCA={n_comps} | "
            f"TW={tw:.4f} | CONT={cont:.4f} | "
            f"DENS={dens:.4f} | TIME={runtime:.2f}"
        )

        results.append({
            "pca": n_comps,
            "trustworthiness": tw,
            "continuity": cont,
            "density_corr": dens,
            "time_sec": runtime,
        })

    # =========================
    # SAVE CSV
    # =========================
    csv_path = os.path.join(OUT_DIR, "rna_pca_ablation.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV -> {csv_path}")

    # =========================
    # PLOT
    # =========================
    plot_pca_progression(embeddings, PCA_DIMS, y)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    run_pca_ablation(seed=SEED)