import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from density_tsne import run_density_tsne

# Replace this import with the actual module where these live
from comparison import (
    compute_knn_density,
    compute_P,
    continuity,
    density_correlation,
)


# =========================================================
# HELPERS
# =========================================================

def timed_run(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    runtime = time.time() - start
    return result, runtime


# =========================================================
# DATA LOADERS
# =========================================================

def load_digits_data(n_samples=1500, seed=42):
    X, y = load_digits(return_X_y=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=2000, seed=42):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X = X.astype(np.float32)
    y = y.astype(int)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_fashion_mnist(n_samples=2000, seed=42):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)

    X = X.astype(np.float32)
    y = y.astype(int)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_pbmc():
    raise NotImplementedError("Implement your RNA loader here.")


# =========================================================
# PLOTTING
# =========================================================
def format_lambda(lam):
    if lam == 0:
        return r'$\lambda = 0$'
    exp = int(np.log10(lam))
    if 10**exp == lam:
        return rf'$\lambda = 10^{{{exp}}}$'
    else:
        return rf'$\lambda = {lam:.2f}$'

def plot_lambda_progression(embeddings, lambdas, y, dataset_name):
    out_dir = "output/ablation_lambda"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        1, len(embeddings),
        figsize=(4 * len(embeddings), 4),
        gridspec_kw={'wspace': 0.0}   # 🔥 remove horizontal spacing
    )

    if len(embeddings) == 1:
        axes = [axes]

    def format_lambda(lam):
        if lam == 0:
            return r'$\lambda = 0$'
        exp = int(np.log10(lam))
        if np.isclose(10**exp, lam):
            return rf'$\lambda = 10^{{{exp}}}$'
        else:
            return rf'$\lambda = {lam:.2f}$'

    num_classes = len(np.unique(y))
    from matplotlib.colors import ListedColormap
    colors = [
        # Blues (unchanged — already good)
        "#1f77b4",  # strong blue
        "#4fa3d1",  # medium blue
        "#a6cee3",  # light blue

        # Oranges (improved contrast)
        "#d95f02",  # dark orange (more brownish, strong anchor)
        "#ff7f0e",  # standard orange
        "#fdd49e",  # very light orange (almost pastel)
    ]
    cmap = ListedColormap(colors)
    for ax, Z, lam in zip(axes, embeddings, lambdas):
        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            c=y % len(colors),  # ensures mapping if more clusters
            cmap=cmap,
            s=6,
            alpha=0.4
        )

        ax.set_title(format_lambda(lam), fontsize=9)

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("0.88")

    # 🔥 remove ALL padding
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    filename = os.path.join(out_dir, f"{dataset_name}_lambda_progression.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved progression image -> {filename}")


def plot_lambda_metrics(results, dataset_name):
    out_dir = "output/images"
    os.makedirs(out_dir, exist_ok=True)

    lambdas = np.array([row["lambda"] for row in results], dtype=float)
    trust_vals = np.array([row["trustworthiness"] for row in results], dtype=float)
    dens_vals = np.array([row["density_corr"] for row in results], dtype=float)

    lambdas_plot = np.where(lambdas > 0, lambdas, 1e-6)

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas_plot, trust_vals, marker="o", label="Trustworthiness")
    plt.plot(lambdas_plot, dens_vals, marker="o", label="Density Corr")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Metric value")
    plt.title(dataset_name)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(out_dir, f"{dataset_name}_lambda_metrics.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved metric plot -> {filename}")


# =========================================================
# ABLATION CORE
# =========================================================

def run_lambda_ablation(X, y, dataset_name, seed=42):
    print(f"\n=== λ Ablation: {dataset_name} ===")

    print("Computing density...")
    rho_high, knn_indices = compute_knn_density(X)

    print("Computing P...")
    P = compute_P(X)

    # If compute_P returns numpy, convert once here
    try:
        import torch
        if not isinstance(P, torch.Tensor):
            P = torch.tensor(P, dtype=torch.float32)
    except Exception:
        pass

    lambdas = [0.0, 0.01, 0.1]

    results = []
    embeddings = []

    for lam in lambdas:
        print(f"\nRunning λ = {lam:.5f}")

        (Z, history), runtime = timed_run(
            f"{dataset_name} (λ={lam:.5f})",
            run_density_tsne,
            X=X,
            P=P,
            knn_indices=knn_indices,
            rho_high=rho_high,
            lambda_density=lam,
            seed=seed,
            verbose=True,
        )

        embeddings.append(Z)

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
            "time_sec": runtime,
        })

    os.makedirs("output", exist_ok=True)

    csv_path = f"output/{dataset_name}_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV -> {csv_path}")

    plot_lambda_progression(embeddings, lambdas, y, dataset_name)
    plot_lambda_metrics(results, dataset_name)


# =========================================================
# MAIN
# =========================================================

def load_pbmc():
    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)

    # ✅ SAME AS MAIN SCRIPT
    sc.tl.pca(adata, n_comps=30)

    # ✅ SAME AS MAIN SCRIPT
    sc.pp.neighbors(adata, n_neighbors=300, n_pcs=20)

    # clustering identical
    sc.tl.louvain(adata, resolution=1.0)

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values

    return X, y

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)


    # RNA
    X, y = load_pbmc()
    run_lambda_ablation(X, y, "rna", seed=seed)