import torch
import time
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, trustworthiness
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances, silhouette_score
import os, csv
import numpy as np
import pacmap
import trimap


from density_tsne import run_density_tsne

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DRSNE.drsne import drsne
def run_drsne(
        X,
        seed=42,
        lambda_density=0.01,
        k_density=30,
        knn_indices=None,
        rho_high=None,
        P=None
):
    Z, _ = drsne(
        X,
        P=P,
        knn_indices=knn_indices,
        rho_high=rho_high,
        lambda_density=lambda_density,
        seed=seed,
        verbose=False
    )

    return Z

import sys
sys.path.append("densvis/densne")

import densne

def run_densne(
        X,
        seed=42,
        perplexity=30,
        dens_lambda=0.01,
        theta=0.5,
        dens_frac=0.5,
        max_iter=800,
        dim=2,
        initial_dims=None
):
    rng = np.random.RandomState(seed)

    init_emb = 1e-4 * rng.normal(size=(X.shape[0], dim)).astype(np.float64)

    Z = densne.run_densne(
        np.asarray(X, dtype=np.float64),
        no_dims=dim,
        perplexity=float(perplexity),
        theta=float(theta),
        randseed=int(seed),
        verbose=False,
        initial_dims=initial_dims,
        use_pca=False,
        max_iter=int(max_iter),
        dens_frac=float(dens_frac),
        dens_lambda=float(dens_lambda),
        final_dens=True,  # ← you enabled this
        initial_emb=init_emb
    )

    # 🔥 CRITICAL FIX
    if isinstance(Z, tuple):
        Z = Z[0]

    return Z
def clean_labels(X, y):
    import numpy as np

    if y is None:
        return X, None

    y = np.asarray(y)

    # remove NaN / None
    if y.dtype.kind in ["f"]:  # float → can have NaN
        mask = ~np.isnan(y)
    else:
        mask = y != None

    X = X[mask]
    y = y[mask]

    # remove "unknown"-type labels (optional but useful)
    if y.dtype.type is np.str_ or y.dtype == object:
        bad_values = {"unknown", "NA", "None", ""}
        mask = np.array([v not in bad_values for v in y])
        X = X[mask]
        y = y[mask]

    # encode labels → integers
    unique, y_encoded = np.unique(y, return_inverse=True)

    # ensure valid for silhouette (≥2 per class)
    counts = np.bincount(y_encoded)
    valid_classes = np.where(counts >= 2)[0]

    mask = np.isin(y_encoded, valid_classes)
    X = X[mask]
    y_encoded = y_encoded[mask]

    if len(np.unique(y_encoded)) < 2:
        return X, None

    return X, y_encoded

# =========================================================
# IO
# =========================================================
def save_embeddings_image(Z_list, titles, y, dataset_name):
    import os
    import matplotlib.pyplot as plt
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/images/comparison", exist_ok=True)

    fig, axes = plt.subplots(1, len(Z_list), figsize=(5 * len(Z_list), 5))

    if len(Z_list) == 1:
        axes = [axes]

    for ax, Z, title in zip(axes, Z_list, titles):

        if y is None:
            ax.scatter(Z[:, 0], Z[:, 1], s=8)  # no coloring
        else:
            ax.scatter(Z[:, 0], Z[:, 1], c=y.astype(int), cmap="tab10", s=8)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    filename = f"{dataset_name}_comparison.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved embedding image to {filename}")



def stress_metric(X, Z):
    D_high = pairwise_distances(X)
    D_low = pairwise_distances(Z)

    num = np.sum((D_high - D_low) ** 2)
    den = np.sum(D_high ** 2) + 1e-8

    return np.sqrt(num / den)


def save_ablation_to_csv(dataset_name, results):
    filename = f"output/{dataset_name}_ablation.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "trustworthiness",
                "continuity",
                "density_corr",
                "silhouette",
                "stress",
                "time_sec"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nAblation saved to {filename}")

def save_best_results_to_csv(dataset_name, results):
    filename = f"{dataset_name}_best_results.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "param_name",
                "param_value",
                "trustworthiness",
                "continuity",
                "density_corr",
                "silhouette",
                "stress",
                "time_sec"
            ]
        )

        writer.writeheader()

        for r in results:
            writer.writerow({
                "method": r["name"],
                "param_name": r["param_name"],
                "param_value": r["param_value"],
                "trustworthiness": r["trustworthiness"],
                "continuity": r["continuity"],
                "density_corr": r["density_corr"],
                "silhouette": r["silhouette"],
                "stress": r["stress"],
                "time_sec": r["time"]
            })

    print(f"\nBest results saved to {filename}")

def select_best_under_tw(candidates, tw_threshold):
    valid = [c for c in candidates if c["trustworthiness"] >= tw_threshold]

    if len(valid) == 0:
        print(f"No configuration reached trustworthiness >= {tw_threshold:.3f}")
        return None

    best = max(valid, key=lambda x: x["density_corr"])
    return best

def tune_1d_method(
        method_name,
        X,
        y,
        knn_indices,
        rho_high,
        seed,
        tw_threshold,
        param_name,
        param_values,
        P=None
):
    candidates = []

    for param_value in param_values:
        print(f"\n{method_name} | {param_name}={param_value}")

        if method_name == "t-SNE":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_tsne,
                X,
                seed,
                param_value
            )



        elif method_name == "UMAP":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_umap,
                X,
                seed,
                param_value
            )

        elif method_name == "PaCMAP":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_pacmap,
                X,
                seed,
                n_neighbors=param_value
            )

        elif method_name == "DR-SNE":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_drsne,
                X,
                seed,
                lambda_density=param_value,
                k_density=30,
                knn_indices=knn_indices,
                rho_high=rho_high,
                P=P
            )
        elif method_name == "TriMAP":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_trimap,
                X,
                seed,
                n_inliers=param_value
            )
        elif method_name == "DensMAP":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_densmap,
                X,
                seed,
                dens_lambda=param_value   # 🔥 SAFE
            )
        elif method_name == "DenSNE":
            Z, runtime = timed_run(
                f"{method_name} ({param_name}={param_value})",
                run_densne,
                X,
                seed,
                perplexity=30,           # 🔒 geometry fixed
                dens_lambda=param_value, # 🔥 tuning
                theta=0.5,
                dens_frac=0.5,
                max_iter=800,
                dim=2,
                initial_dims=None
            )
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        metrics = compute_metrics_for_method(
            method_name,
            X,
            Z,
            knn_indices,
            rho_high,
            runtime,
            y
        )
        metrics["param_name"] = param_name
        metrics["param_value"] = param_value
        metrics["Z"] = Z
        candidates.append(metrics)

    best = select_best_under_tw(candidates, tw_threshold)
    return best, candidates

def tune_density_tsne(
        X,
        y,
        P,
        knn_indices,
        rho_high,
        seed,
        tw_threshold
):
    lambda_grid = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
    candidates = []

    for lambda_density in lambda_grid:
        print(f"\nDensity t-SNE | lambda_density={lambda_density}")

        (Z, _), runtime = timed_run(
            f"Density t-SNE (lambda={lambda_density})",
            run_density_tsne,
            X,
            P,
            knn_indices,
            rho_high,
            lambda_density=lambda_density,
            seed=seed
        )

        metrics = compute_metrics_for_method(
            "Density t-SNE",
            X,
            Z,
            knn_indices,
            rho_high,
            runtime,
            y
        )
        metrics["param_name"] = "lambda_density"
        metrics["param_value"] = lambda_density
        metrics["Z"] = Z
        candidates.append(metrics)

    best = select_best_under_tw(candidates, tw_threshold)
    return best, candidates

# =========================================================
# METRICS
# =========================================================
def continuity(X, Z, n_neighbors=10):
    return trustworthiness(Z, X, n_neighbors=n_neighbors)


def density_correlation(Z, knn_indices, rho_high):
    rho_low = []

    for i in range(len(Z)):
        neighbors = knn_indices[i][1:]
        dists = np.linalg.norm(Z[i] - Z[neighbors], axis=1)
        volume = dists.sum() + 1e-8
        rho_low.append(len(neighbors) / volume)

    rho_low = np.array(rho_low)
    rho_low = rho_low / (rho_low.mean() + 1e-8)

    return np.corrcoef(
        np.log(rho_high + 1e-8),
        np.log(rho_low + 1e-8)
    )[0, 1]


# ✅ NEW ROBUST SILHOUETTE FUNCTION
def compute_silhouette(Z, y):
    if y is None:
        return np.nan

    y = np.asarray(y)
    Z = np.asarray(Z)

    # encode safely
    _, y_encoded = np.unique(y, return_inverse=True)

    # need ≥ 2 clusters
    if len(np.unique(y_encoded)) < 2:
        return np.nan

    # each cluster ≥ 2 samples
    counts = np.bincount(y_encoded)
    if np.min(counts) < 2:
        return np.nan

    # invalid embedding
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        return np.nan

    # collapsed embedding
    if np.allclose(Z, Z[0]):
        return np.nan

    try:
        return float(silhouette_score(Z, y_encoded))
    except Exception:
        return np.nan


# =========================================================
# DATA
# =========================================================
def load_data(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    X, y = load_digits(return_X_y=True)

    if n_samples < len(X):
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


# =========================================================
# DENSITY
# =========================================================
def compute_knn_density(X, k=30):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(X)
    distances, indices = nbrs.kneighbors(X)

    volume = distances[:, 1:].sum(axis=1) + 1e-8
    density = (k - 1) / volume
    density = density / (density.mean() + 1e-8)

    return density, indices


# =========================================================
# HIGH-D SIMILARITY
# =========================================================
def compute_P(X, perplexity=30.0, tol=1e-5):
    n = X.shape[0]
    D = pairwise_distances(X, squared=True, n_jobs=1)

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

        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = Pi

    P = (P + P.T) / (2 * n)
    return torch.tensor(P, dtype=torch.float32)


# =========================================================
# BASELINES
# =========================================================
def run_tsne(X, seed=42, perplexity=30):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        random_state=seed
    )
    return tsne.fit_transform(X)

def run_pacmap(X, seed=42, n_neighbors=10):
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed
    )
    return reducer.fit_transform(X)


def run_trimap(X, seed=42, n_inliers=10):
    reducer = trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=5,
        n_random=5,
        random_state=seed
    )
    return reducer.fit_transform(X)

def run_umap(X, seed=42, n_neighbors=30):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=seed
    )
    return reducer.fit_transform(X)


def run_densmap(
        X,
        seed=42,
        dens_lambda=2.0,
        n_neighbors=15
):
    """
    Standard DensMAP embedding
    """

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

    # 🔥 robust handling across UMAP versions
    if isinstance(result, tuple):
        Z = result[0]
    else:
        Z = result

    return Z


# =========================================================
# TIMING
# =========================================================
def timed_run(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"{name} time: {elapsed:.2f} sec")
    return result, elapsed


# =========================================================
# REPORTING
# =========================================================
def compute_metrics_for_method(name, X, Z, knn_indices, rho_high, runtime, y=None):
    tw = trustworthiness(X, Z, n_neighbors=10)
    cont = continuity(X, Z, n_neighbors=10)
    dens = density_correlation(Z, knn_indices, rho_high)
    stress = stress_metric(X, Z)

    sil = compute_silhouette(Z, y)

    print(
        f"{name:15s} | "
        f"Trustworthiness={tw:.4f} | "
        f"Continuity={cont:.4f} | "
        f"DensityCorr={dens:.4f} | "
        + (f"Silhouette={sil:.4f} | " if not np.isnan(sil) else "Silhouette=nan | ")
        + f"Stress={stress:.4f} | "
          f"Time={runtime:.2f}s"
    )

    return {
        "name": name,
        "trustworthiness": tw,
        "continuity": cont,
        "density_corr": dens,
        "silhouette": sil,
        "stress": stress,
        "time": runtime
    }

