import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, trustworthiness
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances, silhouette_score
import os, csv
from openTSNE import TSNE as OpenTSNE

from density_tsne import run_density_tsne


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


def run_densne(X):
    tsne = OpenTSNE(
        n_components=2,
        perplexity=30,
        initialization="pca",
        n_jobs=8,
        random_state=42
    )
    return tsne.fit(X)


def save_results_to_csv(dataset_name, results):
    filename = f"{dataset_name}_results.csv"

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

        for r in results:
            writer.writerow({
                "method": r["name"],
                "trustworthiness": r["trustworthiness"],
                "continuity": r["continuity"],
                "density_corr": r["density_corr"],
                "silhouette": r["silhouette"],
                "stress": r["stress"],
                "time_sec": r["time"]
            })

    print(f"\nResults saved to {filename}")


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
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
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

        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = Pi

    P = (P + P.T) / (2 * n)
    return torch.tensor(P, dtype=torch.float32)


# =========================================================
# BASELINES
# =========================================================
def run_tsne(X,seed=42):
    tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=seed)
    return tsne.fit_transform(X)


def run_umap(X):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=None
    )
    return reducer.fit_transform(X)


def run_densmap(X):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        densmap=True,
        dens_lambda=2.0,
        dens_frac=0.3,
        random_state=None
    )
    return reducer.fit_transform(X)


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

