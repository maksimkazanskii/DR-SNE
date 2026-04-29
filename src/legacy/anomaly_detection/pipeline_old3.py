import numpy as np
import scanpy as sc
import torch
import random
import time
import os
import csv

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import umap
import pacmap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from density_tsne import run_density_tsne
from sklearn.datasets import load_breast_cancer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models import ResNet18_Weights

device ="cpu"
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])


from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np



def load_thyroid_easy(seed=42, n_samples=5000):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd

    print("Loading thyroid-ann...")

    X, y = fetch_openml("thyroid-ann", return_X_y=True, as_frame=True)

    # -------------------------
    # Handle labels robustly
    # -------------------------
    y = pd.Series(y)

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    # majority class = normal
    counts = y.value_counts()
    normal_class = counts.idxmax()

    print("Class counts:")
    print(counts)
    print(f"Using normal class: {normal_class}")

    y_anomaly = (y != normal_class).astype(int).values

    # -------------------------
    # Handle features robustly
    # -------------------------
    X = pd.DataFrame(X)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # if any columns are still all-NaN/non-numeric after median fill
    X = X.fillna(0)

    X = X.values.astype(np.float32)

    # -------------------------
    # Subsample
    # -------------------------
    rng = np.random.RandomState(seed)
    if n_samples is not None and len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y_anomaly = y_anomaly[idx]

    # -------------------------
    # Scale
    # -------------------------
    X = StandardScaler().fit_transform(X)

    print(f"Shape: {X.shape}")
    print(f"Anomaly %: {y_anomaly.mean() * 100:.2f}%")

    return X, y_anomaly

def load_mvtec_wood(
        root="data/mvtec/wood",
        n_samples=2000,
        seed=42,
        include_train=True
):
    import os
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)

    X = []
    y = []

    # -------------------------
    # helper
    # -------------------------
    def process_folder(folder, label):
        feats = []
        labels = []

        if not os.path.exists(folder):
            return feats, labels

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for fname in files:
            path = os.path.join(folder, fname)

            try:
                img = Image.open(path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model(img).view(-1).cpu().numpy()

                feats.append(feat)
                labels.append(label)

            except Exception:
                continue

        return feats, labels

    # -------------------------
    # TRAIN (normal only)
    # -------------------------
    if include_train:
        train_good = os.path.join(root, "train", "good")
        feats, labels = process_folder(train_good, 0)
        X.extend(feats)
        y.extend(labels)

    # -------------------------
    # TEST
    # -------------------------
    test_dir = os.path.join(root, "test")

    for subfolder in os.listdir(test_dir):
        subpath = os.path.join(test_dir, subfolder)

        if not os.path.isdir(subpath):
            continue

        # good = normal
        label = 0 if subfolder == "good" else 1

        feats, labels = process_folder(subpath, label)

        X.extend(feats)
        y.extend(labels)

    # -------------------------
    # convert to numpy
    # -------------------------
    X = np.array(X)
    y = np.array(y)

    # -------------------------
    # subsample (important)
    # -------------------------
    if len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # -------------------------
    # normalize (important!)
    # -------------------------
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    print(f"MVTec wood: {X.shape}")
    print(f"Anomaly %: {y.mean()*100:.2f}%")

    return X, y

def load_synthetic_anomaly(n_samples=5000, dim=30, seed=42):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    n1 = int(n_samples * 0.5)
    n2 = int(n_samples * 0.3)
    n3 = n_samples - n1 - n2

    mean = np.zeros(dim)

    # cluster 1: very dense
    X1 = rng.multivariate_normal(mean, np.eye(dim) * 0.3, size=n1)

    # cluster 2: medium
    X2 = rng.multivariate_normal(mean + 2, np.eye(dim) * 1.0, size=n2)

    # cluster 3: very spread (anomalies)
    X3 = rng.multivariate_normal(mean + 1, np.eye(dim) * 1.7, size=n3)

    X = np.vstack([X1, X2, X3])

    # define anomaly: ONLY the most spread cluster
    y = np.concatenate([
        np.zeros(n1),
        np.zeros(n2),
        np.ones(n3)
    ])

    # shuffle
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)

    return X, y

def load_fashion_mnist_anomaly(anomaly_class=0, seed=42, n_samples=5000):
    print("Loading Fashion-MNIST...")

    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)

    # 🔥 FIX
    X = X.to_numpy()
    y = y.to_numpy().astype(int)

    # define anomaly
    y_anomaly = (y == anomaly_class).astype(int)

    # subsample
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    X = X[idx]
    y_anomaly = y_anomaly[idx]

    # normalize
    X = StandardScaler().fit_transform(X)

    print(f"Shape: {X.shape}")
    print(f"Anomaly %: {y_anomaly.mean()*100:.2f}%")

    return X, y_anomaly

def set_global_seed(seed=42):
    import os
    import random
    import numpy as np

    # -------------------------
    # Python & NumPy
    # -------------------------
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # -------------------------
    # Torch (CPU + GPU)
    # -------------------------
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # strict determinism (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True)

    except Exception:
        pass

    # -------------------------
    # Extra (VERY IMPORTANT)
    # -------------------------
    # Controls parallelism randomness (BLAS / OpenMP)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


set_global_seed(42)


def sweep_density_tsne_2d(X, y, seed, results, density_k=30):
    print("\n==============================")
    print("Density t-SNE 2D SWEEP (IMPROVED)")
    print("==============================")

    # 🔥 base lambda grid (log-scale, extended)
    base_lambda_list = [
        1e-4, 5e-4,
        1e-3, 5e-3,
        1e-2, 5e-2,
        1e-1, 5e-1,
        1.0, 2.0, 5.0, 10.0
    ]

    # 🔥 better k grid (log-like spacing)
    k_list = [10, 20, 30, 50, 80, 120, 200, 350]

    for k in k_list:
        print(f"\n[Density t-SNE] k = {k}")

        # 🔥 couple perplexity with k
        perplexity = max(5, int(k / 3))
        print(f"Using perplexity = {perplexity}")

        P = compute_P(X, perplexity=perplexity)

        rho_high, knn_indices = compute_knn_density(X, k=k)
        rho_high = np.clip(rho_high, 1e-6, None)

        # 🔥 scale lambda based on k
        lambda_list = [lam * (30.0 / k) for lam in base_lambda_list]

        for lam, lam_eff in zip(base_lambda_list, lambda_list):
            start = time.time()

            Z, _ = run_density_tsne(
                X,
                P,
                knn_indices,
                rho_high,
                lambda_density=lam_eff,
                seed=seed,
                verbose=False
            )

            if isinstance(Z, torch.Tensor):
                Z = Z.detach().cpu().numpy()

            scores = compute_density_scores_transductive_robust(Z, k=density_k)

            elapsed = time.time() - start
            auroc = roc_auc_score(y, scores)
            auprc = average_precision_score(y, scores)

            print(
                f"{'Density t-SNE':15s} | "
                f"k={k:4d} | λ_base={lam:.4g} | λ_eff={lam_eff:.4g} | "
                f"AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
                f"Time={elapsed:.2f}s"
            )

            results.append({
                "method": "Density t-SNE",
                "param_name": "k_lambda",
                "param_value": f"{k}_{lam_eff:.4g}",
                "auroc": auroc,
                "auprc": auprc,
                "time_sec": elapsed
            })

    return results
# =========================================================
# ADD THESE FUNCTIONS (above MAIN PIPELINE)
# =========================================================

def sweep_tsne_2d(X, y, seed, results, density_k=30):
    for perp in [5, 10, 20, 30, 50, 80, 120]:
        for lr in [50, 100, 200, 500, 1000]:
            ee = max(12, perp * 0.5)  # 🔥 adaptive

            results.append(
                evaluate_method_transductive(
                    "t-SNE",
                    lambda X_, p=perp, lr_=lr, ee_=ee: TSNE(
                        n_components=2,
                        perplexity=p,
                        learning_rate=lr_,
                        early_exaggeration=ee_,
                        init="random",
                        random_state=seed
                    ).fit_transform(X_),
                    X,
                    y,
                    "perp_lr",
                    f"{perp}_{lr}",
                    density_k=density_k
                )
            )
    return results


def sweep_umap_2d(X, y, seed, results, density_k=30):
    for k in [10, 20, 30, 50, 80, 120]:
        for md in [0.0, 0.1, 0.3, 0.5]:
            spread = 1.0 + md * 2  # 🔥 coupling

            results.append(
                evaluate_method_transductive(
                    "UMAP",
                    lambda X_, kk=k, mm=md, sp=spread: umap.UMAP(
                        n_neighbors=kk,
                        min_dist=mm,
                        spread=sp,
                        n_components=2,
                        random_state=seed
                    ).fit_transform(X_),
                    X,
                    y,
                    "k_minDist",
                    f"{k}_{md}",
                    density_k=density_k
                )
            )
    return results


def sweep_pacmap_2d(X, y, seed, results, density_k=30):
    for k in [10, 20, 30, 50, 80]:
        for mn in [0.5, 1.0, 2.0]:
            for fp in [1.0, 2.0, 4.0]:  # 🔥 new

                results.append(
                    evaluate_method_transductive(
                        "PaCMAP",
                        lambda X_, kk=k, mm=mn, ff=fp: pacmap.PaCMAP(
                            n_components=2,
                            n_neighbors=kk,
                            MN_ratio=mm,
                            FP_ratio=ff,
                            random_state=seed
                        ).fit_transform(X_),
                        X,
                        y,
                        "k_mn_fp",
                        f"{k}_{mn}_{fp}",
                        density_k=density_k
                    )
                )
    return results


def sweep_densmap_2d(X, y, seed, results, density_k=30):
    for lam in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
        for k in [10, 20, 30, 50, 80]:
            for frac in [0.2, 0.5, 0.8]:  # 🔥 new

                results.append(
                    evaluate_method_transductive(
                        "DensMAP",
                        lambda X_, ll=lam, kk=k, ff=frac: umap.UMAP(
                            n_neighbors=kk,
                            min_dist=0.1,
                            n_components=2,
                            densmap=True,
                            dens_lambda=ll,
                            dens_frac=ff,
                            output_dens=True,
                            random_state=seed
                        ).fit_transform(X_)[0],
                        X,
                        y,
                        "λ_k_frac",
                        f"{lam}_{k}_{frac}",
                        density_k=density_k
                    )
                )
    return results


def load_shuttle(n_samples=5000, seed=42):
    print("Loading Shuttle dataset...")

    X, y = fetch_openml(
        "shuttle",
        version=1,
        return_X_y=True,
        as_frame=False
    )

    y = y.astype(int)

    # ----------------------------------
    # TRUE anomaly labels
    # class 1 = normal, others = anomaly
    # ----------------------------------
    y_binary = (y != 1).astype(int)

    # ----------------------------------
    # Subsample
    # ----------------------------------
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y_binary = y_binary[idx]

    # ----------------------------------
    # Scale features
    # ----------------------------------
    X = StandardScaler().fit_transform(X)

    print(f"Dataset shape: {X.shape}")
    print(f"Anomaly %: {y_binary.mean() * 100:.2f}%")

    return X, y_binary


def load_tumor(seed = 42, path="data/melanoma.h5ad"):
    print("Loading melanoma tumor dataset...")

    import scanpy as sc
    import numpy as np

    # Load dataset (fallback to PBMC if file not found)
    try:
        adata = sc.read_h5ad(path)
    except FileNotFoundError:
        print(f"File {path} not found. Using fallback dataset (pbmc3k).")
        adata = sc.datasets.pbmc3k()

    # -------------------------
    # Features
    # -------------------------
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # -------------------------
    # Labels
    # -------------------------
    if 'cell_type' in adata.obs:
        y = adata.obs['cell_type'].values
    elif 'cell_subtype' in adata.obs:
        y = adata.obs['cell_subtype'].values
    elif 'annotation' in adata.obs:
        y = adata.obs['annotation'].values
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, key_added="leiden")
        y = adata.obs['leiden'].values

    y = np.array(y)

    # -------------------------
    # ✅ FIX: correct masking
    # -------------------------
    bad_labels = {'unknown', 'NA', 'None', ''}
    mask = np.array([label not in bad_labels for label in y])

    X = X[mask]
    y = y[mask]

    # -------------------------
    # Encode labels
    # -------------------------
    _, y_encoded = np.unique(y, return_inverse=True)

    print(f"Final dataset: {X.shape}, classes: {len(np.unique(y_encoded))}")
    return X, y_encoded
# =========================================================
# DEFINE ANOMALIES (RARE CLUSTERS)
# =========================================================
def anomaly_pbmc(y, threshold=0.05, verbose=True):
    """
    Define anomalies as rare cell types based on true labels.

    Parameters
    ----------
    y : array-like
        Ground-truth labels (e.g. cell types)
    threshold : float
        Frequency threshold below which a type is considered rare
    verbose : bool
        Whether to print diagnostics

    Returns
    -------
    labels : np.ndarray
        Binary anomaly labels (1 = anomaly, 0 = normal)
    """

    y = np.asarray(y)

    unique, counts = np.unique(y, return_counts=True)
    freq = counts / len(y)

    rare_types = unique[freq < threshold]
    labels = np.isin(y, rare_types).astype(int)

    if verbose:
        print("\n=== Anomaly definition (true labels) ===")
        for u, c, f in zip(unique, counts, freq):
            print(f"{str(u):20s} | count={c:4d} | freq={f:.4f}")

        print(f"\nRare types (freq < {threshold}): {list(rare_types)}")
        print(f"Anomaly %: {labels.mean() * 100:.2f}%")

    return labels


# =========================================================
# DENSITY SCORE (TRANSDUCTIVE)
# =========================================================
def compute_density_scores_transductive_robust(Z, k=30, alpha=0.5):
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        return np.full(len(Z), np.nan)

    n = len(Z)
    k_eff = min(k, n)

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Z)
    dist, _ = nbrs.kneighbors(Z)

    # -------------------------
    # 1. density (same as before)
    # -------------------------
    volume = dist[:, 1:].sum(axis=1)
    volume = np.clip(volume, 1e-6, None)

    rho = (k_eff - 1) / volume
    rho = rho / (rho.mean() + 1e-6)

    density_score = -np.log(rho + 1e-6)

    # -------------------------
    # 2. isolation score (NEW)
    # mean distance to neighbors
    # -------------------------
    isolation_score = dist[:, 1:].mean(axis=1)

    # normalize isolation
    isolation_score = isolation_score / (isolation_score.mean() + 1e-6)

    # -------------------------
    # 3. rank normalization (VERY IMPORTANT)
    # makes methods comparable
    # -------------------------
    def rank_norm(x):
        ranks = np.argsort(np.argsort(x))
        return ranks / (len(x) - 1 + 1e-8)

    density_rank = rank_norm(density_score)
    isolation_rank = rank_norm(isolation_score)

    # -------------------------
    # 4. combined score
    # -------------------------
    score = density_rank + alpha * isolation_rank

    return score

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
    if k_eff < 2:
        raise ValueError("Need at least 2 points for density estimation.")

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
# EVALUATION WRAPPER (TRANSDUCTIVE)
# =========================================================
def evaluate_method_transductive(
        method_name,
        embed_fn,
        X,
        y,
        param_name="-",
        param_value="-",
        density_k=30
):
    start = time.time()

    Z = embed_fn(X)
    scores = compute_density_scores_transductive_robust(Z, k=density_k)

    elapsed = time.time() - start

    auroc = roc_auc_score(y, scores)
    auprc = average_precision_score(y, scores)

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
# DENSITY t-SNE SWEEP (TRANSDUCTIVE)
# =========================================================
def run_density_tsne_sweep_transductive(X, y, seed, results, density_k=30):
    print("\n==============================")
    print("Density t-SNE FULL SWEEP")
    print("==============================")

    lambda_list = [
        1e-4,  5e-4,
        1e-3,  5e-3,
        1e-2,  5e-2,
        1e-1,  5e-1
    ]
    k_list = [10, 30, 60, 140, 220, 300, 380, 460]

    P = compute_P(X, perplexity=50.0)

    for k in k_list:
        print(f"\n[Density t-SNE] k = {k}")

        rho_high, knn_indices = compute_knn_density(X, k=k)

        rho_high = np.clip(rho_high, 1e-6, None)

        for lam in lambda_list:
            start = time.time()

            Z, _ = run_density_tsne(
                X,
                P,
                knn_indices,
                rho_high,
                lambda_density=lam,
                seed=seed,
                verbose=False
            )

            scores = compute_density_scores_transductive_robust(Z, k=density_k)

            elapsed = time.time() - start
            auroc = roc_auc_score(y, scores)
            auprc = average_precision_score(y, scores)

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
def run_dataset_experiment(load_fn, name, seed=42):
    set_global_seed(seed)

    print(f"\n=== Running {name} ===")

    X, anomaly_labels = load_fn(seed=seed)

    results = []

    # RAW
    scores = compute_density_scores_transductive_robust(X, k=30)
    results.append({
        "method": "RAW",
        "param_name": "-",
        "param_value": "-",
        "auroc": roc_auc_score(anomaly_labels, scores),
        "auprc": average_precision_score(anomaly_labels, scores),
        "time_sec": 0
    })

    # PCA
    results.append(
        evaluate_method_transductive(
            "PCA",
            lambda X_: run_pca_2d(X_, seed=seed),
            X,
            anomaly_labels
        )
    )



    # BASELINES
    results = sweep_tsne_2d(X, anomaly_labels, seed, results)
    results = sweep_umap_2d(X, anomaly_labels, seed, results)
    results = sweep_pacmap_2d(X, anomaly_labels, seed, results)
    results = sweep_densmap_2d(X, anomaly_labels, seed, results)
    results = sweep_density_tsne_2d(X, anomaly_labels, seed, results)

    save_results_csv(results, f"output/anomaly/{name}_seed_{seed}.csv")

def load_mvtec_wrapper(seed=42):
    return load_mvtec_wood(
        root="data/mvtec/wood",
        n_samples=2000,
        seed=seed,
        include_train=True
    )

def load_thyroid_wrapper(seed=42):
    return load_thyroid_easy(
        seed=seed,
        n_samples=5000   # or None to use all data
    )

def run_all():
    for seed in [0,1,2]:
        run_dataset_experiment(load_thyroid_wrapper, "thyroid_large", seed)
        run_dataset_experiment(load_mvtec_wrapper, "mvtec_grid", seed)
        run_dataset_experiment(load_synthetic_anomaly, "synthetic", seed)
        run_dataset_experiment(load_shuttle, "shuttle", seed)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    run_all()
