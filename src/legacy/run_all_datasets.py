from sklearn.datasets import load_digits, fetch_openml
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from src.legacy.comparison import (
    compute_knn_density,
    compute_P,
    run_tsne,
    run_density_tsne,
    run_umap,
    run_densmap,
    timed_run,
    compute_metrics_for_method,
    save_results_to_csv,
    save_embeddings_image
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scanpy as sc
import numpy as np
from densne_wrapper import run_densne

def load_sbert_shift(n_samples=5000):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("Loading AG News sentences...")

    dataset = load_dataset("ag_news", split="train")

    texts = []
    lengths = []

    for item in dataset:
        text = item["text"]
        l = len(text.split())

        if l > 5:  # avoid junk
            texts.append(text)
            lengths.append(l)

        if len(texts) >= n_samples:
            break

    texts = np.array(texts)
    lengths = np.array(lengths)

    # -------------------------
    # SPLIT: short vs long
    # -------------------------
    median_len = np.median(lengths)

    mask_short = lengths <= median_len
    mask_long = lengths > median_len

    texts_short = texts[mask_short]
    texts_long = texts[mask_long]

    # balance sizes
    n = min(len(texts_short), len(texts_long), n_samples // 2)

    texts_short = texts_short[:n]
    texts_long = texts_long[:n]

    texts_all = np.concatenate([texts_short, texts_long])

    # labels: 0 = short, 1 = long
    y = np.array([0]*n + [1]*n)

    print(f"Short: {n}, Long: {n}")

    # -------------------------
    # SBERT embeddings
    # -------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")

    X = model.encode(texts_all, batch_size=64, show_progress_bar=True)

    # normalize
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    return X, y, lengths[:2*n]

def download_tabula_muris(save_path="data/tabula_muris.h5ad"):
    import os
    import requests

    os.makedirs("data", exist_ok=True)

    url = "https://datasets.cellxgene.cziscience.com/8d4c3f02-9f7a-4b2d-9b5b-ecf6f4d0e9c3.h5ad"

    print("Downloading Tabula Muris Senis (FACS)...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Saved to {save_path}")


def load_tabula_muris(path="data/tabula_muris.h5ad", n_samples=5000):
    import scanpy as sc
    import numpy as np
    import os

    print("Loading Tabula Muris...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run download_tabula_muris() first."
        )

    adata = sc.read_h5ad(path)

    # -------------------------
    # Labels (verified column)
    # -------------------------
    if "cell_ontology_class" not in adata.obs:
        raise ValueError("Expected 'cell_ontology_class' in dataset")

    y = np.array(adata.obs["cell_ontology_class"])

    # -------------------------
    # Features
    # -------------------------
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # -------------------------
    # Clean labels
    # -------------------------
    mask = (y != "unknown") & (y != "NA") & (y != "")
    X = X[mask]
    y = y[mask]

    # -------------------------
    # Encode labels
    # -------------------------
    _, y = np.unique(y, return_inverse=True)

    # -------------------------
    # Remove tiny classes
    # -------------------------
    counts = np.bincount(y)
    valid = np.where(counts >= 5)[0]

    mask = np.isin(y, valid)
    X = X[mask]
    y = y[mask]

    # -------------------------
    # Subsample (IMPORTANT for speed)
    # -------------------------
    if n_samples is not None and len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"Final dataset: {X.shape}, classes: {len(np.unique(y))}")

    return X, y

def load_spiral_density(n_samples=5000):

    import numpy as np

    t = np.linspace(0, 4*np.pi, n_samples)

    # density increases with t
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)

    X = np.stack([x, y], axis=1)

    # true density ∝ 1 / spacing
    rho_true = 1 / (1 + t)

    return X, np.zeros(n_samples), rho_true

def load_synthetic_density(n_samples=5000, n_clusters=4, dim=50):

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    np.random.seed(42)

    X_list = []
    y_list = []
    rho_true = []

    samples_per_cluster = n_samples // n_clusters

    # Different variances → different densities
    variances = np.linspace(0.2, 2.0, n_clusters)

    for i, var in enumerate(variances):

        mean = np.random.randn(dim) * 5
        cov = np.eye(dim) * var

        X_cluster = np.random.multivariate_normal(
            mean, cov, size=samples_per_cluster
        )

        # True density ∝ 1 / variance^(dim/2)
        density = 1.0 / (var ** (dim / 2))

        X_list.append(X_cluster)
        y_list.append(np.full(samples_per_cluster, i))
        rho_true.append(np.full(samples_per_cluster, density))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    rho_true = np.concatenate(rho_true)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    rho_true = rho_true[idx]

    # Normalize + optional PCA (like your pipeline)
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    print("Synthetic clusters:", n_clusters)

    return X, y, rho_true

def load_tumor_melanoma(path="data/melanoma.h5ad"):
    print("Loading melanoma tumor dataset...")

    # Load dataset (fallback to PBMC if file not found)
    try:
        adata = sc.read_h5ad(path)
    except FileNotFoundError:
        print(f"File {path} not found. Using fallback dataset (pbmc3k).")
        adata = sc.datasets.pbmc3k()

    # Extract features (X)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()  # Convert sparse matrix to dense if necessary

    # Try to get labels from available keys (cell_type, cell_subtype, or annotation)
    if 'cell_type' in adata.obs:
        y = adata.obs['cell_type'].values
    elif 'cell_subtype' in adata.obs:
        y = adata.obs['cell_subtype'].values
    elif 'annotation' in adata.obs:
        y = adata.obs['annotation'].values
    else:
        # If no labels, compute Leiden clustering
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, key_added="leiden")
        y = adata.obs['leiden'].values

    # Remove invalid or unwanted labels (e.g., unknown, NA, or empty)
    bad_labels = {'unknown', 'NA', 'None', ''}
    y = np.array([label for label in y if label not in bad_labels])
    X = X[:len(y)]  # Ensure that X and y are aligned

    # Encode labels to integers
    _, y_encoded = np.unique(y, return_inverse=True)

    # Return the feature matrix (X) and the encoded labels (y)
    print(f"Final dataset: {X.shape}, classes: {len(np.unique(y_encoded))}")
    return X, y_encoded


def load_pbmc():

    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    # 🔥 FIX
    sc.tl.louvain(adata, resolution=1.0)

    X = adata.obsm["X_pca"]
    y = adata.obs["louvain"].astype(int).values

    print("PBMC clusters:", len(np.unique(y)))  # debug

    return X, y

def load_cifar10_embeddings(n_samples=5000):

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # pretrained ResNet18
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # remove classifier
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            f = model(x)
            features.append(f.numpy())
            labels.append(y.numpy())

    X = np.vstack(features)
    y = np.concatenate(labels)

    # subsample
    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    # normalize
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    return X, y
def load_digits_data():
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=5000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_fashion_mnist(n_samples=5000):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def load_rna(path="rna.csv", label_col=None, n_samples=5000):
    import pandas as pd

    df = pd.read_csv(path)

    if label_col and label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        y = np.zeros(len(df))
        X = df.values

    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X = X[idx]
    y = y[idx]

    # 🔥 RNA preprocessing
    X = np.log1p(X)
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)

    return X, y


def run_pipeline(X, y, dataset_name, seeds, lambda_density=0.1):
    output_dir = "output/comparison_seeds"
    os.makedirs(output_dir, exist_ok=True)

    output_dir_images = "output/images_comparison"
    os.makedirs(output_dir_images, exist_ok=True)

    for seed in seeds:
        np.random.seed(seed)

        print(f"\n========================")
        print(f"DATASET: {dataset_name} | SEED: {seed}")
        print(f"========================")
        print(f"Final dataset: {X.shape}, classes: {len(np.unique(y))}")

        # ----------------------------------
        # Precompute
        # ----------------------------------
        rho_high, knn_indices = compute_knn_density(X)
        P = compute_P(X)

        # ----------------------------------
        # Run methods
        # ----------------------------------
        Z_tsne, t_tsne = timed_run("t-SNE", run_tsne, X, seed)

        (Z_density, _), t_density = timed_run(
            f"Density t-SNE (λ={lambda_density})",
            run_density_tsne,
            X, P, knn_indices, rho_high,
            lambda_density=lambda_density,
            seed=seed
        )

        Z_umap, t_umap = timed_run("UMAP", run_umap, X)
        Z_densmap, t_densmap = timed_run("DensMAP", run_densmap, X)

        Z_densne, t_densne = timed_run("densne", run_densne, X)

        # ----------------------------------
        # Collect results
        # ----------------------------------
        results = []

        results.append(
            compute_metrics_for_method(
                "t-SNE", X, Z_tsne, knn_indices, rho_high, t_tsne, y
            )
        )

        results.append(
            compute_metrics_for_method(
                "Density t-SNE", X, Z_density, knn_indices, rho_high, t_density, y
            )
        )

        results.append(
            compute_metrics_for_method(
                "UMAP", X, Z_umap, knn_indices, rho_high, t_umap, y
            )
        )

        results.append(
            compute_metrics_for_method(
                "DensMAP", X, Z_densmap, knn_indices, rho_high, t_densmap, y
            )
        )

        results.append(
            compute_metrics_for_method(
                "densne", X, Z_densne, knn_indices, rho_high, t_densne, y
            )
        )

        # ----------------------------------
        # Save CSV
        # ----------------------------------
        save_results_to_csv(
            os.path.join(output_dir, f"{dataset_name}_seed_{seed}"),
            results
        )

        # ----------------------------------
        # Save visualization
        # ----------------------------------
        save_embeddings_image(
            [Z_tsne, Z_density, Z_umap, Z_densmap, Z_densne],
            ["t-SNE", "Density t-SNE", "UMAP", "DensMAP", "densne"],
            y,
            os.path.join(output_dir_images, f"{dataset_name}_seed_{seed}")
        )

def main():

    np.random.seed(42)
    seeds = [23,234]
    """
    X, y, lengths = load_sbert_shift()
    run_pipeline(X, y, "sbert_shift", seeds, tw_threshold = 0.95)
    """
    X, y = load_tumor_melanoma()
    run_pipeline(X, y, "tumor", seeds, tw_threshold = 0.90)
    X, y, rho_true = load_spiral_density()
    run_pipeline(X, y, "spiral_density",seeds,  tw_threshold = 0.99)
    X, y, rho_true = load_synthetic_density()
    run_pipeline(X, y, "synthetic_density",seeds, tw_threshold = 0.95)
    X, y = load_cifar10_embeddings()
    run_pipeline(X, y, "mnist",seeds, tw_threshold = 0.90)
    X, y = load_pbmc()
    run_pipeline(X, y, "pbmc",seeds,  tw_threshold = 0.85)
    X, y = load_digits_data()
    run_pipeline(X, y, "digits",seeds,  tw_threshold = 0.95)
    X, y = load_fashion_mnist()
    run_pipeline(X, y, "fashion_mnist",seeds,  tw_threshold = 0.95)

if __name__ == "__main__":
    main()