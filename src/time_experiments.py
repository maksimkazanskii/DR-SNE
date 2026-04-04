import torch
import time
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors # Assuming this function is available as a library
import os

import torch
import numpy as np
from sklearn.decomposition import PCA

import torch
import numpy as np
from sklearn.decomposition import PCA

def run_density_tsne(
        X,
        P,
        knn_indices,
        rho_high,
        n_iter=300,
        warmup=50,
        lr=2.0,
        lambda_density=0.01,
):
    # =========================================================
    # INIT
    # =========================================================
    Z_init = PCA(n_components=2).fit_transform(X)
    Z = torch.tensor(Z_init, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([Z], lr=lr)

    rho_high_t = torch.tensor(rho_high, dtype=torch.float32)
    log_rho_high = torch.log(rho_high_t + 1e-8)

    knn_indices_t = torch.tensor(knn_indices, dtype=torch.long)
    neighbors = knn_indices_t[:, 1:]  # precompute once

    # Ensure that all neighbors indices are within bounds of P (i.e., [0, 9] for 10 neighbors)
    max_index = P.size(1) - 1
    if neighbors.max() >= P.size(1):
        print(f"Warning: Some neighbor indices are out of bounds. Clipping to valid range [0, {max_index}].")
        neighbors = torch.clamp(neighbors, 0, max_index)  # Clamp indices to be within [0, 9]

    # Precompute P_neighbors once
    P_neighbors = P.gather(1, neighbors)

    # Precompute warmup P once
    if warmup > 0:
        P_eff = P * 4.0
        P_eff = P_eff / P_eff.sum()

    history = {"total": [], "kl": [], "density": []}

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for it in range(n_iter):
        optimizer.zero_grad()

        # =========================================
        # FULL PAIRWISE DISTANCE
        # =========================================
        dist_sq = torch.cdist(Z, Z, p=2) ** 2
        dist_sq = dist_sq + 1e-8

        # =========================================
        # Q MATRIX
        # =========================================
        Q = 1.0 / (1.0 + dist_sq)
        Q.fill_diagonal_(0)
        Q /= Q.sum()  # in-place normalization

        # =========================================
        # SPARSE KL (inlined, no function call)
        # =========================================
        Q_neighbors = Q.gather(1, neighbors)

        kl = (
                     P_neighbors * (
                     torch.log(P_neighbors + 1e-8) - torch.log(Q_neighbors + 1e-8)
             )
             ).sum() / Z.shape[0]

        # =========================================
        # DENSITY LOSS (inlined, optimized)
        # =========================================
        Z_center = Z.unsqueeze(1)
        Z_neighbors = Z[neighbors]

        dists_sq = ((Z_center - Z_neighbors) ** 2).sum(dim=2)
        volume = torch.sqrt(dists_sq + 1e-8).sum(dim=1)

        rho_low = neighbors.shape[1] / volume
        rho_low = rho_low / (rho_low.mean() + 1e-8)

        density_loss = (
                (log_rho_high - torch.log(rho_low + 1e-8)) ** 2
        ).mean()

        # =========================================
        # LOSS
        # =========================================
        if it < warmup:
            # Extract the relevant part of Q for the neighbors
            Q_neighbors_subset = Q.gather(1, neighbors)
            loss = (
                           P_eff * (torch.log(P_eff + 1e-8) - torch.log(Q_neighbors_subset + 1e-8))
                   ).sum() / Z.shape[0]
        else:
            loss = kl + lambda_density * density_loss

        # =========================================
        # OPTIMIZATION
        # =========================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Z], 1.0)
        optimizer.step()

        # =========================================
        # LOGGING
        # =========================================
        history["total"].append(loss.item())
        history["kl"].append(kl.item())
        history["density"].append(density_loss.item())

        if it % 50 == 0:
            print(
                f"[Density t-SNE] Iter {it}: "
                f"KL={kl.item():.6f}, Density={density_loss.item():.6f}"
            )

    return Z.detach().numpy(), history


# Function to load melanoma tumor dataset
def load_tumor_melanoma(path="data/melanoma.h5ad"):
    print("Loading melanoma tumor dataset...")
    try:
        adata = sc.read_h5ad(path)
    except FileNotFoundError:
        print(f"File {path} not found. Using fallback dataset (pbmc3k).")
        adata = sc.datasets.pbmc3k()

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Retrieve labels
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

    # Remove invalid labels
    bad_labels = {'unknown', 'NA', 'None', ''}
    y = np.array([label for label in y if label not in bad_labels])
    X = X[:len(y)]
    _, y_encoded = np.unique(y, return_inverse=True)

    return X, y_encoded

# Function to load Fashion-MNIST dataset with a specified number of samples
def load_fashion_mnist(n_samples=5000):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    idx = np.random.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y = y[idx]
    X = StandardScaler().fit_transform(X)
    return X, y

# Function to run timing experiments with different sample sizes for Fashion-MNIST and feature sizes for tumor dataset
def timing_experiment():
    # Sample sizes for Fashion-MNIST and feature sizes for tumor dataset
    fashion_samples = [500, 1000, 2000, 4000, 8000, 16000]
    tumor_features = [32738 // (2 ** i) for i in range(6)]  # Dividing by powers of 2 until 10 features

    # Ensure the output directory exists
    os.makedirs('output/timing/', exist_ok=True)

    # Create an empty DataFrame to store results
    timing_df = pd.DataFrame(columns=["dataset", "sample_size", "num_features", "time", "seed"])

    # Experiment for Fashion-MNIST
    for n_samples in fashion_samples:
        X_fashion, y_fashion = load_fashion_mnist(n_samples=n_samples)

        # Prepare KNN indices and P matrix using NearestNeighbors
        knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn.fit(X_fashion)
        knn_indices = knn.kneighbors(X_fashion)[1]  # Get the indices of the 10 nearest neighbors
        distances, _ = knn.kneighbors(X_fashion)
        P = np.exp(-distances ** 2)  # Convert distances to probabilities using Gaussian kernel
        rho_high = 1.0  # Example high density parameter

        # Run t-SNE for 5 different random seeds
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            start_time = time.time()

            # Run Density t-SNE
            Z, _ = run_density_tsne(X_fashion, torch.tensor(P), knn_indices, rho_high, n_iter=300, warmup=50, lr=2.0, lambda_density=0.01)

            elapsed_time = time.time() - start_time
            timing_df = timing_df.append({
                "dataset": "fashion_mnist",
                "sample_size": n_samples,
                "num_features": X_fashion.shape[1],
                "time": elapsed_time,
                "seed": seed
            }, ignore_index=True)

    # Experiment for Tumor Dataset
    X_tumor, y_tumor = load_tumor_melanoma(path="data/melanoma.h5ad")

    # Experiment for different feature sizes in the tumor dataset
    for num_features in tumor_features:
        X_tumor_reduced = X_tumor[:, :num_features]  # Reducing features

        # Prepare KNN indices and P matrix using NearestNeighbors
        knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn.fit(X_tumor_reduced)
        knn_indices = knn.kneighbors(X_tumor_reduced)[1]
        distances, _ = knn.kneighbors(X_tumor_reduced)
        P = np.exp(-distances ** 2)
        rho_high = 1.0  # Example high density parameter

        # Run t-SNE for 5 different random seeds
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            start_time = time.time()

            # Run Density t-SNE
            Z, _ = run_density_tsne(X_tumor_reduced, torch.tensor(P), knn_indices, rho_high, n_iter=300, warmup=50, lr=2.0, lambda_density=0.01)

            elapsed_time = time.time() - start_time
            timing_df = timing_df.append({
                "dataset": "tumor_melanoma",
                "sample_size": X_tumor_reduced.shape[0],
                "num_features": num_features,
                "time": elapsed_time,
                "seed": seed
            }, ignore_index=True)

    # Save the timing results to CSV
    timing_df.to_csv('output/timing/n.csv', index=False)

    # Save the feature sizes for tumor dataset
    features_df = pd.DataFrame({"num_features": tumor_features})
    features_df.to_csv('output/timing/features.csv', index=False)

# Run the timing experiment
timing_experiment()