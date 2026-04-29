from sklearn.datasets import load_digits, fetch_openml
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from comparison import (
    compute_knn_density,
    compute_P,
    save_best_results_to_csv,
    tune_density_tsne,
    tune_1d_method
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scanpy as sc
import numpy as np
import random
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





def save_interim_results(
        Z_list,
        titles,
        y,
        results,
        dataset_name,
        seed,
        output_dir="output/interim"
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for Z, title, res in zip(Z_list, titles, results):

        method = res["name"]
        param_name = res["param_name"]
        param_value = res["param_value"]

        filename = (
            f"{dataset_name}_seed_{seed}_"
            f"{method}_{param_name}_{param_value}.npz"
        )

        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(
            filepath,
            Z=Z,
            y=y,
            method=method,
            param_name=param_name,
            param_value=param_value,
            trustworthiness=res["trustworthiness"],
            continuity=res["continuity"],
            density_corr=res["density_corr"],
            silhouette=res["silhouette"],
            stress=res["stress"],
            time=res["time"]
        )

        print(f"Saved interim: {filepath}")

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

def load_synthetic_density(n_samples=5000, n_clusters=4, dim=50, seed=42):

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(seed)

    X_list = []
    y_list = []
    rho_true = []

    samples_per_cluster = n_samples // n_clusters
    variances = np.linspace(0.2, 2.0, n_clusters)

    for i, var in enumerate(variances):

        mean = rng.randn(dim) * 5
        cov = np.eye(dim) * var

        X_cluster = rng.multivariate_normal(mean, cov, size=samples_per_cluster)

        density = 1.0 / (var ** (dim / 2))

        X_list.append(X_cluster)
        y_list.append(np.full(samples_per_cluster, i))
        rho_true.append(np.full(samples_per_cluster, density))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    rho_true = np.concatenate(rho_true)

    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]
    rho_true = rho_true[idx]

    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=seed, svd_solver="randomized").fit_transform(X)

    return X, y, rho_true

def load_tumor_melanoma(path="data/melanoma.h5ad"):
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

def load_digits_data():
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y


def load_mnist(n_samples=5000, seed=42):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)

    return X, y

def load_fashion_mnist(n_samples=5000, seed=42):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)

    return X, y


def load_rna(path="rna.csv", label_col=None, n_samples=5000, seed=42):
    import pandas as pd

    df = pd.read_csv(path)

    if label_col and label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        y = np.zeros(len(df))
        X = df.values

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    X = X[idx]
    y = y[idx]

    X = np.log1p(X)
    X = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=seed, svd_solver="randomized").fit_transform(X)

    return X, y


def run_pipeline_tuned(X, y, dataset_name, seeds, tw_threshold):
    output_dir = "output/comparison_seeds_best"
    os.makedirs(output_dir, exist_ok=True)

    output_dir_images = "output/images_comparison_best"
    os.makedirs(output_dir_images, exist_ok=True)

    for seed in seeds:
        set_global_seed(seed)

        print(f"\n========================")
        print(f"DATASET: {dataset_name} | SEED: {seed}")
        print(f"========================")
        print(f"Final dataset: {X.shape}, classes: {len(np.unique(y))}")
        print(f"Trustworthiness threshold: {tw_threshold}")

        rho_high, knn_indices = compute_knn_density(X)
        P = compute_P(X)

        best_results = []
        best_embeddings = []
        best_titles = []


        best_density, _ = tune_density_tsne(
            X=X,
            y=y,
            P=P,
            knn_indices=knn_indices,
            rho_high=rho_high,
            seed=seed,
            tw_threshold=tw_threshold
        )
        if best_density is not None:
            best_results.append(best_density)
            best_embeddings.append(best_density["Z"])
            best_titles.append(f"DR-SNE\nλ={best_density['param_value']}")


        best_densne, _ = tune_1d_method(
                method_name="DenSNE",
                X=X,
                y=y,
                knn_indices=knn_indices,
                rho_high=rho_high,
                seed=seed,
                tw_threshold=tw_threshold,
                param_name="dens_lambda",
                param_values=[0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
            )

        if best_densne is not None:
            best_results.append(best_densne)
            best_embeddings.append(best_densne["Z"])
            best_titles.append(f"DenSNE\nλ={best_densne['param_value']}")

        # t-SNE: tune perplexity
        best_tsne, _ = tune_1d_method(
            method_name="t-SNE",
            X=X,
            y=y,
            knn_indices=knn_indices,
            rho_high=rho_high,
            seed=seed,
            tw_threshold=tw_threshold,
            param_name="perplexity",
            param_values=[5, 10, 20, 30, 50, 75, 100]
        )
        if best_tsne is not None:
            best_results.append(best_tsne)
            best_embeddings.append(best_tsne["Z"])
            best_titles.append(f"t-SNE\nperp={best_tsne['param_value']}")

        best_pacmap, _ = tune_1d_method(
            method_name="PaCMAP",
            X=X,
            y=y,
            knn_indices=knn_indices,
            rho_high=rho_high,
            seed=seed,
            tw_threshold=tw_threshold,
            param_name="n_neighbors",
            param_values=[5, 10, 15, 30, 50]
        )

        if best_pacmap is not None:
            best_results.append(best_pacmap)
            best_embeddings.append(best_pacmap["Z"])
            best_titles.append(f"PaCMAP\nk={best_pacmap['param_value']}")
        # DR-SNE
        #
        # : tune lambda_density

        # UMAP: tune n_neighbors
        best_umap, _ = tune_1d_method(
            method_name="UMAP",
            X=X,
            y=y,
            knn_indices=knn_indices,
            rho_high=rho_high,
            seed=seed,
            tw_threshold=tw_threshold,
            param_name="n_neighbors",
            param_values=[5, 10, 15, 30, 50, 100]
        )
        if best_umap is not None:
            best_results.append(best_umap)
            best_embeddings.append(best_umap["Z"])
            best_titles.append(f"UMAP\nk={best_umap['param_value']}")


        # DensMAP: tune n_neighbors
        best_densmap, _ = tune_1d_method(
            method_name="DensMAP",
            X=X,
            y=y,
            knn_indices=knn_indices,
            rho_high=rho_high,
            seed=seed,
            tw_threshold=tw_threshold,
            param_name="dens_lambda",
            param_values=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        )
        if best_densmap is not None:
            best_results.append(best_densmap)
            best_embeddings.append(best_densmap["Z"])
            best_titles.append(f"DensMAP\nk={best_densmap['param_value']}")


        # save csv
        stripped_results = []
        for r in best_results:
            rr = dict(r)
            rr.pop("Z", None)
            stripped_results.append(rr)

        save_best_results_to_csv(
            os.path.join(output_dir, f"{dataset_name}_seed_{seed}"),
            stripped_results
        )

        # save image
        #if len(best_embeddings) > 0:
        #    save_embeddings_image(
        #        best_embeddings,
        #        best_titles,
        #        y,
        #        os.path.join(output_dir_images, f"{dataset_name}_seed_{seed}")
        #    )
        # save embeddings for later plotting
        if len(best_embeddings) > 0:
            save_interim_results(
                best_embeddings,
                best_titles,
                y,
                best_results,
                dataset_name,
                seed
            )

def main():

    seeds = [23,234,234345]

    X, y = load_tumor_melanoma()
    run_pipeline_tuned(X, y, "tumor", seeds,          tw_threshold = 0.90)
    X, y, rho_true = load_spiral_density()
    run_pipeline_tuned(X, y, "spiral_density",seeds,  tw_threshold = 0.99)
    X, y, rho_true = load_synthetic_density()
    run_pipeline_tuned(X, y, "mnist",seeds,           tw_threshold = 0.88)
    X, y = load_pbmc()
    run_pipeline_tuned(X, y, "pbmc",seeds,            tw_threshold = 0.85)
    X, y = load_digits_data()
    run_pipeline_tuned(X, y, "digits",seeds,          tw_threshold = 0.95)
    X, y = load_fashion_mnist()
    run_pipeline_tuned(X, y, "fashion_mnist",seeds,  tw_threshold = 0.95)

if __name__ == "__main__":
    main()