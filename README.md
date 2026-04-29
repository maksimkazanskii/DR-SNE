# DR-SNE: Density-Regularized Stochastic Neighbor Embedding

**DR-SNE** is a dimensionality reduction method that explicitly preserves **relative density structure** alongside local neighborhood relationships. It extends the stochastic neighbor embedding (SNE / t-SNE) framework by introducing a principled density alignment objective.

---

## Overview

Dimensionality reduction methods such as t-SNE focus on preserving **local geometry**, but often distort the **distribution of probability mass**, leading to misleading embeddings.

DR-SNE addresses this limitation by reformulating dimensionality reduction as the **joint alignment of**:

- **Conditional structure** (local neighborhoods)
- **Marginal density** (distribution of data across space)

This is achieved by augmenting the standard t-SNE objective with a **density regularization term**.

---

## Key Idea

Standard t-SNE optimizes:

L_KL

DR-SNE introduces an additional term:

L = L_KL + λ · L_dens

where:

- **L_KL (Kullback–Leibler divergence)**  
  Measures how well the embedding preserves **local neighborhood structure**.  
  It compares pairwise similarities in the original space vs the embedding and penalizes when nearby points are not kept close.

- **L_dens (density loss)**  
  Measures how well the embedding preserves **relative density**.  
  It penalizes differences between local density estimates in the original space and the embedding, ensuring that dense regions remain dense and sparse regions remain sparse.

- **λ (lambda)**  
  Controls the trade-off:
  - small λ → behaves like standard t-SNE (focus on structure)
  - large λ → stronger density preservation
## ✨ Features

- Preserves **relative density variations**
- Provides **interpretable trade-off** via a single parameter ($\lambda$)
- Improves embeddings in **density-driven** and **hybrid datasets**
- Enhances downstream tasks such as **anomaly detection**
- Compatible with standard t-SNE acceleration techniques (e.g., Barnes-Hut, FIt-SNE)

---

## 📊 When to Use DR-SNE

DR-SNE is particularly effective in:

- **Density-driven data**
    - anomaly detection
    - rare-event analysis

- **Hybrid structure datasets**
    - single-cell RNA-seq (PBMC, tumor)
    - datasets with both clusters and density variation

Less critical for:
- purely clustered data (e.g., MNIST classification visualization)
- purely geometric manifold tasks

---

##  Installation

```bash
git clone https://github.com/your-repo/dr-sne
cd dr-sne
pip install -r requirements.txt
```

##  How to Run (API Usage)

### 1. DR-SNE (main method)

```python
from DRSNE.drsne import drsne

Z = drsne(
  X,
  n_components=2,
  perplexity=50.0,
  lambda_density=0.01,
  k_density=50,
  n_iter=1000,
  warmup=50,
  lr=2.0,
  seed=42,
  verbose=True
)

```

## ⚠️ Important

**Current implementation has $\mathcal{O}(n^2)$ time complexity and does not include Barnes–Hut or other acceleration techniques.** It is therefore currently not optimized for large-scale datasets. For large $n$, consider subsampling or integrating existing accelerated t-SNE methods.
**An optimized, scalable version is currently in development and will be released soon.**
---
## Structure

## 📂 Source Code Structure (`src/`)

```text
src/
├── DRSNE/                     # Main DR-SNE implementation
│   └── drsne.py
│
├── DENSNE/                   # Baseline: DenSNE
│   └── densne.py
│
├── ablations/                # Controlled ablation experiments
│   ├── ablation_studies_lambda.py   # Sweep over λ (density regularization)
│   ├── ablation_studies_nn.py       # Sweep over k (neighborhood size)
│   ├── ablation_studies_pca.py      # Sweep over PCA dimensionality
│   ├── ablation_pareto.py           # Pareto trade-off computation (TW vs density)
│   ├── ablation_studies_*_plot.py   # Visualization of ablation results
│   ├── ablation_studies_table_lambda.py  # Table generation (λ study)
│   └── pareto_ablation_plot.py      # Final Pareto front visualization
│
├── ood/                      # Out-of-distribution / anomaly detection evaluation
│   ├── pipeline.py           # Main evaluation pipeline (embeddings → anomaly scores)
│   ├── datasets.py           # Dataset loaders (real + synthetic anomaly datasets)
│   ├── dataset_characteristics.py   # Dataset statistics and diagnostics
│   ├── combine_seeds.py      # Aggregation across multiple runs
│   ├── best_choose.py        # Selection of best hyperparameters
│   ├── check_files.py        # Sanity checks for experiment outputs
│   └── plot_ablations.py     # Visualization of OOD performance
│
├── timing_experiments/       # Runtime analysis
│   ├── timing_density.py
│   └── plot_timing.py
│
├── legacy/                  # Older pipelines (kept for reproducibility)
│   ├── anomaly_detection/
│   ├── comparison.py
│   ├── dtsne_legacy.py
│   └── run_all_datasets.py
│
├── comparison.py             # Core benchmarking pipeline
├── run_all_datasets.py       # Full experiment orchestration
├── density_tsne.py           # Density-regularized t-SNE training loop
├── density_tsne_old.py       # Legacy version
├── aggregate_stats.py        # LaTeX table generation
├── aggregation_stats_comparison.py  # Aggregation across seeds
├── plot_comparison.py        # Final comparison figures
├── plot_ablation_results.py  # Trade-off visualization

# How to Run

## 1. Run DR-SNE on your data

```python
from DRSNE.drsne import drsne

Z = drsne(
    X,
    n_components=2,
    perplexity=50.0,
    lambda_density=0.01,
    k_density=50,
    n_iter=1000,
    warmup=50,
    lr=2.0,
    seed=42,
    verbose=True
)
```

## How to run experiments (reproducibility)
### Parameters

| Parameter | Description |
|---|---|
| `X` | Input data with shape `(n_samples, n_features)` |
| `n_components` | Output embedding dimension, usually `2` |
| `perplexity` | Controls neighborhood size in the t-SNE similarity distribution |
| `lambda_density` | Strength of density preservation |
| `k_density` | Number of neighbors used for density estimation |
| `n_iter` | Number of optimization iterations |
| `warmup` | Initial iterations without density regularization |
| `lr` | Learning rate for the Adam optimizer |
| `seed` | Random seed for reproducibility |
| `verbose` | If `True`, prints training progress |

### Typical settings

| Use case | Suggested settings |
|---|---|
| Standard t-SNE behavior | `lambda_density = 0.0` |
| Balanced behavior | `lambda_density = 1e-3` to `1e-2` |
| Density-focused behavior | `lambda_density = 1e-2` to `1e-1`, `k_density = 100` to `300` |

### 2. Run full experiments

```bash
python src/run_all_datasets.py
```

This runs DR-SNE and baseline methods across multiple datasets.

### 3. Run comparison experiments

```bash
python src/comparison.py
```

This runs method comparisons and computes metrics such as trustworthiness, density correlation, stress, and runtime.

### 4. Run ablation studies

Lambda sweep:

```bash
python src/ablations/ablation_studies_lambda.py
```

Neighborhood-size sweep:

```bash
python src/ablations/ablation_studies_nn.py
```

PCA-dimensionality sweep:

```bash
python src/ablations/ablation_studies_pca.py
```

Pareto trade-off analysis:

```bash
python src/ablations/ablation_pareto.py
```

### 5. Run anomaly detection / OOD experiments

```bash
python src/ood/pipeline.py
```

This evaluates embeddings using anomaly detection scores such as kNN, LOF, Isolation Forest, centroid distance, and related metrics.

### 6. Aggregate results

```bash
python src/aggregate_stats.py
python src/aggregation_stats_comparison.py
```

These scripts aggregate results across datasets, methods, and seeds.

### 7. Plot results

```bash
python src/plot_comparison.py
python src/plot_ablation_results.py
```

These scripts generate comparison and ablation figures.

### 8. Run timing experiments

```bash
python src/timing_experiments/timing_density.py
python src/timing_experiments/plot_timing.py
```

