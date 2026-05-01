import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "results"

DATASETS = [
    ("thyroid", "Thyroid"),
    ("shuttle", "Shuttle"),
    ("pbmc", "PBMC"),
    ("cifar", "CIFAR"),
    ("synthetic", "Synthetic (Spiral)"),
    ("fashion", "Fashion MNIST"),

]

DETECTORS = {
    "kNN": "AUPRC_kNN",
    "LOF": "AUPRC_LOF",
    "IF": "AUPRC_IF",
    "Cent": "AUPRC_centroid",
}

COLORS = {
    "kNN": "#1f77b4",
    "LOF": "#ff7f0e",
    "IF": "#2ca02c",
    "Cent": "#d62728",
}

K_MODE = "best_k"  # or "mean_k"

# 🔥 changed here
fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey=False)
axes = axes.flatten()

for ax, (dataset, title) in zip(axes, DATASETS):
    path = os.path.join(BASE, f"{dataset}_2d.csv")
    df = pd.read_csv(path)

    df = df[df["method"] == "DR-SNE"].copy()
    df = df[df["param1_name"] == "lambda"].copy()

    df["lambda"] = df["param1_value"].astype(float)
    df["k_density"] = df["param2_value"].astype(float)

    for det_name, metric_col in DETECTORS.items():

        if K_MODE == "best_k":
            grouped = (
                df.groupby(["lambda", "k_density"])[metric_col]
                .agg(["mean", "std"])
                .reset_index()
            )

            best_rows = (
                grouped.sort_values(["lambda", "mean"], ascending=[True, False])
                .groupby("lambda")
                .head(1)
                .sort_values("lambda")
            )

            x = best_rows["lambda"].values
            y = best_rows["mean"].values
            s = best_rows["std"].fillna(0).values

        else:
            grouped = (
                df.groupby("lambda")[metric_col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("lambda")
            )

            x = grouped["lambda"].values
            y = grouped["mean"].values
            s = grouped["std"].fillna(0).values

        # handle lambda=0 for log scale
        x_plot = np.array(x, dtype=float)
        positive = x_plot[x_plot > 0]
        eps = positive.min() / 2 if len(positive) > 0 else 1e-6
        x_plot[x_plot == 0] = eps

        # 🔥 thinner lines
        ax.plot(
            x_plot,
            y,
            color=COLORS[det_name],
            linewidth=1.2,
            label=det_name,
        )

        # 🔥 small points
        ax.scatter(
            x_plot,
            y,
            color=COLORS[det_name],
            s=12,   # ~2–3 pt in paper
            zorder=3,
        )

        # std shading
        ax.fill_between(
            x_plot,
            y - s,
            y + s,
            color=COLORS[det_name],
            alpha=0.12,
            linewidth=0,
            )

        # best point marker (slightly larger)
        best_idx = np.argmax(y)
        ax.scatter(
            x_plot[best_idx],
            y[best_idx],
            color=COLORS[det_name],
            s=28,
            edgecolor="black",
            linewidth=0.3,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$\lambda$")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("AUPRC")

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.88])

plt.savefig("drsne_lambda_curves_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("drsne_lambda_curves_clean.pdf", bbox_inches="tight")

plt.show()