import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
INPUT_DIR = "output/ablation_pareto_seeds"
OUTPUT_DIR = "output/ablation_pareto_seeds/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    "pbmc",
    "mnist",
    "fashion_mnist",
    "digits",
    "synthetic_density",
]

# =========================================================
# STYLE (NeurIPS-friendly)
# =========================================================
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
})

# =========================================================
# LOAD + AGGREGATE
# =========================================================
def load_dataset_aggregate(name):
    pattern = os.path.join(INPUT_DIR, f"{name}_seed*.csv")
    files = glob.glob(pattern)

    if len(files) == 0:
        raise ValueError(f"No files found for dataset: {name}")

    df_all = []

    for f in files:
        df = pd.read_csv(f)
        df_all.append(df)

    df = pd.concat(df_all, ignore_index=True)

    # group by lambda
    grouped = df.groupby("lambda").agg({
        "trustworthiness": ["mean", "std"],
        "density_corr": ["mean", "std"],
    }).reset_index()

    # flatten columns
    grouped.columns = [
        "lambda",
        "tw_mean", "tw_std",
        "dens_mean", "dens_std"
    ]

    # sort left → right
    grouped = grouped.sort_values("dens_mean")

    return grouped


# =========================================================
# MAIN PLOT
# =========================================================
def plot_all():
    plt.figure(figsize=(6, 5))

    colors = plt.cm.tab10.colors
    lambda_text = None

    for i, name in enumerate(DATASETS):
        df = load_dataset_aggregate(name)

        lambdas = df["lambda"].values
        tw_mean = df["tw_mean"].values
        tw_std = df["tw_std"].values
        dens_mean = df["dens_mean"].values
        dens_std = df["dens_std"].values

        # store λ text once
        if lambda_text is None:
            lam_list = [f"{l:g}" for l in lambdas]
            mid = len(lam_list) // 2
            lambda_text = (
                    ", ".join(lam_list[:mid]) + "\n" +
                    ", ".join(lam_list[mid:])
            )

        color = colors[i % len(colors)]

        # =====================================================
        # MEAN LINE
        # =====================================================
        plt.plot(
            dens_mean,
            tw_mean,
            marker="o",
            markersize=3,
            linewidth=2,
            linestyle="--",
            color=color,
            label=name
        )

        # =====================================================
        # STD BAND
        # =====================================================
        plt.fill_between(
            dens_mean,
            tw_mean - tw_std,
            tw_mean + tw_std,
            alpha=0.15,
            color=color
        )

    # =====================================================
    # AXES
    # =====================================================
    plt.xlim(0.3, 1.02)
    plt.ylim(0.7, 1.0)

    plt.xlabel("Density Correlation")
    plt.ylabel("Trustworthiness")

    plt.grid(alpha=0.25)

    # =====================================================
    # LEGEND
    # =====================================================
    legend = plt.legend(
        frameon=False,
        fontsize=9,
        title=f"Datasets\nλ (left→right):\n{lambda_text}"
    )
    legend.get_title().set_fontsize(8)

    plt.tight_layout()

    # save
    out_path = os.path.join(OUTPUT_DIR, "pareto_all_datasets_with_std.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved -> {out_path}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot_all()