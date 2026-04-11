import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
INPUT_DIR = "output/ablation_pareto"
OUTPUT_DIR = "output/ablation_pareto/plots"
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
# LOAD CSV
# =========================================================
def load_dataset_csv(name):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    df = pd.read_csv(path)

    # sort left → right (density)
    df = df.sort_values("density_corr")

    return df


# =========================================================
# MAIN PLOT
# =========================================================
def plot_all():
    plt.figure(figsize=(6, 5))

    colors = plt.cm.tab10.colors
    lambda_text = None

    for i, name in enumerate(DATASETS):
        df = load_dataset_csv(name)

        lambdas = df["lambda"].values
        tw = df["trustworthiness"].values
        dens = df["density_corr"].values

        # store λ once
        if lambda_text is None:
            lam_list = [f"{l:g}" for l in lambdas]

            # 🔥 split into two rows
            mid = len(lam_list) // 2
            lambda_text = (
                    ", ".join(lam_list[:mid]) + "\n" +
                    ", ".join(lam_list[mid:])
            )

        color = colors[i % len(colors)]

        plt.plot(
            dens,
            tw,
            marker="o",
            markersize=2.5,
            linewidth=1.8,
            linestyle="--",   # 🔥 dashed lines
            color=color,
            label=name
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
    out_path = os.path.join(OUTPUT_DIR, "pareto_all_datasets_neurips.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved -> {out_path}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot_all()