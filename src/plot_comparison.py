import os
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

INPUT_DIR = "output/interim"
OUT_DIR = "output/final_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# FIXED METHOD ORDER (DISPLAY NAMES)
# -------------------------------------------------
METHODS = ["t-SNE", "UMAP", "PaCMAP", "DensMAP", "DenSNE", "DR-SNE"]

# -------------------------------------------------
# COLORS
# -------------------------------------------------
colors = [
    "#1f77b4", "#4fa3d1", "#a6cee3",
    "#d95f02", "#e6550d", "#fd8d3c",
    "#fdae6b",
    "#fcbba1", "#fee0d2",
    "#fdd49e", "#f6b26b",
    "#1f4e79",
    "#4c7fa1",
    "#9fbfd9",
    "#2b5d7d",
    "#7fa6c9",
    "#c6d9ec",
]
cmap = ListedColormap(colors)


# -------------------------------------------------
# LOAD DATA WITH PRIORITY LOGIC
# -------------------------------------------------
def load_embeddings(dataset):
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith(dataset + "_")]

    raw_data = {}

    # -----------------------
    # LOAD ALL METHODS FIRST
    # -----------------------
    for f in files:
        path = os.path.join(INPUT_DIR, f)
        d = np.load(path, allow_pickle=True)

        method = str(d["method"])

        Z = d["Z"]
        y = d["y"]

        if y.dtype.type is np.str_ or y.dtype == object:
            _, y = np.unique(y, return_inverse=True)

        rho = d["rho_high"] if "rho_high" in d else None

        raw_data[method] = {
            "Z": Z,
            "y": y.astype(int),
            "rho": rho
        }

    # -----------------------
    # RESOLVE METHOD NAMES
    # -----------------------
    data_dict = {}

    has_dr = ("DR-SNE" in raw_data) or ("DRSNE" in raw_data)
    has_density = ("Density t-SNE" in raw_data)

    for method in METHODS:

        if method == "DR-SNE":
            if "DR-SNE" in raw_data:
                key = "DR-SNE"
            elif "DRSNE" in raw_data:
                key = "DRSNE"
            elif has_density:
                key = "Density t-SNE"
            else:
                continue
        else:
            key = method

        if key not in raw_data:
            continue

        data_dict[method] = raw_data[key]

    return data_dict


# -------------------------------------------------
# PLOTTING
# -------------------------------------------------
def plot_grid(datasets, filename):

    n_rows = len(METHODS)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0}
    )

    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, dataset in enumerate(datasets):

        data_dict = load_embeddings(dataset)

        for row, method in enumerate(METHODS):

            ax = axes[row, col]

            if method not in data_dict:
                ax.axis("off")
                continue

            data = data_dict[method]

            Z = data["Z"]
            y = data["y"]
            rho = data["rho"]

            # plot low-density first
            if rho is not None:
                order = np.argsort(rho)
                Z = Z[order]
                y = y[order]

            # styling
            if "spiral" in dataset:
                s_val = 2
                alpha_val = 0.6
            else:
                s_val = 3
                alpha_val = 0.35

            ax.scatter(
                Z[:, 0],
                Z[:, 1],
                c=y % len(colors),
                cmap=cmap,
                s=s_val,
                alpha=alpha_val
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("0.85")

            if col == 0:
                ax.set_ylabel(method, fontsize=10)

            if row == 0:
                ax.set_title(dataset, fontsize=12)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    save_path = os.path.join(OUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved -> {save_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    plot_grid(
        datasets=["pbmc", "fashion_mnist", "shuttle"],
        filename="comparison_main.png"
    )

    plot_grid(
        datasets=["digits", "mnist", "spiral_density"],
        filename="comparison_app.png"
    )