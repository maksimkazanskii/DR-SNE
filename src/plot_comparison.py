import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

INPUT_DIR = "output/interim"
OUT_DIR = "output/final_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# FIXED METHOD ORDER
# -------------------------------------------------
METHODS = ["t-SNE", "UMAP", "PaCMAP", "DensMAP", "Density t-SNE"]

# -------------------------------------------------
# COLORS (same as your style)
# -------------------------------------------------
colors = [
    # blues
    "#1f77b4", "#4fa3d1", "#a6cee3",

    # oranges
    "#d95f02", "#e6550d", "#fd8d3c",
    "#fdae6b",

    # light reds
    "#fcbba1", "#fee0d2",

    # muted / soft tones
    "#fdd49e", "#f6b26b",

    # muted blues (replace browns)
    "#1f4e79",
    "#4c7fa1",
    "#9fbfd9",
    "#2b5d7d",
    "#7fa6c9",
    "#c6d9ec",
]
cmap = ListedColormap(colors)


# -------------------------------------------------
# LOAD DATA (FIXED)
# -------------------------------------------------
def load_embeddings(dataset):
    # ✅ FIX: exact dataset match instead of substring
    files = [f for f in os.listdir(INPUT_DIR) if f.startswith(dataset + "_")]

    data_dict = {}

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        d = np.load(path, allow_pickle=True)

        method = str(d["method"])

        if method not in METHODS:
            continue

        Z = d["Z"]
        y = d["y"]

        # label fix
        if y.dtype.type is np.str_ or y.dtype == object:
            _, y = np.unique(y, return_inverse=True)

        rho = d["rho_high"] if "rho_high" in d else None

        data_dict[method] = {
            "Z": Z,
            "y": y.astype(int),
            "rho": rho
        }

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

            # -----------------------------------
            # density sorting (unchanged)
            # -----------------------------------
            if rho is not None:
                order = np.argsort(rho)
                Z = Z[order]
                y = y[order]

            # -----------------------------------
            # spiral styling (unchanged)
            # -----------------------------------
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

            # row labels
            if col == 0:
                ax.set_ylabel(method, fontsize=10)

            # column titles
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

    # FIGURE 1 (real-world + biology)
    plot_grid(
        datasets=["pbmc", "fashion_mnist", "tumor"],
        filename="comparison_main.png"
    )

    # FIGURE 2 (benchmarks + synthetic)
    plot_grid(
        datasets=["digits", "mnist", "spiral_density"],
        filename="comparison_app.png"
    )