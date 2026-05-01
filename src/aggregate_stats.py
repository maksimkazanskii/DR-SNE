import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "/output/comparison_seeds_best_legacy4"

METHODS = ["t-SNE", "UMAP", "PaCMAP", "DensMAP", "Density t-SNE", "DENSNE"]


def get_dataset_name(filename):
    base = os.path.basename(filename)
    return base.split("_seed_")[0]


# -------------------------------------------------
# Load all CSVs
# -------------------------------------------------
all_files = glob.glob(os.path.join(INPUT_DIR, "*_best_results.csv"))

data = {}

for f in all_files:
    dataset = get_dataset_name(f)

    df = pd.read_csv(f)

    # Expect columns: method, trustworthiness, density_correlation
    for _, row in df.iterrows():
        method = row["method"]

        if method not in METHODS:
            continue

        tw = row["trustworthiness"]
        dens = row["density_correlation"]

        data.setdefault(dataset, {}).setdefault(method, {"tw": [], "dens": []})

        data[dataset][method]["tw"].append(tw)
        data[dataset][method]["dens"].append(dens)


# -------------------------------------------------
# Formatting
# -------------------------------------------------
def format_val(values):
    mean = np.mean(values)
    std = np.std(values)

    return f"{mean:.3f} $\\pm$ {std:.3f}", mean


# -------------------------------------------------
# Print LaTeX
# -------------------------------------------------
for dataset in sorted(data.keys()):

    print(f"\n\\multicolumn{{6}}{{c}}{{\\textbf{{{dataset.replace('_', ' ').title()}}}}} \\\\")
    print("\\cmidrule(lr){1-6}")

    # -------------------------
    # TRUSTWORTHINESS
    # -------------------------
    row_vals = []
    means = []

    for method in METHODS:
        vals = data[dataset].get(method, {}).get("tw", [])
        if len(vals) == 0:
            row_vals.append("--")
            means.append(-1)
            continue

        s, m = format_val(vals)
        row_vals.append(s)
        means.append(m)

    best_idx = int(np.argmax(means))

    for i in range(len(row_vals)):
        if i == best_idx and row_vals[i] != "--":
            row_vals[i] = f"\\textbf{{{row_vals[i]}}}"

    print("TW $\\uparrow$ & " + " & ".join(row_vals) + " \\\\")

    # -------------------------
    # DENSITY
    # -------------------------
    row_vals = []
    means = []

    for method in METHODS:
        vals = data[dataset].get(method, {}).get("dens", [])
        if len(vals) == 0:
            row_vals.append("--")
            means.append(-1)
            continue

        s, m = format_val(vals)
        row_vals.append(s)
        means.append(m)

    best_idx = int(np.argmax(means))

    for i in range(len(row_vals)):
        if i == best_idx and row_vals[i] != "--":
            row_vals[i] = f"\\textbf{{{row_vals[i]}}}"

    print("Density $\\uparrow$ & " + " & ".join(row_vals) + " \\\\")
    print("\n\\addlinespace[4pt]")