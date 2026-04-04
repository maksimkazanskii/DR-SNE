import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# =========================================================
# LOAD CSV
# =========================================================
def load_results(csv_path):
    results = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "lambda": float(row["lambda"]),
                "trustworthiness": float(row["trustworthiness"]),
                "density_corr": float(row["density_corr"]),
            })

    return results


# =========================================================
# MAIN PLOT
# =========================================================
def plot_combined_tradeoff():

    os.makedirs("output/images", exist_ok=True)

    datasets = [
        "digits",
        "mnist",
        "fashion_mnist",
        "rna"
    ]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(7, 6))

    for dataset, color in zip(datasets, colors):

        csv_path = f"output/{dataset}_ablation.csv"

        if not os.path.exists(csv_path):
            print(f"Skipping {dataset} (no CSV)")
            continue

        print(f"Processing {dataset}...")

        results = load_results(csv_path)

        tw = np.array([r["trustworthiness"] for r in results])
        dens = np.array([r["density_corr"] for r in results])
        lambdas = [r["lambda"] for r in results]

        # Plot curve
        plt.plot(tw, dens, marker='o', color=color, label=dataset)

        # Annotate λ (optional but useful)
        for i, lam in enumerate(lambdas):
            label = "0" if lam == 0 else f"{lam:.0e}"
            plt.text(tw[i], dens[i], label, fontsize=6, color=color)

    plt.xlabel("Trustworthiness ↑")
    plt.ylabel("Density Correlation ↑")
    plt.title("Trade-off across datasets")

    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    filename = "output/images/tradeoff_combined.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"\nSaved combined figure → {filename}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot_combined_tradeoff()