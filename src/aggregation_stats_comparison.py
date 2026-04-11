import os
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
INPUT_DIR = "output/comparison_seeds_best"
OUTPUT_DIR = "output/aggregated_methods"

DATASETS = [
    "tumor",
    "spiral_density",
    "mnist",
    "pbmc",
    "digits",
    "fashion_mnist",
]

SEEDS = [23, 234, 34532]


# =========================================================
# CORE AGGREGATION (METHOD LEVEL)
# =========================================================
def aggregate_dataset(dataset_name, seeds):
    print(f"\n==============================")
    print(f"Aggregating dataset: {dataset_name}")
    print(f"==============================")

    all_dfs = []

    for seed in seeds:
        filename = os.path.join(
            INPUT_DIR,
            f"{dataset_name}_seed_{seed}_best_results.csv"
        )

        if not os.path.exists(filename):
            print(f"⚠️ Missing file: {filename}")
            continue

        df = pd.read_csv(filename)
        df["seed"] = seed
        all_dfs.append(df)

    if len(all_dfs) == 0:
        print(f"❌ No data found for {dataset_name}")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # 🔥 IMPORTANT: group ONLY by method
    group_cols = ["method"]

    metrics = [
        "trustworthiness",
        "continuity",
        "density_corr",
        "silhouette",
        "stress",
        "time_sec"
    ]

    # -------------------------
    # MEAN + STD across seeds + params
    # -------------------------
    df_mean = df_all.groupby(group_cols)[metrics].mean().reset_index()
    df_std = df_all.groupby(group_cols)[metrics].std().reset_index()

    df_mean = df_mean.rename(columns={m: f"{m}_mean" for m in metrics})
    df_std = df_std.rename(columns={m: f"{m}_std" for m in metrics})

    df_final = pd.merge(df_mean, df_std, on=group_cols)

    # -------------------------
    # Optional ranking score
    # -------------------------
    df_final["score"] = (
            df_final["density_corr_mean"]
            - 0.1 * df_final["stress_mean"]
    )

    # Sort by main metric
    df_final = df_final.sort_values(
        "density_corr_mean",
        ascending=False
    )

    # -------------------------
    # SAVE
    # -------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_methods_only.csv"
    )

    df_final.to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")
    print(df_final)


# =========================================================
# MAIN
# =========================================================
def main():
    for dataset in DATASETS:
        aggregate_dataset(dataset, SEEDS)


if __name__ == "__main__":
    main()