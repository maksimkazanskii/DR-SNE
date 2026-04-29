import os
import glob
import pandas as pd


INPUT_DIR = "results"
OUTPUT_DIR = "results_summary"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def summarize_one_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    if "seed" not in df.columns:
        raise ValueError(f"'seed' column not found in {csv_path}")

    # Metrics to aggregate
    metric_cols = [
        c for c in df.columns
        if c.startswith("AUROC_") or c.startswith("AUPRC_")
    ]

    if not metric_cols:
        raise ValueError(f"No AUROC/AUPRC columns found in {csv_path}")

    # Grouping columns = everything except seed and metric columns
    group_cols = [
        c for c in df.columns
        if c not in ["seed"] + metric_cols
    ]

    # Build aggregation dict
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "std"]

    summary = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Flatten multi-index columns
    new_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "":
                new_cols.append(base)
            else:
                new_cols.append(f"{base}_{stat}")
        else:
            new_cols.append(col)

    summary.columns = new_cols

    # Optional: add number of seeds actually present per config
    count_df = (
        df.groupby(group_cols, dropna=False)["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )

    summary = summary.merge(count_df, on=group_cols, how="left")

    # Save
    base_name = os.path.basename(csv_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{name_wo_ext}_summary.csv")

    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    for csv_path in csv_files:
        summarize_one_csv(csv_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()