import os
import pandas as pd
import numpy as np

INPUT_DIR = "results_summary"
OUTPUT_DIR = "results_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

AUPRC_METRICS = [
    "AUPRC_kNN",
    "AUPRC_LOF",
    "AUPRC_IF",
    "AUPRC_centroid",
    "AUPRC_linear",
]


def format_params(row):
    p1n = row.get("param1_name")
    p1v = row.get("param1_value")
    p2n = row.get("param2_name")
    p2v = row.get("param2_value")

    parts = []

    if pd.notna(p1n) and str(p1n) != "":
        parts.append(f"{p1n}={p1v}")
    if pd.notna(p2n) and str(p2n) != "":
        parts.append(f"{p2n}={p2v}")

    return ", ".join(parts) if parts else ""


def clean_std(row, std_col):
    if "n_seeds" in row and row["n_seeds"] == 1:
        return None
    val = row.get(std_col)
    return None if pd.isna(val) else val


# =========================================================
# OLD BEHAVIOR (PER-METRIC BEST)
# =========================================================

def choose_best_row(df_method, metric):
    mean_col = f"{metric}_mean"

    if mean_col not in df_method.columns:
        return None

    best_idx = df_method[mean_col].idxmax()
    return df_method.loc[best_idx]


# =========================================================
# NEW FAIR SELECTION (AUPRC_IF ONLY)
# =========================================================

def choose_fair_row(df_method):
    mean_col = "AUPRC_IF_mean"

    if mean_col not in df_method.columns:
        return None

    best_idx = df_method[mean_col].idxmax()
    return df_method.loc[best_idx]


# =========================================================
# PROCESS
# =========================================================

def process_file(filepath):
    df = pd.read_csv(filepath)

    filename = os.path.splitext(os.path.basename(filepath))[0]

    for dataset in df["dataset"].dropna().unique():
        df_dataset = df[df["dataset"] == dataset].copy()

        best_rows = []
        fair_rows = []

        for method in df_dataset["method"].dropna().unique():
            df_method = df_dataset[df_dataset["method"] == method].copy()

            # =========================
            # ORIGINAL TABLE
            # =========================
            best_out = {"method": method}

            for metric in AUPRC_METRICS:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"

                if mean_col not in df_method.columns:
                    best_out[f"{metric}_mean"] = None
                    best_out[f"{metric}_std"] = None
                    best_out[f"{metric}_best_params"] = None
                    continue

                best = choose_best_row(df_method, metric)

                best_out[f"{metric}_mean"] = best[mean_col]
                best_out[f"{metric}_std"] = clean_std(best, std_col)
                best_out[f"{metric}_best_params"] = format_params(best)

            best_rows.append(best_out)

            # =========================
            # FAIR TABLE (NeurIPS)
            # =========================
            fair = choose_fair_row(df_method)

            if fair is None:
                continue

            fair_out = {"method": method}

            for metric in AUPRC_METRICS:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"

                if mean_col not in df_method.columns:
                    fair_out[f"{metric}_mean"] = None
                    fair_out[f"{metric}_std"] = None
                    continue

                fair_out[f"{metric}_mean"] = fair[mean_col]
                fair_out[f"{metric}_std"] = clean_std(fair, std_col)

            fair_out["selected_by"] = "AUPRC_IF"
            fair_out["best_params"] = format_params(fair)

            fair_rows.append(fair_out)

        # =========================
        # SAVE BOTH TABLES
        # =========================

        best_df = pd.DataFrame(best_rows)
        fair_df = pd.DataFrame(fair_rows)

        best_path = os.path.join(
            OUTPUT_DIR,
            f"{filename}_{dataset}_best_aupcr.csv"
        )

        fair_path = os.path.join(
            OUTPUT_DIR,
            f"{filename}_{dataset}_fair.csv"
        )

        best_df.to_csv(best_path, index=False)
        fair_df.to_csv(fair_path, index=False)

        print(f"Saved: {best_path}")
        print(f"Saved: {fair_path}")


def main():
    for file in sorted(os.listdir(INPUT_DIR)):
        if file.endswith(".csv"):
            process_file(os.path.join(INPUT_DIR, file))


if __name__ == "__main__":
    main()