import numpy as np
from sklearn.model_selection import train_test_split

# =========================================================
# SPLIT FUNCTION (SHARED)
# =========================================================
def split_anomaly_dataset(X, y, val_size=0.2, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    Xn, Xa = X[y == 0], X[y == 1]
    X_train, X_temp = train_test_split(
        Xn, test_size=(val_size + test_size), random_state=seed
    )
    val_ratio = val_size / (val_size + test_size)
    X_val_n, X_test_n = train_test_split(
        X_temp, test_size=val_ratio, random_state=seed
    )
    if len(Xa) >= 2:
        Xa_val, Xa_test = train_test_split(Xa, test_size=0.5, random_state=seed)
    elif len(Xa) == 1:
        Xa_val, Xa_test = Xa, np.empty((0, X.shape[1]))
    else:
        Xa_val = Xa_test = np.empty((0, X.shape[1]))

    X_val = np.concatenate([X_val_n, Xa_val])
    y_val = np.concatenate([np.zeros(len(X_val_n)), np.ones(len(Xa_val))])

    X_test = np.concatenate([X_test_n, Xa_test])
    y_test = np.concatenate([np.zeros(len(X_test_n)), np.ones(len(Xa_test))])

    # shuffle val/test
    def shuffle(X_, y_):
        idx = rng.permutation(len(X_))
        return X_[idx], y_[idx]

    X_val, y_val = shuffle(X_val, y_val)
    X_test, y_test = shuffle(X_test, y_test)

    y_train = np.zeros(len(X_train))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =========================================================
# THYROID
# =========================================================
def load_thyroid_easy(seed=42, n_samples=5000):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    print("Loading thyroid-ann...")

    X, y = fetch_openml("thyroid-ann", return_X_y=True, as_frame=True)

    y = pd.Series(y)
    if y.dtype == "object":
        y = y.astype("category").cat.codes

    normal_class = y.value_counts().idxmax()
    y = (y != normal_class).astype(int).values

    X = pd.DataFrame(X)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.values.astype(np.float32)

    rng = np.random.RandomState(seed)
    if len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X = StandardScaler().fit_transform(X)

    return split_anomaly_dataset(X, y, seed=seed)


# =========================================================
# SYNTHETIC
# =========================================================
def load_synthetic_anomaly(n_samples=5000, dim=30, seed=42):
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    n1, n2 = int(n_samples*0.5), int(n_samples*0.3)
    n3 = n_samples - n1 - n2

    X = np.vstack([
        rng.multivariate_normal(np.zeros(dim), np.eye(dim)*0.3, n1),
        rng.multivariate_normal(np.ones(dim)*2, np.eye(dim), n2),
        rng.multivariate_normal(np.ones(dim), np.eye(dim)*1.7, n3)
    ])

    y = np.concatenate([np.zeros(n1), np.zeros(n2), np.ones(n3)])

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X = StandardScaler().fit_transform(X)

    return split_anomaly_dataset(X, y, seed=seed)


# =========================================================
# SHUTTLE
# =========================================================
def load_shuttle(seed=42, n_samples=5000):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    print("Loading Shuttle...")

    X, y = fetch_openml("shuttle", version=1, return_X_y=True, as_frame=False)
    y = (y.astype(int) != 1).astype(int)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)

    X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)

    return split_anomaly_dataset(X, y, seed=seed)



# =========================================================
# USAGE
# =========================================================
if __name__ == "__main__":
    datasets = {
        "thyroid": load_thyroid_easy(seed=42),
        "synthetic": load_synthetic_anomaly(seed=42),
        "shuttle": load_shuttle(seed=42),
    }

    print("\n=== SUMMARY ===")
    for name, (train, val, test) in datasets.items():
        print(f"\n{name}")
        for split_name, (X_, y_) in zip(
                ["train", "val", "test"], [train, val, test]
        ):
            print(f"{split_name:5s}: {X_.shape}, anomaly%={y_.mean()*100:.2f}")