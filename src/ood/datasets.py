import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99




from sklearn.datasets import fetch_openml


def load_density_rings(
        n_norm=4500,
        n_anom=500,
        dim=20,
        seed=42
):
    import numpy as np

    rng = np.random.RandomState(seed)

    # -----------------------------------
    # 2D structure (rings)
    # -----------------------------------
    def sample_ring(n, radius, noise):
        angles = rng.uniform(0, 2*np.pi, size=n)
        r = radius + rng.normal(scale=noise, size=n)

        x = np.stack([
            r * np.cos(angles),
            r * np.sin(angles)
        ], axis=1)
        return x

    # Dense inner ring
    X_inner = sample_ring(n_norm // 2, radius=5.0, noise=0.2)

    # Sparse outer ring (same radius scale!)
    X_outer = sample_ring(n_norm // 2, radius=6.0, noise=1.0)

    X_2d = np.vstack([X_inner, X_outer])

    # -----------------------------------
    # Embed into high-D
    # -----------------------------------
    W = rng.normal(size=(2, dim))
    X_hd = X_2d @ W

    # -----------------------------------
    # ANOMALIES
    # -----------------------------------
    X_anom = []

    for _ in range(n_anom):
        if rng.rand() < 0.5:
            # anomaly between rings (density gap)
            r = rng.uniform(5.2, 5.8)
            noise = 0.3
        else:
            # anomaly inside dense ring but too noisy
            r = rng.uniform(5.0, 5.0)
            noise = 1.2

        angle = rng.uniform(0, 2*np.pi)
        x2d = np.array([
            r * np.cos(angle),
            r * np.sin(angle)
        ]) + rng.normal(scale=noise, size=2)

        xhd = x2d @ W
        X_anom.append(xhd)

    X_anom = np.array(X_anom)

    # -----------------------------------
    # FINAL
    # -----------------------------------
    X = np.vstack([X_hd, X_anom])
    y = np.array([0] * len(X_hd) + [1] * len(X_anom), dtype=np.int64)

    return X.astype(np.float32), y

def load_kddcup99_small(
        n_norm=4500,
        n_anom=500,
        seed=42,
):
    import numpy as np
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    rng = np.random.RandomState(seed)
    data = fetch_kddcup99(subset="SA", percent10=True)
    X = np.array(data.data, dtype=object)
    y = np.array([t.decode("utf-8") for t in data.target])

    y_bin = (y != "normal.").astype(int)

    # =========================================================
    # FEATURE SPLIT
    # =========================================================
    cat_idx = [1, 2, 3]
    num_idx = [i for i in range(X.shape[1]) if i not in cat_idx]

    X_cat = X[:, cat_idx]
    X_num = X[:, num_idx].astype(float)

    # =========================================================
    # SPLIT NORMAL / ANOMALY
    # =========================================================
    X_norm_cat = X_cat[y_bin == 0]
    X_norm_num = X_num[y_bin == 0]

    X_anom_cat = X_cat[y_bin == 1]
    X_anom_num = X_num[y_bin == 1]

    # safety
    if len(X_norm_cat) < n_norm:
        raise ValueError("Not enough normal samples")
    if len(X_anom_cat) < n_anom:
        raise ValueError("Not enough anomaly samples")

    # =========================================================
    # SAMPLE
    # =========================================================
    idx_norm = rng.choice(len(X_norm_cat), n_norm, replace=False)
    idx_anom = rng.choice(len(X_anom_cat), n_anom, replace=False)

    Xn_cat = X_norm_cat[idx_norm]
    Xn_num = X_norm_num[idx_norm]

    Xa_cat = X_anom_cat[idx_anom]
    Xa_num = X_anom_num[idx_anom]

    # =========================================================
    # ENCODE (fit ONLY on normal samples)
    # =========================================================
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(Xn_cat)

    def transform(X_num, X_cat):
        return np.hstack([X_num, enc.transform(X_cat)])

    Xn = transform(Xn_num, Xn_cat)
    Xa = transform(Xa_num, Xa_cat)

    # =========================================================
    # MERGE
    # =========================================================
    X_final = np.vstack([Xn, Xa])
    y_final = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)

    # =========================================================
    # SHUFFLE
    # =========================================================
    perm = rng.permutation(len(X_final))
    X_final = X_final[perm]
    y_final = y_final[perm]

    # =========================================================
    # SCALE (fit on normals only — important)
    # =========================================================
    scaler = StandardScaler().fit(Xn)

    X_final = scaler.transform(X_final)

    return X_final.astype(np.float32), y_final


def load_fashion_anomaly(normal_class=0, n_norm=4500, n_anom=500, seed=42):
    rng = np.random.RandomState(seed)

    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    X = X.astype(np.float32)

    X_norm = X[y == normal_class]
    X_anom = X[y != normal_class]

    idx_norm = rng.choice(len(X_norm), n_norm, replace=False)
    idx_anom = rng.choice(len(X_anom), n_anom, replace=False)

    X_final = np.vstack([X_norm[idx_norm], X_anom[idx_anom]])
    y_final = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)

    return X_final, y_final

def load_shuttle(seed=42, n_samples=5000):
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    # -----------------------------------
    # LOAD
    # -----------------------------------
    X, y = fetch_openml("shuttle", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # -----------------------------------
    # DEFINE ANOMALY
    # -----------------------------------
    # Class 1 = normal, all others = anomaly
    y_bin = (y != 1).astype(np.int64)

    # -----------------------------------
    # SUBSAMPLE WHILE PRESERVING
    # ORIGINAL CLASS DISTRIBUTION
    # -----------------------------------
    if n_samples is not None and n_samples < len(X):
        classes, counts = np.unique(y, return_counts=True)
        proportions = counts / counts.sum()

        # initial allocation
        target_counts = np.floor(proportions * n_samples).astype(int)

        # make sure sum matches exactly n_samples
        remainder = n_samples - target_counts.sum()
        if remainder > 0:
            frac = proportions * n_samples - target_counts
            add_order = np.argsort(-frac)
            for i in add_order[:remainder]:
                target_counts[i] += 1

        selected_idx = []

        for cls, n_take in zip(classes, target_counts):
            cls_idx = np.where(y == cls)[0]
            n_take = min(n_take, len(cls_idx))
            if n_take > 0:
                chosen = rng.choice(cls_idx, size=n_take, replace=False)
                selected_idx.append(chosen)

        selected_idx = np.concatenate(selected_idx)
        rng.shuffle(selected_idx)

        X = X[selected_idx]
        y = y[selected_idx]
        y_bin = y_bin[selected_idx]

    # -----------------------------------
    # SCALE
    # -----------------------------------
    X = StandardScaler().fit_transform(X)

    # -----------------------------------
    # INFO
    # -----------------------------------
    print("Shuttle dataset")
    print("Shape:", X.shape)
    print("Anomaly ratio:", y_bin.mean())

    return X.astype(np.float32), y_bin

def load_fashion_dino(
        normal_class=0,
        n_norm=4500,
        n_anom=500,
        seed=42,
        device="cpu"   # change to "cuda" if available
):
    import numpy as np
    import torch
    import torchvision.transforms as T
    from torchvision.datasets import FashionMNIST

    rng = np.random.RandomState(seed)

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------
    dataset = FashionMNIST(root="./data", train=True, download=True)

    X = dataset.data.numpy()   # (N, 28, 28)
    y = dataset.targets.numpy()

    # -----------------------------------
    # SPLIT NORMAL / ANOMALY
    # -----------------------------------
    X_norm = X[y == normal_class]
    X_anom = X[y != normal_class]

    idx_norm = rng.choice(len(X_norm), n_norm, replace=False)
    idx_anom = rng.choice(len(X_anom), n_anom, replace=False)

    X_sel = np.concatenate([X_norm[idx_norm], X_anom[idx_anom]], axis=0)
    y_sel = np.array([0]*n_norm + [1]*n_anom, dtype=np.int64)

    # -----------------------------------
    # DINO TRANSFORM
    # -----------------------------------
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.Grayscale(num_output_channels=3),  # convert to 3-channel
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # -----------------------------------
    # LOAD DINO MODEL
    # -----------------------------------
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )
    model.eval().to(device)

    # -----------------------------------
    # EXTRACT EMBEDDINGS
    # -----------------------------------
    batch_size = 64
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(X_sel), batch_size):
            batch = X_sel[i:i+batch_size]

            imgs = torch.stack([transform(img) for img in batch]).to(device)

            feats = model(imgs)   # (B, D)
            embeddings.append(feats.cpu().numpy())

    X_emb = np.vstack(embeddings).astype(np.float32)

    # -----------------------------------
    # SHUFFLE
    # -----------------------------------
    perm = rng.permutation(len(X_emb))
    X_emb = X_emb[perm]
    y_sel = y_sel[perm]

    return X_emb, y_sel
def load_pbmc(seed=42, n_pcs=50, n_hvg=2000, rare_k=2):
    import numpy as np
    import scanpy as sc

    sc.settings.seed = seed

    adata = sc.datasets.pbmc3k()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=1.0, random_state=seed)

    X = adata.obsm["X_pca"].astype(np.float32)
    y_clusters = adata.obs["leiden"].astype(int).values

    counts = np.bincount(y_clusters)
    rare_types = np.argsort(counts)[:rare_k]
    y = np.isin(y_clusters, rare_types).astype(np.int64)

    # just return full dataset (NO subsampling)
    return X, y


def load_tcga(seed=42, n_samples=6000):
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(seed)

    print("Loading tumor dataset (TCGA via OpenML)...")

    X, y = fetch_openml(
        name="TCGA-PANCAN-HiSeq",
        version=1,
        return_X_y=True,
        as_frame=False
    )

    X = X.astype(np.float32)

    # -----------------------------------
    # DEFINE ANOMALY
    # -----------------------------------
    # choose one cancer type as anomaly
    classes, counts = np.unique(y, return_counts=True)

    anomaly_class = classes[np.argmin(counts)]  # rarest cancer
    y_bin = (y == anomaly_class).astype(np.int64)

    # -----------------------------------
    # SUBSAMPLE
    # -----------------------------------
    idx = rng.choice(len(X), n_samples, replace=False)
    X = X[idx]
    y_bin = y_bin[idx]

    # -----------------------------------
    # SCALE + REDUCE
    # -----------------------------------
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=100, random_state=seed).fit_transform(X)

    print("Shape:", X.shape)
    print("Anomaly ratio:", y_bin.mean())

    return X.astype(np.float32), y_bin

def load_tumor(path="data/cleaned_dataset_Thyroid1.csv"):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(path)

    # -----------------------------------
    # TARGET
    # -----------------------------------
    y = df["binaryClass"].values.astype(np.int64)

    # -----------------------------------
    # FEATURES
    # -----------------------------------
    X = df.drop(columns=["binaryClass"]).values.astype(np.float32)

    # -----------------------------------
    # SCALE
    # -----------------------------------
    X = StandardScaler().fit_transform(X)

    print("Shape:", X.shape)
    print("Anomaly ratio:", y.mean())

    return X, y

def load_spiral(
        n_samples=5000,
        dim=50,
        anomaly_frac=0.05,
        seed=42
):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(seed)

    # -----------------------------------
    # 1. BASE SPIRAL
    # -----------------------------------
    t = np.linspace(0, 4 * np.pi, n_samples)

    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)

    X_2d = np.stack([x, y], axis=1)

    # -----------------------------------
    # 2. DENSITY MODULATION
    # -----------------------------------
    density = 1.0 + 0.8 * np.sin(3 * t)

    prob = density / density.sum()

    idx = rng.choice(n_samples, size=n_samples, replace=True, p=prob)
    X_2d = X_2d[idx]
    t = t[idx]

    # -----------------------------------
    # 3. NOISE
    # -----------------------------------
    X_2d += rng.normal(scale=0.05, size=X_2d.shape)

    # -----------------------------------
    # 4. HIGH-D EMBEDDING
    # -----------------------------------
    W = rng.normal(size=(2, dim))
    X_hd = X_2d @ W

    X_hd = StandardScaler().fit_transform(X_hd)

    if dim > 50:
        X_hd = PCA(n_components=50, random_state=seed).fit_transform(X_hd)

    # -----------------------------------
    # 5. DEFINE "RARE = LOW DENSITY"
    # -----------------------------------
    # (exact analog of PBMC rare cell logic)

    density_sampled = 1.0 + 0.8 * np.sin(3 * t)

    threshold = np.percentile(density_sampled, anomaly_frac * 100)

    y = (density_sampled <= threshold).astype(np.int64)

    return X_hd.astype(np.float32), y



def load_resnet_cifar10(
        n_samples=5000,
        rare_classes=(0,),   # anomalies = these classes
        seed=42,
        batch_size=64
):
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import warnings

    # -----------------------------------
    # SILENCE TORCHVISION WARNING
    # -----------------------------------
    warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")

    rng = np.random.RandomState(seed)

    # -----------------------------------
    # DATA
    # -----------------------------------
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    idx = rng.choice(len(dataset), n_samples, replace=False)
    subset = Subset(dataset, idx)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -----------------------------------
    # MODEL (ResNet50 feature extractor)
    # -----------------------------------
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -----------------------------------
    # EXTRACT EMBEDDINGS (FAST BATCHED)
    # -----------------------------------
    X_list = []
    y_class = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            feats = model(imgs).squeeze(-1).squeeze(-1)  # (B, 2048)
            X_list.append(feats.cpu().numpy())
            y_class.append(labels.numpy())

    X = np.vstack(X_list)
    y_class = np.concatenate(y_class)

    # -----------------------------------
    # NORMALIZE + PCA
    # -----------------------------------
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=50, random_state=seed).fit_transform(X)

    # -----------------------------------
    # TRUE ANOMALY LABELS
    # -----------------------------------
    y = np.isin(y_class, rare_classes).astype(np.int64)

    print("\nResNet CIFAR10 dataset")
    print("Shape:", X.shape)
    print("Classes:", np.unique(y_class))
    print("Rare classes:", rare_classes)
    print("Anomaly ratio:", y.mean())

    return X.astype(np.float32), y

def load_swiss_density(n=5000, seed=42):
    from sklearn.datasets import make_swiss_roll
    import numpy as np

    X, t = make_swiss_roll(n, noise=0.1, random_state=seed)

    # density modulation
    density = 1 + 0.8 * np.sin(t / 2)

    prob = density / density.sum()
    idx = np.random.choice(n, size=n, replace=True, p=prob)

    X = X[idx]
    t = t[idx]
    density = density[idx]

    # -----------------------------------
    # DEFINE ANOMALIES FROM TRUE DENSITY
    # -----------------------------------
    threshold = np.percentile(density, 10)
    y = (density < threshold).astype(np.int64)

    return X.astype(np.float32), y