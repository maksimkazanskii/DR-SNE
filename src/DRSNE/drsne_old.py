import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def drsne(
        X,
        n_components=2,
        perplexity=50.0,
        lambda_density=0.0,
        k_density=50,
        n_iter=1000,
        warmup=50,
        lr=2.0,
        seed=42,
        verbose=False,
):

    # =========================================================
    # SEED
    # =========================================================
    torch.manual_seed(seed)
    np.random.seed(seed)

    # =========================================================
    # KNN DENSITY (inline compute_knn_density)
    # =========================================================
    n = len(X)
    k_eff = min(k_density + 1, n)

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X)
    distances, indices = nbrs.kneighbors(X)

    if k_eff > 1:
        d = distances[:, 1:]

        volume = d.sum(axis=1) + 1e-8
        rho_high = (k_eff - 1) / volume
    else:
        volume = distances.sum(axis=1) + 1e-8
        rho_high = 1.0 / volume

    rho_high /= (rho_high.mean() + 1e-8)

    knn_indices = indices.astype(np.int64)
    rho_high = rho_high.astype(np.float32)

    # =========================================================
    # P MATRIX (inline compute_P)
    # =========================================================
    D = pairwise_distances(X, squared=True)
    P = np.zeros((n, n), dtype=np.float32)
    log_perp = np.log(perplexity)
    rho_high_t = torch.tensor(rho_high, dtype=torch.float32)
    log_rho_high = torch.log(rho_high_t + 1e-8)
    for i in range(n):
        beta = 1.0
        betamin, betamax = -np.inf, np.inf

        Di = np.delete(D[i], i)

        for _ in range(50):
            Pi = np.exp(-Di * beta)
            Pi /= (Pi.sum() + 1e-8)

            H = -np.sum(Pi * np.log(Pi + 1e-8))
            Hdiff = H - log_perp

            if abs(Hdiff) < 1e-5:
                break

            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2

        inds = np.concatenate((np.r_[0:i], np.r_[i + 1:n]))
        P[i, inds] = Pi

    P = (P + P.T) / (2 * n)

    # =========================================================
    # PREP TORCH
    # =========================================================
    Z = torch.tensor(
        np.random.normal(0, 1e-4, size=(n, n_components)),
        dtype=torch.float32,
        requires_grad=True
    )

    optimizer = torch.optim.Adam([Z], lr=lr)

    # !rho_high_t = torch.tensor(rho_high)
    # !log_rho_high = torch.log(rho_high_t + 1e-8)
    P_torch = torch.tensor(P, dtype=torch.float32)
    neighbors = torch.tensor(knn_indices[:, 1:], dtype=torch.long)

    P_neighbors = P_torch.gather(1, neighbors)

    P_neighbors = P_torch.gather(1, neighbors)
    P_neighbors = P_neighbors / (P_neighbors.sum(dim=1, keepdim=True) + 1e-8)



    if warmup > 0:
        P_eff = P_torch * 4.0
        P_eff = P_eff / P_eff.sum()

    # =========================================================
    # TRAIN LOOP (inline run_density_tsne)
    # =========================================================
    for it in range(n_iter):
        optimizer.zero_grad()

        dist_sq = torch.cdist(Z, Z, p=2) ** 2
        dist_sq = dist_sq + 1e-8

        Q = 1.0 / (1.0 + dist_sq)
        Q.fill_diagonal_(0)
        Q /= Q.sum()

        Q_neighbors = Q.gather(1, neighbors)
        Q_neighbors = Q_neighbors / (Q_neighbors.sum(dim=1, keepdim=True) + 1e-8)

        kl = (
                     P_neighbors *
                     (torch.log(P_neighbors + 1e-8) - torch.log(Q_neighbors + 1e-8))
             ).sum() / n

        # density
        Z_center = Z.unsqueeze(1)
        Z_neighbors = Z[neighbors]

        dists_sq = ((Z_center - Z_neighbors) ** 2).sum(dim=2)
        # !!! volume = (torch.sqrt(dists_sq + 1e-8) ** n_components).sum(dim=1)
        volume = torch.sqrt(dists_sq + 1e-8).sum(dim=1)
        rho_low = neighbors.shape[1] / volume
        rho_low = rho_low / (rho_low.detach().mean() + 1e-8)
        density_loss = (
                (log_rho_high - torch.log(rho_low + 1e-8)) ** 2
        ).mean()

        if it < warmup:
            loss = (
                           P_eff *
                           (torch.log(P_eff + 1e-8) - torch.log(Q + 1e-8))
                   ).sum() / n
        else:
            loss = kl + lambda_density * density_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([Z], 1.0)
        optimizer.step()

        if verbose and it % 50 == 0:
            print(f"[DR-SNE] Iter {it}: KL={kl.item():.6f}, Density={density_loss.item():.6f}")

    return Z.detach().cpu().numpy()