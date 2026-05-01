import torch
import numpy as np
from sklearn.decomposition import PCA


def drsne(
        X,
        P,
        knn_indices,
        rho_high,
        n_iter=600,
        warmup=50,
        lr=2.0,
        lambda_density=0.01,
        seed = 42,
        verbose = True
):
    # =========================================================
    # INIT
    # =========================================================
    torch.manual_seed(seed)
    np.random.seed(seed)

    Z_init = np.random.normal(0, 1e-4, size=(X.shape[0], 2))
    Z = torch.tensor(Z_init, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([Z], lr=lr)

    if isinstance(rho_high, torch.Tensor):
        rho_high_t = rho_high.detach().clone().float()
    else:
        rho_high_t = torch.tensor(rho_high, dtype=torch.float32)

    if isinstance(knn_indices, torch.Tensor):
        knn_indices_t = knn_indices.detach().clone().long()
    else:
        knn_indices_t = torch.tensor(knn_indices, dtype=torch.long)
    log_rho_high = torch.log(rho_high_t + 1e-8)

    knn_indices_t = torch.tensor(knn_indices, dtype=torch.long)
    neighbors = knn_indices_t[:, 1:]  # precompute once

    # Precompute P_neighbors once
    P_neighbors = P.gather(1, neighbors)

    # Precompute warmup P once
    if warmup > 0:
        P_eff = P * 4.0
        P_eff = P_eff / P_eff.sum()

    history = {"total": [], "kl": [], "density": []}

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for it in range(n_iter):
        optimizer.zero_grad()

        # =========================================
        # FULL PAIRWISE DISTANCE
        # =========================================
        dist_sq = torch.cdist(Z, Z, p=2) ** 2
        dist_sq = dist_sq + 1e-8

        # =========================================
        # Q MATRIX
        # =========================================
        Q = 1.0 / (1.0 + dist_sq)
        Q.fill_diagonal_(0)
        Q /= Q.sum()  # in-place normalization

        # =========================================
        # SPARSE KL (inlined, no function call)
        # =========================================
        Q_neighbors = Q.gather(1, neighbors)

        kl = (
                     P_neighbors * (
                     torch.log(P_neighbors + 1e-8) - torch.log(Q_neighbors + 1e-8)
             )
             ).sum() / Z.shape[0]

        # =========================================
        # DENSITY LOSS (inlined, optimized)
        # =========================================
        Z_center = Z.unsqueeze(1)
        Z_neighbors = Z[neighbors]

        dists_sq = ((Z_center - Z_neighbors) ** 2).sum(dim=2)
        volume = torch.sqrt(dists_sq + 1e-8).sum(dim=1)

        rho_low = neighbors.shape[1] / volume
        rho_low = rho_low / (rho_low.detach().mean() + 1e-8)

        density_loss = (
                (log_rho_high - torch.log(rho_low + 1e-8)) ** 2
        ).mean()

        # =========================================
        # LOSS
        # =========================================
        if it < warmup:
            loss = (
                           P_eff * (torch.log(P_eff + 1e-8) - torch.log(Q + 1e-8))
                   ).sum() / Z.shape[0]
        else:
            loss = kl + lambda_density * density_loss

        # =========================================
        # OPTIMIZATION
        # =========================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Z], 1.0)
        optimizer.step()

        # =========================================
        # LOGGING
        # =========================================
        history["total"].append(loss.item())
        history["kl"].append(kl.item())
        history["density"].append(density_loss.item())

        if it % 50 == 0 and verbose:
            print(
                f"[Density t-SNE] Iter {it}: "
                f"KL={kl.item():.6f}, Density={density_loss.item():.6f}"
            )

    return Z.detach().numpy(), history