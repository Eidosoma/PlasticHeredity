"""Phase F infrastructure: composome atlases, mirroring the Kahana /
historical GARD10 protocol (tgs_acluster):

1. run independent training lineages (spawn domain 12; never the
   evaluated trajectories);
2. collect PRE-FISSION compositions (the assembly at nmax, before
   split);
3. non-drift filter: keep generations whose adjacent pre-fission cosine
   similarity to the previous generation exceeds 0.9 (tgs_nondrift
   analog);
4. k-means over the retained compositions (k in 1..6 selected by mean
   silhouette on cosine distance, fixed random_state — deterministic);
   cluster centers = composomes/compotypes.

Distance to atlas: d(x, A) = 1 - max_a cosine(x, a).
"""

from __future__ import annotations

import numpy as np

import sim
import cohort
import growth_trace as GT

ATLAS_DOMAIN = 12
N_LINEAGES = 3
N_FISSIONS = 500
KMAX = 6


def _kmeans_cosine(X, k, seed=0, n_init=10, iters=100):
    """k-means on L2-normalized rows (sqEuclidean on normalized rows is
    monotone in cosine distance). Deterministic given seed."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        idx = rng.choice(len(Xn), size=k, replace=False)
        C = Xn[idx].copy()
        for _ in range(iters):
            simm = Xn @ C.T
            lab = np.argmax(simm, axis=1)
            newC = np.zeros_like(C)
            for j in range(k):
                m = lab == j
                if m.any():
                    v = Xn[m].mean(axis=0)
                    newC[j] = v / max(np.linalg.norm(v), 1e-12)
                else:
                    newC[j] = Xn[rng.integers(len(Xn))]
            if np.allclose(newC, C, atol=1e-10):
                C = newC
                break
            C = newC
        inertia = float(np.sum(1.0 - np.max(Xn @ C.T, axis=1)))
        if best is None or inertia < best[0]:
            best = (inertia, C, np.argmax(Xn @ C.T, axis=1))
    return best[1], best[2]


def _mean_silhouette_cosine(X, labels, centers):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    k = len(centers)
    if k == 1:
        return -1.0
    D = 1.0 - (Xn @ Xn.T)
    vals = []
    for i in range(len(Xn)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == j].mean()
                for j in range(k) if j != labels[i] and (labels == j).any())
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else -1.0


def build_atlas(m, cand, entropy, nmax, n_lineages=N_LINEAGES,
                n_fissions=N_FISSIONS, subsample=400):
    """Composome atlas for (matrix m, candidate) from independent
    lineages. Returns dict with unit-normalized composome centers,
    chosen k, occupancy fractions, and the training pre-fission count."""
    beta, n0 = cohort.matrix_and_init(entropy, m)
    cand_i = cohort.CANDIDATES.index(cand)
    pres = []
    for li in range(n_lineages):
        rng = cohort._rng(entropy, ATLAS_DOMAIN, cand_i, m, li)
        out = GT.traced_run_fissions(n0, beta, cand, n_fissions, rng,
                                     nmax, grid_step=nmax)  # no snaps
        pres.extend(r["parent"].astype(float) for r in out["recs"])
    P = np.array(pres)
    # non-drift filter (adjacent similarity > 0.9)
    keep = [0]
    for i in range(1, len(P)):
        if sim.cosine_h(P[i], P[i - 1]) > sim.H_THRESH:
            keep.append(i)
    Pk = P[keep]
    if len(Pk) > subsample:
        sel = np.random.default_rng(m * 7 + cand_i).choice(
            len(Pk), size=subsample, replace=False)
        Pk = Pk[sel]
    best = None
    for k in range(1, min(KMAX, len(Pk)) + 1):
        C, lab = _kmeans_cosine(Pk, k, seed=1000 + m)
        s = _mean_silhouette_cosine(Pk, lab, C) if k > 1 else 0.0
        if best is None or s > best[0]:
            best = (s, k, C, lab)
    s, k, C, lab = best
    occ = np.array([(lab == j).mean() for j in range(k)])
    return {"centers": C, "k": k, "silhouette": s, "occupancy": occ,
            "n_train": int(len(Pk)), "n_total_pre": int(len(P))}


def nearest_sim(x, atlas) -> float:
    xn = x.astype(float)
    nrm = np.linalg.norm(xn)
    if nrm < 1e-12:
        return 0.0
    return float(np.max(atlas["centers"] @ (xn / nrm)))


def dist(x, atlas) -> float:
    return 1.0 - nearest_sim(x, atlas)


def nearest_center(x, atlas) -> int:
    xn = x.astype(float)
    xn = xn / max(np.linalg.norm(xn), 1e-12)
    return int(np.argmax(atlas["centers"] @ xn))
