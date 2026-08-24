"""Self-replicator (composome/compotype) detection, classic GARD pipeline.

Per Segré et al. 2000 / Shenhav, Oz & Lancet 2007 / GARD10 conventions:
  1. compare pre-fission compositions of consecutive generations with cosine
     similarity H; a generation is in "drift" if its mean H to predecessor and
     successor is below the threshold (classic value 0.9);
  2. cluster non-drift compositions with cosine k-means, k chosen by max
     silhouette -> compotypes;
  3. (Pigozzi & Levin variant) the most recurring compotype centroid defines
     the self-replicator; every molecular step whose composition has
     H >= threshold to that centroid is labeled self-replicating.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

H_THRESHOLD = 0.9


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def cosine_to(comps: np.ndarray, center: np.ndarray) -> np.ndarray:
    return _unit(comps) @ _unit(center)


def nondrift_mask(pre_fission: np.ndarray, threshold: float = H_THRESHOLD):
    """Generations whose mean similarity to predecessor & successor >= threshold."""
    u = _unit(pre_fission)
    h_next = np.sum(u[:-1] * u[1:], axis=1)
    n = len(pre_fission)
    avg = np.empty(n)
    avg[0] = h_next[0]
    avg[-1] = h_next[-1]
    if n > 2:
        avg[1:-1] = 0.5 * (h_next[:-1] + h_next[1:])
    return avg >= threshold


def compotypes(nondrift_comps: np.ndarray, k_max: int = 8, seed: int = 0):
    """Cosine k-means over non-drift compositions; k by max silhouette.
    Returns (centroids, labels)."""
    u = _unit(nondrift_comps.astype(float))
    n = len(u)
    if n < 3:
        return u.mean(axis=0, keepdims=True), np.zeros(n, dtype=int)
    best = (None, None, -np.inf)
    for k in range(2, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(u)
        if len(np.unique(km.labels_)) < 2:
            continue
        s = silhouette_score(u, km.labels_, metric="cosine")
        if s > best[2]:
            best = (km.cluster_centers_, km.labels_, s)
    if best[0] is None:
        return u.mean(axis=0, keepdims=True), np.zeros(n, dtype=int)
    return best[0], best[1]


def label_self_replication(counts: np.ndarray, fission_steps: np.ndarray,
                           threshold: float = H_THRESHOLD, seed: int = 0):
    """Boolean per molecular step: within threshold of the dominant compotype.

    Falls back to the single most self-similar generation pair when every
    generation is in drift (no composome ever forms)."""
    pre = counts[fission_steps].astype(float)
    mask = nondrift_mask(pre, threshold)
    if mask.sum() >= 2:
        cents, labels = compotypes(pre[mask], seed=seed)
        dominant = cents[np.bincount(labels).argmax()]
    else:
        u = _unit(pre)
        h_next = np.sum(u[:-1] * u[1:], axis=1)
        dominant = pre[int(np.argmax(h_next))]
    return cosine_to(counts.astype(float), dominant) >= threshold
