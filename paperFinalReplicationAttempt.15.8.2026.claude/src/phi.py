"""Causal emergence Phi_r for multivariate time series.

Phi_r = I(X_t; X_{t+1}) - I(M1_t; X_{t+1}) - I(M2_t; X_{t+1}) over the
minimum-information bipartition {M1, M2} — Rosas et al. 2020's practical
criterion Psi with the whole system as macro variable (Pigozzi & Levin M&M).

Two estimators, both Gaussian:
  - local (windowless): one Gaussian fit per run, per-step local MI values;
    mean over steps equals the static measure (phyid convention);
  - sliding-window: refit per window (nonstationarity robustness check).

MIB: spectral-clustering sweep over soft/hard-thresholded |corr| graphs
(Toker & Sommer 2019), candidates scored by inter-part Gaussian MI normalized
by mean part entropy; static per run.
"""

import numpy as np
from sklearn.cluster import KMeans


def clr(counts: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    """Centered log-ratio of relative compositions; drops the last component."""
    x = counts + pseudocount
    x = x / x.sum(axis=1, keepdims=True)
    logx = np.log(x)
    z = logx - logx.mean(axis=1, keepdims=True)
    return z[:, :-1]


def zscore(z: np.ndarray) -> np.ndarray:
    sd = z.std(axis=0)
    return (z - z.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def shrunk_cov(z: np.ndarray, shrink: float = None) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward scaled identity on the joint matrix."""
    if shrink is None:
        from sklearn.covariance import ledoit_wolf
        c, _ = ledoit_wolf(z)
        return c
    c = np.cov(z, rowvar=False)
    d = c.shape[0]
    return c * (1 - shrink) + shrink * np.trace(c) / d * np.eye(d)


def _logdet(c: np.ndarray) -> float:
    return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(c))))


def gaussian_mi_from_cov(cov: np.ndarray, idx_a, idx_b) -> float:
    a, b = np.asarray(idx_a), np.asarray(idx_b)
    ab = np.concatenate([a, b])
    return 0.5 * (_logdet(cov[np.ix_(a, a)]) + _logdet(cov[np.ix_(b, b)])
                  - _logdet(cov[np.ix_(ab, ab)]))


def gaussian_entropy(cov: np.ndarray, idx) -> float:
    i = np.asarray(idx)
    d = len(i)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + _logdet(cov[np.ix_(i, i)]))


# ---------------------------------------------------------------- MIB search

def _spectral_candidates(corr: np.ndarray, rng: np.random.Generator):
    """Candidate bipartitions from spectral clustering of thresholded |corr| graphs."""
    n = corr.shape[0]
    w0 = np.abs(np.nan_to_num(corr))
    np.fill_diagonal(w0, 0.0)
    seen, cands = set(), []
    for beta in np.logspace(0, 1, 5):
        wb = w0 ** beta
        for pct in (0, 50, 80, 90):
            w = wb.copy()
            if pct > 0:
                w[w < np.percentile(wb[wb > 0], pct)] = 0.0
            deg = w.sum(axis=1)
            deg[deg < 1e-12] = 1e-12
            d_isqrt = 1.0 / np.sqrt(deg)
            l_sym = np.eye(n) - (w * d_isqrt).T * d_isqrt
            vals, vecs = np.linalg.eigh(l_sym)
            emb = vecs[:, 1:3]
            labels = KMeans(n_clusters=2, n_init=3,
                            random_state=int(rng.integers(1 << 31))).fit_predict(emb)
            if labels.min() == labels.max():
                continue
            key = tuple(labels) if labels[0] == 0 else tuple(1 - labels)
            if key in seen:
                continue
            seen.add(key)
            cands.append((np.where(labels == 0)[0], np.where(labels == 1)[0]))
    if not cands:
        half = n // 2
        cands.append((np.arange(half), np.arange(half, n)))
    return cands


def min_information_bipartition(z: np.ndarray, seed: int = 0):
    """MIB of the columns of z: candidate with least normalized inter-part MI."""
    corr = np.corrcoef(z, rowvar=False)
    cov = shrunk_cov(z)
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for m1, m2 in _spectral_candidates(corr, rng):
        mi = gaussian_mi_from_cov(cov, m1, m2)
        k = 0.5 * (gaussian_entropy(cov, m1) + gaussian_entropy(cov, m2))
        score = mi / max(abs(k), 1e-12)
        if score < best_score:
            best, best_score = (m1, m2), score
    return best


# ------------------------------------------------------- local (per-step) Phi_r

def _local_mi(x: np.ndarray, cov: np.ndarray, idx_a, idx_b) -> np.ndarray:
    """Per-sample local MI i(a_t; b_t) = h(a_t) + h(b_t) - h(a_t, b_t)."""
    from scipy.stats import multivariate_normal as mvn
    a, b = np.asarray(idx_a), np.asarray(idx_b)
    ab = np.concatenate([a, b])
    out = np.zeros(x.shape[0])
    for idx, sign in ((a, -1.0), (b, -1.0), (ab, +1.0)):
        sub = cov[np.ix_(idx, idx)]
        lp = mvn.logpdf(x[:, idx], mean=np.zeros(len(idx)), cov=sub,
                        allow_singular=True)
        out += sign * lp  # h = -logpdf; i = h_a + h_b - h_ab = -lp_a - lp_b + lp_ab
    return out


def phi_r_local(counts: np.ndarray, lag: int = 1, pseudocount: float = 1.0,
                mib_seed: int = 0):
    """Windowless per-step Phi_r trajectory (length n_steps - lag).

    Returns (phi_traj, info) where info holds the MIB parts and static value.
    """
    z = zscore(clr(counts, pseudocount))
    d = z.shape[1]
    m1, m2 = min_information_bipartition(z, seed=mib_seed)
    joint = np.hstack([z[:-lag], z[lag:]])
    joint = joint - joint.mean(axis=0)
    cov = shrunk_cov(joint)
    past, future = np.arange(d), np.arange(d, 2 * d)
    i_whole = _local_mi(joint, cov, past, future)
    i_m1 = _local_mi(joint, cov, m1, future)
    i_m2 = _local_mi(joint, cov, m2, future)
    phi = i_whole - i_m1 - i_m2
    info = {"m1": m1, "m2": m2, "phi_static": float(phi.mean())}
    return phi, info


# ------------------------------------------------------------ windowed Phi_r

def phi_r_window(z: np.ndarray, m1=None, m2=None, lag: int = 1,
                 shrink: float = None) -> float:
    """Static Phi_r on one window (rows = time, cols = components)."""
    d = z.shape[1]
    if m1 is None or m2 is None:
        m1, m2 = min_information_bipartition(z)
    joint = np.hstack([z[:-lag], z[lag:]])
    cov = shrunk_cov(joint, shrink)
    past, future = np.arange(d), np.arange(d, 2 * d)
    return (gaussian_mi_from_cov(cov, past, future)
            - gaussian_mi_from_cov(cov, m1, future)
            - gaussian_mi_from_cov(cov, m2, future))


def phi_r_trajectory_windowed(counts: np.ndarray, window: int = 200,
                              stride: int = 10, lag: int = 1,
                              pseudocount: float = 1.0, mib_seed: int = 0):
    """Sliding-window Phi_r with a static per-run MIB; value at window right edge."""
    z = zscore(clr(counts, pseudocount))
    m1, m2 = min_information_bipartition(z, seed=mib_seed)
    steps, vals = [], []
    for end in range(window, z.shape[0] + 1, stride):
        zw = z[end - window:end]
        vals.append(phi_r_window(zw, m1, m2, lag))
        steps.append(end - 1)
    return np.array(steps), np.array(vals)
