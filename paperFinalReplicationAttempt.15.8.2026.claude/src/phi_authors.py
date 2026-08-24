"""Best-evidence reconstruction of the authors' operational Phi_r pipeline
(from Pigozzi, Goldstein & Levin, Commun Biol 2025 + Pigozzi & Levin RL
preprint methods text):

  1. substrate: CLR-transformed relative compositions, last component dropped
     (GARD paper); 2. pairwise lag-1 Gaussian MI matrix, I = -ln(1-rho^2)/2;
  3. graph Laplacian of that matrix, Fiedler-vector bipartition;
  4. average the (CLR) channels within each part -> two scalar series;
  5. Gaussian PhiID atoms (MMI) on the 2-scalar system, local (per-step);
  6. Phi_r(t) = str + stx + sty + sts  (downward causation + causal
     decoupling — "emergence capacity"), NOT the printed Psi formula.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")
from phi import clr

ROOT = Path(__file__).parent.parent


def lag1_mi_matrix(z: np.ndarray) -> np.ndarray:
    """Pairwise lag-1 Gaussian MI: I_ij = -0.5*ln(1 - rho(z_i[t], z_j[t+1])^2),
    symmetrized."""
    a, b = z[:-1], z[1:]
    an = (a - a.mean(0)) / (a.std(0) + 1e-12)
    bn = (b - b.mean(0)) / (b.std(0) + 1e-12)
    rho = an.T @ bn / len(an)
    rho = np.clip((rho + rho.T) / 2, -0.999999, 0.999999)
    mi = -0.5 * np.log(1 - rho ** 2)
    np.fill_diagonal(mi, 0.0)
    return mi


def fiedler_parts(w: np.ndarray):
    lap = np.diag(w.sum(axis=1)) - w
    vals, vecs = np.linalg.eigh(lap)
    f = vecs[:, 1]
    m1, m2 = np.where(f >= 0)[0], np.where(f < 0)[0]
    if len(m1) == 0 or len(m2) == 0:
        half = w.shape[0] // 2
        m1, m2 = np.arange(half), np.arange(half, w.shape[0])
    return m1, m2


def phi_authors(counts: np.ndarray, tau: int = 1):
    """Per-step emergence-capacity trajectory (length n_steps - tau)."""
    from phyid.calculate import calc_PhiID
    z = clr(counts.astype(float))
    mi = lag1_mi_matrix(z)
    m1, m2 = fiedler_parts(mi)
    a = z[:, m1].mean(axis=1)
    b = z[:, m2].mean(axis=1)
    atoms, _ = calc_PhiID(a, b, tau=tau, kind="gaussian", redundancy="MMI")
    ec = atoms["str"] + atoms["stx"] + atoms["sty"] + atoms["sts"]
    return np.asarray(ec), (m1, m2)


def main():
    for sub in ("runs_coarse", "runs"):
        outdir = ROOT / "results" / "phi_authors" / sub
        outdir.mkdir(parents=True, exist_ok=True)
        files = sorted((ROOT / "results" / sub).glob("run_*.npz"))
        for i, f in enumerate(files):
            d = np.load(f)
            ec, (m1, m2) = phi_authors(d["counts"])
            np.savez_compressed(outdir / f.name, phi=ec.astype(np.float32),
                                m1=m1, m2=m2)
            if (i + 1) % 25 == 0:
                print(f"{sub}: {i + 1}/{len(files)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
