"""Alternative Phi_r trajectory estimators, for the C5/C7 recovery sweep.

Variants (all on CLR-transformed coarse-universe compositions):
  local      — baseline: multivariate parts, one Gaussian fit/run, local values
  win100     — windowed static Phi_r, window 100, stride 1 (right-aligned)
  scalar2    — MIB parts coarse-grained to 2 scalar series (mean of CLR
               channels), then local Phi_r on the 2-D system
  phyid_mmi  — phyid ΦID atoms on the same 2-scalar system, MMI redundancy:
               Phi_r(t) = Σ synergy-source atoms − Σ redundancy-source atoms
  tau2/tau4  — baseline with lag 2 / 4
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from phi import (clr, zscore, min_information_bipartition, phi_r_local,
                 phi_r_window, shrunk_cov, _local_mi)

ROOT = Path(__file__).parent.parent


def scalar_parts(counts, mib_seed=0):
    z = zscore(clr(counts))
    m1, m2 = min_information_bipartition(z, seed=mib_seed)
    a = zscore(z[:, m1].mean(axis=1, keepdims=True))
    b = zscore(z[:, m2].mean(axis=1, keepdims=True))
    return a.ravel(), b.ravel()


def phi_scalar2_local(counts, mib_seed=0, lag=1):
    a, b = scalar_parts(counts, mib_seed)
    z = np.stack([a, b], axis=1)
    joint = np.hstack([z[:-lag], z[lag:]])
    joint = joint - joint.mean(axis=0)
    cov = shrunk_cov(joint, shrink=1e-3)
    past, future = np.arange(2), np.arange(2, 4)
    return (_local_mi(joint, cov, past, future)
            - _local_mi(joint, cov, [0], future)
            - _local_mi(joint, cov, [1], future))


def phi_phyid(counts, mib_seed=0, tau=1):
    from phyid.calculate import calc_PhiID
    a, b = scalar_parts(counts, mib_seed)
    atoms, _ = calc_PhiID(a, b, tau=tau, kind="gaussian", redundancy="MMI")
    syn = atoms["str"] + atoms["stx"] + atoms["sty"] + atoms["sts"]
    red = atoms["rtr"] + atoms["rtx"] + atoms["rty"] + atoms["rts"]
    return np.asarray(syn - red)


def windowed(counts, mib_seed=0, window=100):
    z = zscore(clr(counts))
    m1, m2 = min_information_bipartition(z, seed=mib_seed)
    vals = np.full(z.shape[0] - 1, np.nan)
    for end in range(window, z.shape[0]):
        vals[end - 1] = phi_r_window(z[end - window:end], m1, m2, lag=1,
                                     shrink=0.05)
    first = np.flatnonzero(~np.isnan(vals))[0]
    vals[:first] = vals[first]
    return vals


def main():
    outdir = ROOT / "results" / "phi_variants"
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz"))
    for i, f in enumerate(files):
        d = np.load(f)
        counts = d["counts"].astype(float)
        seed = int(f.stem.split("_")[1])
        out = {"local": d["phi"].astype(np.float32)}
        out["win100"] = windowed(counts, seed).astype(np.float32)
        out["scalar2"] = phi_scalar2_local(counts, seed).astype(np.float32)
        out["phyid_mmi"] = phi_phyid(counts, seed).astype(np.float32)
        for tau in (2, 4):
            phi, _ = phi_r_local(counts, lag=tau, mib_seed=seed)
            out[f"tau{tau}"] = phi.astype(np.float32)
        np.savez_compressed(outdir / f.name, **out)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    # correlation between variants on one run, as a coherence check
    d = np.load(outdir / files[0].name)
    n = min(len(d[k]) for k in d.files)
    for k in d.files:
        if k == "local":
            continue
        c = np.corrcoef(d["local"][:n], d[k][:n])[0, 1]
        print(f"corr(local, {k}) = {c:.3f}")


if __name__ == "__main__":
    main()
