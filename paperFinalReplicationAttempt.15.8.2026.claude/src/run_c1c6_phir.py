"""C1 (aggregate trend) + C6 (spike geometry) — and the full C1-C4/C6
battery — under code-faithful Phi_R and the repo's 3-atom "emergence",
via the instrument-agnostic analysis_corr.analyze().

Modes:
  coarse (default): project Phi_R / emergence3 from the cached 16-atom
    stacks (results/lattice_features.pkl). Printed-Psi comparator is
    already on record in results/corr_stats_coarse.json.
  fine: compute phi_r_code_local on results/runs/ (the fine universe,
    regenerated with run_sims.py fine), Pool 12; adds sign regime and
    consistency autocorrelation. -> results/phir_fine.json
"""

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from analysis_corr import analyze
from phi_r_code import ORDER, PHIR_ATOMS, R, U0, U1, S, phi_r_code_local

ROOT = Path(__file__).parent.parent
PHIR_IDX = [ORDER.index(a) for a in PHIR_ATOMS]
E3_IDX = [ORDER.index(a) for a in ((S, S), (S, U0), (S, U1))]


def coarse():
    with open(ROOT / "results" / "lattice_features.pkl", "rb") as fh:
        cached = pickle.load(fh)
    out = {}
    for name, idx in (("phir", PHIR_IDX), ("emergence3", E3_IDX)):
        runs = [{"phi": r["atoms16"][:, idx].sum(axis=1),
                 "sr": r["sr"][1:], "sr_full": r["sr"], "seed": r["seed"]}
                for r in cached]
        res, _, _ = analyze(runs)
        out[name] = res
        print(f"=== {name} ===")
        print(json.dumps(res, indent=2, default=float), flush=True)
    (ROOT / "results" / "c1c6_phir.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("written results/c1c6_phir.json")


def _fine_one(f):
    d = np.load(f)
    counts = d["counts"].astype(float)
    return {"phi": np.asarray(phi_r_code_local(counts, "full"), float),
            "sr": d["sr"][1:], "sr_full": d["sr"],
            "seed": int(Path(f).stem.split("_")[1])}


def fine():
    files = [str(f) for f in
             sorted((ROOT / "results" / "runs").glob("run_*.npz"))]
    with Pool(12) as pool:
        runs = pool.map(_fine_one, files)
    runs = [r for r in runs if np.isfinite(r["phi"]).any()]
    res, _, _ = analyze(runs)
    means = np.array([np.nanmean(r["phi"]) for r in runs])
    autoc = [float(np.corrcoef(r["phi"][:-1], r["phi"][1:])[0, 1])
             for r in runs]
    from scipy import stats as _st
    srp = np.array([r["sr_full"].mean() for r in runs])
    rho, p = _st.spearmanr(means, srp)
    out = {
        "battery": res,
        "sign_regime": {
            "phir_run_mean": f"{means.mean():+.4f}±{means.std():.4f}",
            "phir_runs_positive": int((means > 0).sum()),
            "n_runs": len(runs)},
        "consistency_phi_autocorr":
            f"{np.nanmean(autoc):.3f}±{np.nanstd(autoc):.3f}",
        "polarity_runmean_phir_vs_sr_prob":
            {"rho": float(rho), "p": float(p)},
    }
    print(json.dumps(out, indent=2, default=float), flush=True)
    (ROOT / "results" / "phir_fine.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("written results/phir_fine.json")


if __name__ == "__main__":
    fine() if "fine" in sys.argv[1:] else coarse()
