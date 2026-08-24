"""Zeros-handling fork for the code-faithful Phi_R: the paper's stated
CLR preprocessing is undefined at zero counts and no pseudocount is
stated; the authors' public repo contains no CLR at all. Four readings:

  anchor   — pseudocount 1.0, drop last component (this repo's
             registered choice; the on-record Phi_R numbers).
  fable    — pseudocount 0.5, no component drop (sister-replication
             convention).
  literal  — NO pseudocount, CLR applied verbatim: log(0) = -inf makes
             every row's geometric mean -inf, so every row with any
             zero degenerates. Emulated exactly, not "fixed" — the
             expected outcome (total degeneracy) is itself the finding.
  presentonly — a plausible unstated driver fix in the style of their
             dead-channel filter: restrict to species with nonzero
             counts at EVERY step of the run, then CLR without
             pseudocount.

Outcomes per variant: sign regime, C3 headline numbers, Pearson of run
means vs the anchor. -> results/zeros_fork.json
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

from phi_r_code import (DEAD_EPS, PHIR_ATOMS, fiedler_bipartition,
                        local_phi_id, mi_matrix_lag1)
from reanalyze_authors import headline

ROOT = Path(__file__).parent.parent
VARIANTS = ("anchor", "fable", "literal", "presentonly")


def clr_variant(counts, variant):
    """Returns (T x k) CLR-ish matrix per the variant's reading."""
    counts = np.asarray(counts, dtype=np.float64)
    if variant == "anchor":
        x = counts + 1.0
        x = x / x.sum(axis=1, keepdims=True)
        z = np.log(x)
        z = z - z.mean(axis=1, keepdims=True)
        return z[:, :-1]
    if variant == "fable":
        x = counts + 0.5
        x = x / x.sum(axis=1, keepdims=True)
        z = np.log(x)
        return z - z.mean(axis=1, keepdims=True)
    if variant == "literal":
        x = counts / counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log(x)                     # -inf at zeros, verbatim
            return z - z.mean(axis=1, keepdims=True)
    if variant == "presentonly":
        keep = (counts > 0).all(axis=0)
        if keep.sum() < 3:
            return np.zeros((counts.shape[0], 0))
        x = counts[:, keep]
        x = x / x.sum(axis=1, keepdims=True)
        z = np.log(x)
        return z - z.mean(axis=1, keepdims=True)
    raise ValueError(variant)


def phi_r_from_clr(Zrows):
    """phi_r_code_local's post-CLR pipeline on a (T x k) matrix."""
    if Zrows.shape[0] < 20 or Zrows.shape[1] == 0:
        return np.full(max(Zrows.shape[0] - 1, 0), np.nan)
    Z = Zrows.T
    with np.errstate(invalid="ignore"):
        Z = Z[np.nan_to_num(Z.std(axis=1), nan=0.0) > DEAD_EPS]
        Z = Z[np.isfinite(Z).all(axis=1)]     # their mask drops non-finite
    if Z.shape[0] < 3:
        return np.full(Zrows.shape[0] - 1, np.nan)
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1, keepdims=True)
    a, b = fiedler_bipartition(mi_matrix_lag1(Z))
    if not a or not b:
        return np.full(Zrows.shape[0] - 1, np.nan)
    edge = np.vstack([np.nanmean(Z[a], axis=0), np.nanmean(Z[b], axis=0)])
    pi = local_phi_id(edge)
    return np.sum([pi[at] for at in PHIR_ATOMS], axis=0)


def _one(args):
    counts, sr = args
    row = {}
    for v in VARIANTS:
        row[v] = np.asarray(phi_r_from_clr(clr_variant(counts, v)), float)
    row["sr"] = sr
    row["counts"] = counts
    return row


def main():
    with open(ROOT / "results" / "lattice_features.pkl", "rb") as fh:
        cached = pickle.load(fh)
    with Pool(12) as pool:
        rows = pool.map(_one, [(r["counts"], r["sr"]) for r in cached])
    out = {}
    anchor_means = np.array([np.nanmean(r["anchor"]) for r in rows])
    for v in VARIANTS:
        means = np.array([np.nanmean(r[v]) for r in rows])
        finite = np.isfinite(means)
        entry = {
            "runs_defined": int(finite.sum()),
            "run_mean": (f"{np.nanmean(means):+.4f}"
                         f"±{np.nanstd(means):.4f}") if finite.any()
                        else "undefined",
            "runs_positive": int((means > 0).sum()),
        }
        ok = finite & np.isfinite(anchor_means)
        entry["runmean_pearson_vs_anchor"] = (
            round(float(np.corrcoef(means[ok], anchor_means[ok])[0, 1]), 4)
            if ok.sum() > 2 else None)
        hl_rows = [{"phi": r[v], "sr": r["sr"][1:], "counts": r["counts"]}
                   for r in rows if np.isfinite(r[v]).any()]
        entry["headline"] = headline(hl_rows) if hl_rows else "undefined"
        out[v] = entry
        print(f"=== {v} ===")
        print(json.dumps(entry, indent=2, default=float), flush=True)
    (ROOT / "results" / "zeros_fork.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("written results/zeros_fork.json")


if __name__ == "__main__":
    main()
