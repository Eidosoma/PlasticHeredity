"""G3-ADJ: launch-state x convention factorial
(G3_ADJUDICATION.md; sealed). Fresh x conv-A cell runs on the SEALED
G3 seed keys and must reproduce the sealed pulse rows exactly."""

import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr

import sim
import features as Ft
import cohort
import run_intervention as RI
import run_steering as RS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g3_adjudication")
STEER_TAG = "steering-2026-08-13"
ADJ_TAG = "g3-adj-2026-08-18"
N_MAT = 24
REPS = [0, 1]
PULSES = [1, 2, 4, 8, 16, 32, 60]
FREE = 60
CELLS = [("fresh", "A"), ("fresh", "B"), ("natural", "A"),
         ("natural", "B")]
BOOT_N = 4096
BOOT_SEED = 41

_ENT_S = cohort.domain_entropy("confirmation", STEER_TAG)
_ENT_A = cohort.domain_entropy("confirmation", ADJ_TAG)


def cell_rng(launch, conv, cand_i, m, rep):
    if launch == "fresh" and conv == "A":
        return cohort._rng(_ENT_S, 22, 0, cand_i, m, rep)   # sealed keys
    cell_id = CELLS.index((launch, conv))
    return cohort._rng(_ENT_A, 32, cell_id, cand_i, m, rep)


def natural_state(cand, cand_i, m):
    beta, n0 = cohort.matrix_and_init(_ENT_S, m)
    rng = cohort._rng(_ENT_S, 7, cand_i, m, 0)
    tr = sim.run_fissions(n0, beta, cand, 60, rng)
    if tr["n_done"] < 60:
        return beta, None
    return beta, tr["daughters"][59]


def steer_cell(n0, beta, cand, rng, pulse, conv):
    """Pulse steering under convention A (edits 1..P-1, unedited
    anchor) or B (edits 1..P, post-edit anchor); then F60 release.
    Returns t07 (censored at FREE+1) or None on extinction."""
    n = n0.copy()
    hs = []
    for f in range(1, pulse + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            return None
        hs.append(float(step["H"][0]))
        n = step["final"]
        do_edit = (f < pulse) if conv == "A" else True
        if do_edit:
            X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
            n = RI.apply_swap(n, RS.marginal_swap(
                n, beta, X9, RI._BUNDLES[cand], -1))
    anchor = n.astype(float)
    rel = sim.run_fissions(n, beta, cand, FREE, rng)
    d = rel["daughters"].astype(float)
    ah = np.array([sim.cosine_h(d[i], anchor) for i in range(len(d))])
    return int(np.argmax(ah < 0.7)) + 1 if (ah < 0.7).any() \
        else FREE + 1


def unit(args):
    m, cand, rep, launch, conv = args
    cand_i = cohort.CANDIDATES.index(cand)
    if launch == "fresh":
        beta, s0 = cohort.matrix_and_init(_ENT_S, m)
    else:
        beta, s0 = natural_state(cand, cand_i, m)
        if s0 is None:
            return None
    out = {"matrix": m, "candidate": cand, "rep": rep,
           "launch": launch, "conv": conv, "t07": {}}
    for p in PULSES:
        rng = cell_rng(launch, conv, cand_i, m, rep)  # fresh per pulse
        t = steer_cell(s0, beta, cand, rng, p, conv)
        if t is not None:
            out["t07"][p] = t
    return out


def estimand(units):
    """Per-matrix rep-mean per pulse -> 7-point Spearman per matrix ->
    mean over matrices + whole-matrix bootstrap."""
    per = {}
    for u in units:
        for p, t in u["t07"].items():
            per.setdefault(u["matrix"], {}).setdefault(
                p, []).append(t)
    rhos = {}
    for m, d in per.items():
        ps = sorted(d)
        if len(ps) < 5:
            continue
        ys = [np.mean(d[p]) for p in ps]
        if np.std(ys) > 0:
            rhos[m] = spearmanr(ps, ys).correlation
    mats = sorted(rhos)
    if not mats:
        return (np.nan, np.nan, np.nan, {})
    rng = np.random.default_rng(BOOT_SEED)
    bs = [np.nanmean([rhos[mm] for mm in
                      rng.choice(mats, len(mats), True)])
          for _ in range(BOOT_N)]
    means = {p: float(np.mean([np.mean(per[m][p]) for m in per
                               if p in per[m]])) for p in PULSES}
    return (float(np.nanmean(list(rhos.values()))),
            float(np.nanquantile(bs, 0.025)),
            float(np.nanquantile(bs, 0.975)), means)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = _ENT_S

    t0 = time.time()
    jobs = [(m, c, r, la, cv) for c in cohort.CANDIDATES
            for m in range(N_MAT) for r in REPS
            for (la, cv) in CELLS]
    with Pool(12) as pool:
        units = [u for u in pool.map(unit, jobs) if u]
    print(f"G3-ADJ campaign in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "g3_adj_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    # replay gate: fresh x A must equal sealed g3_units pulse rows
    sealed = pickle.load(open(os.path.join(HERE, "results_g",
                                           "g3_units.pkl"), "rb"))
    skey = {(u["matrix"], u["candidate"], u["rep"]): u["pulse"]
            for u in sealed}
    mism = tot = 0
    for u in units:
        if (u["launch"], u["conv"]) != ("fresh", "A"):
            continue
        ref = skey[(u["matrix"], u["candidate"], u["rep"])]
        for p, t in u["t07"].items():
            if p in ref:
                tot += 1
                if ref[p]["t07"] != t:
                    mism += 1
    print(f"REPLAY GATE (fresh x A vs sealed G3): "
          f"{'PASS' if mism == 0 else 'FAIL'} "
          f"({mism} of {tot} rows differ)", flush=True)

    results = {"replay_gate": bool(mism == 0)}
    for cand in cohort.CANDIDATES:
        entry = {}
        print(f"\n=== G3-ADJ candidate {cand} ===")
        for (la, cv) in CELLS:
            cu = [u for u in units if u["candidate"] == cand
                  and u["launch"] == la and u["conv"] == cv]
            rho, lo, hi, means = estimand(cu)
            entry[f"{la}_{cv}"] = {"rho": rho, "ci": [lo, hi],
                                   "excludes0": bool(lo > 0 or hi < 0),
                                   "halflife_by_pulse": means}
            print(f"  {la:8s} conv {cv}: Spearman {rho:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]"
                  f"{'*' if lo > 0 or hi < 0 else ' '}  "
                  + " ".join(f"P{p}={means.get(p, float('nan')):.1f}"
                             for p in PULSES))
        results[cand] = entry

    with open(os.path.join(OUT, "g3_adj_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g3_adj_results.json"))


if __name__ == "__main__":
    main()
