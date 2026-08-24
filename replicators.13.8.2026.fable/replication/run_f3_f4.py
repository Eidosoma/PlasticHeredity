"""Phase F Stage 2: F3 release-delay ladder + F4 perturbation-type x
clock factorial, with the continuous restoring-force estimator.

F3 (does the freshly written state have a short-lived basin?):
- Written states: model_down regeneration, reps {0} (registered trim),
  24 matrices x 2 candidates = 48 states; single release trajectory per
  state (domain 9, identical to Phase E's path).
- Challenge at t_release in {0, 1, 2, 5, 10, 60} fissions along that
  path; arms {none, k4, k16, fission-like} x 32 branches x 24-fission
  recovery (branch streams domain 13). fission-like = an extra
  binomial(0.5) thinning, keeping one half (Kahana's natural
  perturbation; mass regrows in the first branch generation).
- Anchors: PRIMARY = the written composition (the "installed
  destination" question); SECONDARY = nearest atlas composome.
  Light natural controls at t in {0, 60} (noop-lineage states at
  fissions 60 and 120, anchored on themselves).

F4 (perturbation type x measurement clock, at t_release = 0):
- Types: none, fission-like, k8 swap, adversarial swap, radial-toward
  and radial-away (8 molecules moved toward/away the nearest composome,
  mass-preserving). Cosine displacement of each type reported
  (magnitude matching is measured, not enforced — registered).
- Clocks: within-growth (traced first generation, post-perturbation ->
  pre-fission, mass grid 5) and cross-generation (1, 3, 10, 24
  fissions).
- PRIMARY outcome: continuous radial drift
  R(r) = E[d(next) - d(current) | d(current) ~ r] against the atlas,
  per clock; registered margins: <= -0.01/fission (cross-gen) or
  <= -0.02/growth-cycle (within-growth) sustained over r in
  [0.05, 0.25] with CI excluding 0. Contraction C_t is deferred to F6
  (registered simplification: branches from one state start identical).
"""

import json
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim
import features as Ft
import cohort
import atlas as AT
import growth_trace as GT
import run_intervention as RI
import run_steering as RS
from run_release_challenge import classify_branch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_f")
FIG = os.path.join(OUT, "figures")
TAG = "steering-2026-08-13"
N_MAT = 24
DELAYS = [0, 1, 2, 5, 10, 60]
F3_ARMS = ["none", "k4", "k16", "fission"]
F4_TYPES = ["none", "fission", "k8", "adversarial",
            "radial_to", "radial_away"]
N_BR3 = 32
N_BR4 = 24
REC = 24
N_BOOT = 1024

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
GREEN = "#3E7D5B"
INK = "#33383D"

_ATLASES = None


def kswap_perturb(n, k, rng):
    removed = rng.multivariate_hypergeometric(n, k)
    add = np.bincount(rng.integers(0, sim.NG, k), minlength=sim.NG)
    return (n - removed + add).astype(np.int64)


def fission_perturb(n, rng):
    a = rng.binomial(n, 0.5)
    b = n - a
    pick = a if rng.random() < 0.5 else b
    return (pick if pick.sum() > 0 else a).astype(np.int64)


def radial_perturb(n, center, k, rng, toward=True):
    """Move k molecules toward (or away from) the unit composome
    center: remove from types where composition exceeds (resp. falls
    short of) the center profile, add to types where it falls short
    (resp. exceeds). Mass-preserving."""
    x = n / n.sum()
    gap = center / max(center.sum(), 1e-12) - x       # + = under-represented
    order_add = np.argsort(gap)[::-1] if toward else np.argsort(gap)
    order_rem = np.argsort(gap) if toward else np.argsort(gap)[::-1]
    ne = n.copy()
    added = 0
    for j in order_add:
        while added < k:
            ne[j] += 1
            added += 1
            break
    # distribute remaining adds across top-k gap types
    for j in order_add[1:k]:
        if added >= k:
            break
        ne[j] += 1
        added += 1
    removed = 0
    for i in order_rem:
        while removed < k and ne[i] > 0:
            take = min(int(ne[i]), k - removed)
            ne[i] -= take
            removed += take
            break
    assert ne.sum() == n.sum() + (added - removed)
    # exact mass fix (guaranteed small)
    while ne.sum() > n.sum():
        i = int(np.argmax(ne))
        ne[i] -= 1
    while ne.sum() < n.sum():
        ne[int(order_add[0])] += 1
    assert ne.sum() == n.sum() and (ne >= 0).all()
    return ne


def apply_perturbation(name, n, beta, cand, atl, rng, X9):
    if name == "none":
        return n
    if name == "fission":
        return fission_perturb(n, rng)
    if name == "k4":
        return kswap_perturb(n, 4, rng)
    if name == "k8":
        return kswap_perturb(n, 8, rng)
    if name == "k16":
        return kswap_perturb(n, 16, rng)
    if name == "adversarial":
        swap = RS.marginal_swap(n, beta, X9, RI._BUNDLES[cand], +1)
        return RI.apply_swap(n, swap)
    center = atl["centers"][AT.nearest_center(n, atl)]
    return radial_perturb(n, center, 8, rng,
                          toward=(name == "radial_to"))


def branch_traces(s0, beta, cand, cand_i, key, n_br, anchor_w, atl,
                  traced=False):
    """Run n_br branches; per branch return anchor-similarity trace,
    atlas-distance trace, inheritance flags, end top1, end center; if
    traced, also within-growth (start,end) atlas distances of the first
    generation."""
    aw = anchor_w.astype(float)
    outs = []
    for b in range(n_br):
        rb = cohort._rng(RI._ENT, 13, cand_i, *key, b)
        if traced:
            tr = GT.traced_run_fissions(s0, beta, cand, REC, rb, 80,
                                        grid_step=5)
            recs = tr["recs"]
            daughters = [r["daughter"] for r in recs]
            inh = np.array([r["inherited"] for r in recs])
            wg = None
            if recs:
                sm, sc = recs[0]["snaps"][0]
                em, ec = recs[0]["snaps"][-1]
                wg = (AT.dist(sc, atl), AT.dist(ec, atl))
        else:
            br = sim.run_fissions(s0, beta, cand, REC, rb)
            daughters = list(br["daughters"])
            inh = br["inherited"]
            wg = None
        d = np.array([AT.dist(dd, atl) for dd in daughters])
        ah = np.array([sim.cosine_h(dd.astype(float), aw)
                       for dd in daughters])
        ft = 0.0
        if daughters:
            last = daughters[-1].astype(float)
            ft = float((last / max(last.sum(), 1)).max())
        outs.append({"ah": ah, "datl": d, "inh": inh, "final_top1": ft,
                     "end_center": AT.nearest_center(daughters[-1], atl)
                     if daughters else -1, "wg": wg})
    return outs


def unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    cfg = "frozen02" if cand == "02" else "frozen03"
    atl = _ATLASES[(cfg, m)]

    # written state (rep 0) + its release path (identical to Phase E)
    hs_w, holder = [], {}
    def wlog(f, n, swap, H, updates):
        hs_w.append(H)
        holder["n"] = n
    RS.steer_lineage(n0, beta, cand, cand_i, m, 0, "model_down", log=wlog)
    written = holder["n"]
    rr = cohort._rng(RI._ENT, 9, cand_i, m, 0)
    rel = sim.run_fissions(written, beta, cand, 60, rr)
    states_at = {0: written}
    for t in DELAYS[1:]:
        states_at[t] = rel["daughters"][t - 1] if t <= rel["n_done"] \
            else rel["final"]

    # natural lineage for controls
    rn = cohort._rng(RI._ENT, 7, cand_i, m, 0)
    nat = sim.run_fissions(n0, beta, cand, 120, rn)
    nat_at = {0: nat["daughters"][59], 60: nat["daughters"][119]}

    res = {"matrix": m, "candidate": cand, "f3": {}, "f3_nat": {},
           "f4": {}, "disp": {}}

    def X9_for(hs, g, mass):
        return Ft.direct9(g, 100, np.array(hs), mass)

    # ---------------- F3 ladder --------------------------------------
    for ti, t in enumerate(DELAYS):
        s = states_at[t]
        hs_full = hs_w + list(rel["H"][:t])
        X9 = X9_for(hs_full, len(hs_full), int(s.sum()))
        for ai, arm in enumerate(F3_ARMS):
            pr = cohort._rng(RI._ENT, 11, cand_i, m, 0, ti, ai)
            s0 = apply_perturbation(arm, s, beta, cand, atl, pr, X9)
            outs = branch_traces(s0, beta, cand, cand_i,
                                 (m, 0, ti, ai), N_BR3, written, atl)
            cls = [classify_branch(o["ah"], o["inh"], o["final_top1"])
                   for o in outs]
            res["f3"][(t, arm)] = {
                "held_ret": float(np.mean([c in ("held", "returned")
                                           for c in cls])),
                "mode": float(np.mean([c == "mode_recovered"
                                       for c in cls])),
                "end_atl_dist": float(np.mean(
                    [o["datl"][-1] for o in outs if len(o["datl"])]
                    or [np.nan])),
                "drift_pairs": [(float(o["datl"][i]),
                                 float(o["datl"][i + 1]))
                                for o in outs
                                for i in range(len(o["datl"]) - 1)][::4],
            }
    for t in (0, 60):
        s = nat_at[t]
        X9 = X9_for(list(nat["H"][:60 + t]), 60 + t, int(s.sum()))
        for ai, arm in enumerate(F3_ARMS):
            pr = cohort._rng(RI._ENT, 11, cand_i, m, 1, t, ai)
            s0 = apply_perturbation(arm, s, beta, cand, atl, pr, X9)
            outs = branch_traces(s0, beta, cand, cand_i,
                                 (m, 1, t, ai), N_BR3, s, atl)
            cls = [classify_branch(o["ah"], o["inh"], o["final_top1"])
                   for o in outs]
            res["f3_nat"][(t, arm)] = {
                "held_ret": float(np.mean([c in ("held", "returned")
                                           for c in cls]))}

    # ---------------- F4 factorial at t=0 ----------------------------
    s = written
    X9 = X9_for(hs_w, len(hs_w), int(s.sum()))
    for pi, ptype in enumerate(F4_TYPES):
        pr = cohort._rng(RI._ENT, 14, cand_i, m, pi)
        s0 = apply_perturbation(ptype, s, beta, cand, atl, pr, X9)
        res["disp"][ptype] = float(1 - sim.cosine_h(
            s0.astype(float), s.astype(float)))
        outs = branch_traces(s0, beta, cand, cand_i,
                             (m, 2, pi, 0), N_BR4, written, atl,
                             traced=True)
        horizon_ah = {h: float(np.mean([o["ah"][h - 1] for o in outs
                                        if len(o["ah"]) >= h]))
                      for h in (1, 3, 10, 24)}
        wg = [o["wg"] for o in outs if o["wg"] is not None]
        res["f4"][ptype] = {
            "anchor_at": horizon_ah,
            "wg_pairs": [(float(a), float(b)) for a, b in wg],
            "drift_pairs": [(float(o["datl"][i]), float(o["datl"][i + 1]))
                            for o in outs
                            for i in range(len(o["datl"]) - 1)][::4],
        }
    return res


def radial_drift(pairs, lo=0.05, hi=0.25):
    P = np.array(pairs)
    if not len(P):
        return np.nan, 0
    m = (P[:, 0] >= lo) & (P[:, 0] <= hi)
    if not m.any():
        return np.nan, 0
    return float(np.mean(P[m, 1] - P[m, 0])), int(m.sum())


def main():
    global _ATLASES
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(OUT, "atlases.pkl"), "rb") as f:
        _ATLASES = pickle.load(f)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"F3/F4 campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        mats = np.array([u["matrix"] for u in cu])
        entry = {"f3": {}, "f3_nat": {}, "f4": {}, "disp": {},
                 "drift": {}}
        rng = np.random.default_rng(999)
        for t in DELAYS:
            for arm in F3_ARMS:
                v = np.array([u["f3"][(t, arm)]["held_ret"] for u in cu])
                entry["f3"][f"{t}_{arm}"] = {
                    "held_ret": float(v.mean()),
                    "ci": RI.boot_lower(v, mats, rng, n=N_BOOT),
                    "mode": float(np.mean([u["f3"][(t, arm)]["mode"]
                                           for u in cu])),
                }
        for t in (0, 60):
            for arm in F3_ARMS:
                v = np.array([u["f3_nat"][(t, arm)]["held_ret"]
                              for u in cu])
                entry["f3_nat"][f"{t}_{arm}"] = float(v.mean())
        allpairs_cross = [p for u in cu for key in u["f3"]
                          for p in u["f3"][key]["drift_pairs"]]
        entry["drift"]["cross_gen"] = radial_drift(allpairs_cross)
        wg_all = [p for u in cu for t_ in u["f4"].values()
                  for p in t_["wg_pairs"]]
        entry["drift"]["within_growth"] = radial_drift(wg_all)
        for ptype in F4_TYPES:
            entry["f4"][ptype] = {
                "anchor_at": {k: float(np.mean(
                    [u["f4"][ptype]["anchor_at"][k] for u in cu]))
                    for k in (1, 3, 10, 24)},
            }
            entry["disp"][ptype] = float(np.mean(
                [u["disp"][ptype] for u in cu]))
        results[cand] = entry

        print(f"\n=== F3 candidate {cand} (held+returned vs WRITTEN "
              f"anchor) ===")
        hdr = f"{'delay':>6s}" + "".join(f"{a:>10s}" for a in F3_ARMS)
        print(hdr)
        for t in DELAYS:
            row = f"{t:6d}"
            for arm in F3_ARMS:
                row += f"{entry['f3'][f'{t}_{arm}']['held_ret']:10.2f}"
            print(row)
        print("natural controls (anchor = self): "
              + ", ".join(f"t={t} {arm}: "
                          f"{entry['f3_nat'][f'{t}_{arm}']:.2f}"
                          for t in (0, 60) for arm in ("none", "k16")))
        print(f"radial drift (atlas): cross-gen "
              f"{entry['drift']['cross_gen'][0]:+.4f}/fission "
              f"(n={entry['drift']['cross_gen'][1]}) | within-growth "
              f"{entry['drift']['within_growth'][0]:+.4f}/cycle "
              f"(n={entry['drift']['within_growth'][1]})")
        print("F4 anchor sim at horizons 1/3/10/24 and displacement:")
        for ptype in F4_TYPES:
            a = entry["f4"][ptype]["anchor_at"]
            print(f"  {ptype:12s} disp {entry['disp'][ptype]:.3f} | "
                  + " ".join(f"{a[k]:.2f}" for k in (1, 3, 10, 24)))

    with open(os.path.join(OUT, "f3_f4_results.json"), "w") as f:
        json.dump({c: {k: v for k, v in results[c].items()}
                   for c in results}, f, indent=2, default=str)

    # figure: F3 ladder
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    colors = {"none": BLUE, "k4": GREEN, "k16": AMBER, "fission": GRAY}
    for j, cand in enumerate(cohort.CANDIDATES):
        ax = axes[j]
        for arm in F3_ARMS:
            ys = [results[cand]["f3"][f"{t}_{arm}"]["held_ret"]
                  for t in DELAYS]
            ax.plot(range(len(DELAYS)), ys, "o-", color=colors[arm],
                    lw=1.3, ms=4, label=arm)
        ax.set_xticks(range(len(DELAYS)), [str(t) for t in DELAYS])
        ax.set_xlabel("release delay before challenge (fissions)")
        ax.set_title(f"Candidate {cand}")
        if j == 0:
            ax.set_ylabel("held+returned vs written anchor")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("F3: no delay rescues the written anchor — "
                 "held/returned by challenge delay")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_f3_delay_ladder.png"))
    plt.close(fig)
    print("\nwritten:", os.path.join(OUT, "f3_f4_results.json"))


if __name__ == "__main__":
    main()
