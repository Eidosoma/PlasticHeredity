"""Phase I: reciprocal causal bridge between reconstructed Phi-r and
plastic heredity (preregistered in PHIR_BRIDGE.md; domain 27).

Six arms, identical CRN action panels and growth streams; Plastic-H
outcomes measured over 60 steered fissions; realized Phi-r measured
on traced within-growth snapshot series of fissions 41-60 (downstream
of edits, never used for selection).
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
from scipy.stats import spearmanr

import sim
import features as Ft
import cohort
import phir
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_bridge")
TAG = "phir-bridge-2026-08-17"
DOMAIN = 27
N_MAT = 24
REPS = [0, 1]
STEER = 60
PANEL = 24
PHI_FROM = 41                     # traced fissions 41..60
ARMS = ["ph_stab", "ph_destab", "phir_max", "phir_min", "random",
        "noop"]
BOOT_N = 2048
BOOT_SEED = 13

_ENT = cohort.domain_entropy("confirmation", TAG)


def _r(*key):
    return cohort._rng(_ENT, DOMAIN, *key)


def traced_step(n, beta, cand, rng, record):
    """One growth+fission cycle recording the composition after EVERY
    molecular update (event for 02, Poisson step for 03) into
    `record`; RNG call order identical to sim.run_fissions."""
    if cand == "02":
        nn = n.copy()
        c = beta @ nn
        total = int(nn.sum())
        events = 0
        while total < sim.NMAX:
            join, leave = sim.event_rates(nn, c, total)
            mu = sim._sample_categorical(
                np.concatenate([join, leave]), rng)
            if mu < sim.NG:
                nn[mu] += 1
                c += beta[:, mu]
                total += 1
            else:
                k = mu - sim.NG
                nn[k] -= 1
                c -= beta[:, k]
                total -= 1
                if total == 0:
                    return None, np.nan
            record.append(nn.copy())
            events += 1
            if events >= 40 * sim.MAXSTEPS:
                break
        grown = nn
    else:
        nn = n.copy()
        total = int(nn.sum())
        steps = 0
        while total < sim.NMAX and steps < sim.MAXSTEPS:
            c = beta @ nn
            join, leave = sim.event_rates(nn, c, total)
            s = join.sum() + leave.sum()
            dt = sim.EVENTS_PER_STEP / s
            joins = rng.poisson(join * dt)
            leaves = np.minimum(rng.poisson(leave * dt), nn)
            nn = nn + joins - leaves
            total = int(nn.sum())
            steps += 1
            record.append(nn.copy())
            if total == 0:
                return None, np.nan
        grown = nn
    if grown.sum() < 2:
        return None, np.nan
    parent = grown
    if cand == "02":
        ca, cb = sim._split_equal(parent, rng)
        d = ca
    else:
        ca, cb = sim._split_binomial(parent, rng)
        d = ca if rng.random() < 0.5 else cb
        if d.sum() == 0:
            d = ca if ca.sum() > 0 else cb
    h = sim.cosine_h(parent.astype(float), d.astype(float))
    return d, h


def draw_panel(n, rng):
    """PANEL legal mass-preserving swaps (remove present i, add j)."""
    present = np.where(n > 0)[0]
    panel = []
    for _ in range(PANEL):
        i = int(present[rng.integers(len(present))])
        j = int(rng.integers(sim.NG - 1))
        panel.append((i, j + 1 if j >= i else j))
    return panel


def v2_scores(n, beta, cand, hs, f, panel):
    X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
    rows = [Ft.graph_state_195(RI.apply_swap(n, sw), beta)
            for sw in panel]
    return np.asarray(RI.score_states(RI._BUNDLES[cand], X9, rows),
                      dtype=float)


def surr_scores(n, beta, panel):
    return np.array([phir.phi_r_surrogate(RI.apply_swap(n, sw), beta)
                     for sw in panel])


def unit(args):
    m, cand, rep, arm = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_r(0, m))
    n = sim.make_initial_state(_r(1, m))
    rng = _r(2, cand_i, m, rep)          # CRN: arm-independent
    hs, record, agree = [], [], []
    for f in range(1, STEER + 1):
        if f >= PHI_FROM:
            d, h = traced_step(n, beta, cand, rng, record)
            if d is None:
                break
            hs.append(h)
            n = d
        else:
            step = sim.run_fissions(n, beta, cand, 1, rng)
            if step["n_done"] < 1:
                break
            hs.append(float(step["H"][0]))
            n = step["final"]
        if f == STEER:
            break
        panel = draw_panel(n, _r(3, cand_i, m, rep, f))
        if arm == "noop":
            if f % 5 == 0:               # agreement log, every 5th
                v2 = v2_scores(n, beta, cand, hs, f, panel)
                su = surr_scores(n, beta, panel)
                if np.std(v2) > 0 and np.std(su) > 0:
                    agree.append((
                        float(spearmanr(-v2, su).correlation),
                        int(int(np.argmin(v2)) == int(np.argmax(su)))))
            continue
        if arm == "random":
            pick = panel[int(_r(4, cand_i, m, rep, f).integers(PANEL))]
        elif arm in ("ph_stab", "ph_destab"):
            sc = v2_scores(n, beta, cand, hs, f, panel)
            pick = panel[int(np.argmin(sc) if arm == "ph_stab"
                             else np.argmax(sc))]
        else:
            sc = surr_scores(n, beta, panel)
            pick = panel[int(np.argmax(sc) if arm == "phir_max"
                             else np.argmin(sc))]
        n = RI.apply_swap(n, pick)
    hs = np.array(hs)
    inh = hs > sim.H_THRESH
    runs, cur = [0], 0
    for v in inh:
        cur = cur + 1 if v else 0
        runs.append(cur)
    phi = phir.phi_r_series(np.array(record, dtype=np.float64)) \
        if record else np.nan
    return {"matrix": m, "candidate": cand, "rep": rep, "arm": arm,
            "inherit": float(inh.mean()) if len(inh) else np.nan,
            "breaks": int((~inh).sum()),
            "longest_run": int(max(runs)),
            "f12": float(Ft.joint_break_run3(inh[48:60]))
            if len(inh) >= 60 else np.nan,
            "phi": float(phi),
            "n_phi": len(record),
            "agree": agree}


def boot_pairs(units, arm_a, arm_b, key):
    per = {}
    for u in units:
        per.setdefault((u["matrix"], u["rep"]), {})[u["arm"]] = u
    diffs = {}
    for (m, rep), d in per.items():
        if arm_a in d and arm_b in d:
            a, b = d[arm_a][key], d[arm_b][key]
            if np.isfinite(a) and np.isfinite(b):
                diffs.setdefault(m, []).append(a - b)
    means = {m: np.mean(v) for m, v in diffs.items()}
    mats = list(means)
    rng = np.random.default_rng(BOOT_SEED)
    bs = [np.mean([means[mm] for mm in
                   rng.choice(mats, size=len(mats), replace=True)])
          for _ in range(BOOT_N)]
    return (float(np.mean(list(means.values()))),
            float(np.quantile(bs, 0.025)),
            float(np.quantile(bs, 0.975)))


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = _ENT

    t0 = time.time()
    jobs = [(m, c, r, a) for c in cohort.CANDIDATES
            for m in range(N_MAT) for r in REPS for a in ARMS]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    print(f"campaign in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "phir_bridge_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        entry = {"arms": {}, "tests": {}}
        for arm in ARMS:
            au = [u for u in cu if u["arm"] == arm]
            entry["arms"][arm] = {
                k: float(np.nanmean([u[k] for u in au]))
                for k in ("inherit", "breaks", "longest_run", "f12",
                          "phi")}
            entry["arms"][arm]["n_phi"] = float(np.mean(
                [u["n_phi"] for u in au]))
        for name, a, b, key in (
                ("T1_phstab_minus_phdestab_PHI", "ph_stab",
                 "ph_destab", "phi"),
                ("T2_phirmax_minus_phirmin_INHERIT", "phir_max",
                 "phir_min", "inherit"),
                ("T2_phirmax_minus_phirmin_BREAKS", "phir_max",
                 "phir_min", "breaks"),
                ("T2_phirmax_minus_phirmin_LONGRUN", "phir_max",
                 "phir_min", "longest_run"),
                ("VALIDITY_phstab_minus_phdestab_INHERIT", "ph_stab",
                 "ph_destab", "inherit"),
                ("MANIP_phirmax_minus_phirmin_PHI", "phir_max",
                 "phir_min", "phi"),
                ("T4_random_minus_noop_INHERIT", "random", "noop",
                 "inherit"),
                ("T4_random_minus_noop_PHI", "random", "noop", "phi")):
            d, lo, hi = boot_pairs(cu, a, b, key)
            entry["tests"][name] = {"diff": d, "ci": [lo, hi],
                                    "excludes0": bool(lo > 0
                                                      or hi < 0)}
        ag = [x for u in cu if u["arm"] == "noop"
              for x in u["agree"]]
        entry["T3_agreement"] = {
            "mean_spearman": float(np.mean([a for a, _ in ag]))
            if ag else None,
            "top_choice_rate": float(np.mean([t for _, t in ag]))
            if ag else None,
            "n_panels": len(ag)}
        results[cand] = entry

        print(f"\n=== Phase I candidate {cand} ===")
        print(f"{'arm':10s} {'inherit':>8s} {'breaks':>7s} "
              f"{'longrun':>8s} {'f12':>6s} {'phi':>8s}")
        for arm in ARMS:
            a = entry["arms"][arm]
            print(f"{arm:10s} {a['inherit']:8.3f} {a['breaks']:7.2f} "
                  f"{a['longest_run']:8.1f} {a['f12']:6.3f} "
                  f"{a['phi']:8.4f}")
        for k, v in entry["tests"].items():
            print(f"{k}: {v['diff']:+.4f} "
                  f"CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
                  + ("  *" if v["excludes0"] else ""))
        t3 = entry["T3_agreement"]
        print(f"T3 agreement: spearman {t3['mean_spearman']} "
              f"top-choice {t3['top_choice_rate']} "
              f"(n={t3['n_panels']})")

    with open(os.path.join(OUT, "phir_bridge_results.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:",
          os.path.join(OUT, "phir_bridge_results.json"))


if __name__ == "__main__":
    main()
