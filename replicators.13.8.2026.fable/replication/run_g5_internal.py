"""Phase G5 (EXPLORATORY, labeled): the internalized-controller ladder
(preregistered in PHASE_G.md; domain 24).

Can information-restricted local rules replace the global v2
controller?

  L0  memoryless local rule: the frozen C3 influence rule
      (stabilize = remove the LEAST catalytically influential present
      type, add the MOST influential)
  L1  L0 gated by one bit of memory (edit only after a break)
  L2  L0 gated by streak (edit only while trailing inherited run < 3)
  L3  distilled transparent policy: two depth-3 decision trees on
      per-type LOCAL features (abundance, out-influence percentile,
      in-boost percentile, presence) imitating frozen-v2 edit choices
      on regenerated 25x dev-matrix states; frozen before any
      confirmation lineage runs

Compared against v2_down (full controller), random, noop on:
  maintenance    (inheritance over 60 fissions, home regime)
  recovery       (k8 perturbation at fission 30, post-perturb
                  inheritance; CRN with the maintenance lineage)
  generalization (three transfer regimes, maintenance only)

KINETIC PROTOTYPE (model extension, appendix): growth kinetics with
leave rates damped by catalytic out-influence percentile —
leave_i / (1 + lambda * pct_i), lambda in {0.1, 0.3}, percentiles
refreshed once per growth phase from the post-fission composition.
No editor at all; each candidate keeps its own growth/fission contract
(02 Gillespie + hypergeometric split + first daughter; 03
vector-Poisson + binomial split + uniform daughter). lambda=0 control
uses the frozen sim path verbatim. Frozen sim untouched.
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
from sklearn.tree import DecisionTreeClassifier

import sim
import features as Ft
import cohort
import run_intervention as RI
import run_steering as RS
from run_f3_f4 import kswap_perturb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
TAG = "steering-2026-08-13"
DEV_TAG = "25x-2026-08-13"
N_MAT = 24
REPS = [0, 1]
STEER = 60
PERTURB_AT = 30
POLICIES = ["L0", "L1", "L2", "L3", "v2_down", "random", "noop"]
REGIMES = [(-4.0, 5.0), (-3.0, 4.0), (-5.0, 4.0)]
LAMBDAS = [0.1, 0.3]

_TREES = None      # distilled (remove_tree, add_tree) per candidate


def type_features(n, beta):
    """Per-type LOCAL features: abundance share, out-influence
    percentile, in-boost percentile, present flag."""
    N = max(int(n.sum()), 1)
    x = n / N
    infl = x @ beta
    boost = (beta @ n) / N
    rp = np.argsort(np.argsort(infl)) / (sim.NG - 1)
    bp = np.argsort(np.argsort(boost)) / (sim.NG - 1)
    return np.column_stack([x, rp, bp, (n > 0).astype(float)])


def rule_swap_stabilize(n, beta):
    """L0: remove least-influential present type, add most-influential
    (the frozen C3 influence rule, stabilizing orientation)."""
    x = n / max(n.sum(), 1)
    infl = x @ beta
    present = np.where(n > 0)[0]
    i = int(present[np.argmin(infl[present])])
    order = np.argsort(infl)[::-1]
    j = int(order[0]) if int(order[0]) != i else int(order[1])
    return (i, j)


def tree_swap(n, beta, trees):
    rem_t, add_t = trees
    if rem_t is None:
        return rule_swap_stabilize(n, beta)
    F = type_features(n, beta)
    present = np.where(n > 0)[0]
    pr = rem_t.predict_proba(F[present])[:, 1]
    i = int(present[np.argmax(pr)])
    pa = add_t.predict_proba(F)[:, 1]
    order = np.argsort(pa)[::-1]
    j = int(order[0]) if int(order[0]) != i else int(order[1])
    return (i, j)


def distill_trees():
    """Imitate frozen-v2 stabilizing choices on regenerated 25x dev
    trajectories (same streams as the dev cohort: domain 2), states at
    fissions 30 and 60. Frozen to g5_trees.pkl before confirmation."""
    ent = cohort.domain_entropy("dev", DEV_TAG)
    trees = {}
    for cand in cohort.CANDIDATES:
        cand_i = cohort.CANDIDATES.index(cand)
        Xr, yr, Xa, ya = [], [], [], []
        for m in range(40):
            beta, n0 = cohort.matrix_and_init(ent, m)
            rng = cohort._rng(ent, 2, cand_i, m)
            tr = sim.run_fissions(n0, beta, cand, 62, rng)
            for lm in (30, 60):
                if lm > tr["n_done"]:
                    continue
                s = tr["daughters"][lm - 1]
                X9 = Ft.direct9(lm, 100, tr["H"][:lm], int(s.sum()))
                i, j = RS.marginal_swap(s, beta, X9,
                                        RI._BUNDLES[cand], -1)
                F = type_features(s, beta)
                present = np.where(s > 0)[0]
                for t in present:
                    Xr.append(F[t])
                    yr.append(int(t == i))
                neg = cohort._rng(ent, 24, cand_i, m, lm)
                sub = list(neg.choice(sim.NG, 20, replace=False)) + [j]
                for t in sub:
                    Xa.append(F[t])
                    ya.append(int(t == j))
        if sum(yr) == 0 or sum(ya) == 0:
            trees[cand] = (None, None)
            continue
        rem_t = DecisionTreeClassifier(max_depth=3, random_state=0,
                                       class_weight="balanced")
        rem_t.fit(np.array(Xr), np.array(yr))
        add_t = DecisionTreeClassifier(max_depth=3, random_state=0,
                                       class_weight="balanced")
        add_t.fit(np.array(Xa), np.array(ya))
        trees[cand] = (rem_t, add_t)
    return trees


def policy_swap(policy, n, beta, cand, cand_i, hs, f, m, rep, ent):
    if policy == "noop":
        return None
    if policy == "random":
        rr = cohort._rng(ent, 24, 9, cand_i, m, rep, f)
        present = np.where(n > 0)[0]
        i = int(present[rr.integers(len(present))])
        j = int(rr.integers(sim.NG - 1))
        return (i, j + 1 if j >= i else j)
    if policy == "v2_down":
        X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
        return RS.marginal_swap(n, beta, X9, RI._BUNDLES[cand], -1)
    if policy == "L3":
        return tree_swap(n, beta, _TREES[cand])
    if policy == "L1" and hs and hs[-1] > sim.H_THRESH:
        return None                       # edit only right after a break
    if policy == "L2":
        run = 0
        for h in reversed(hs):
            if h > sim.H_THRESH:
                run += 1
            else:
                break
        if run >= 3:
            return None                   # edit only while streak < 3
    return rule_swap_stabilize(n, beta)


def run_policy(n0, beta, cand, cand_i, m, rep, policy, perturb_at=None,
               ent=None):
    """One 60-fission steered lineage. CRN: the growth stream key omits
    both policy and perturbation, so all arms share randomness."""
    ent = ent if ent is not None else RI._ENT
    rng = cohort._rng(ent, 24, 0, cand_i, m, rep)
    n = n0.copy()
    hs = []
    for f in range(1, STEER + 1):
        step = sim.run_fissions(n, beta, cand, 1, rng)
        if step["n_done"] < 1:
            break
        hs.append(float(step["H"][0]))
        n = step["final"]
        if perturb_at and f == perturb_at:
            pr = cohort._rng(ent, 24, 8, cand_i, m, rep)
            n = kswap_perturb(n, 8, pr)
        if f == STEER:
            break
        swap = policy_swap(policy, n, beta, cand, cand_i, hs, f, m,
                           rep, ent)
        if swap is not None:
            n = RI.apply_swap(n, swap)
    inh = np.array(hs) > sim.H_THRESH
    post = inh[perturb_at:] if perturb_at else inh
    return {"inherit": float(np.mean(inh)) if len(inh) else np.nan,
            "post_inherit": float(np.mean(post)) if len(post)
            else np.nan}


def home_unit(args):
    m, cand, rep = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    out = {}
    for p in POLICIES:
        out[p] = {"maint": run_policy(n0, beta, cand, cand_i, m, rep, p),
                  "recov": run_policy(n0, beta, cand, cand_i, m, rep, p,
                                      perturb_at=PERTURB_AT)}
    return {"matrix": m, "candidate": cand, "rep": rep, **out}


def regime_unit(args):
    ri, m, cand = args
    A, S = REGIMES[ri]
    ent = cohort.domain_entropy("confirmation",
                                f"g5-regime-{ri}-2026-08-14")
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(cohort._rng(ent, 0, m), a_mu=A, sigma=S)
    n0 = sim.make_initial_state(cohort._rng(ent, 1, m))
    out = {}
    for p in ("L0", "L3", "v2_down", "noop"):
        out[p] = run_policy(n0, beta, cand, cand_i, m, 0, p,
                            ent=ent)["inherit"]
    return {"regime": ri, "matrix": m, "candidate": cand, **out}


def biased_grow_gillespie(n, beta, rng, damp):
    """Candidate 02 growth with leave rates divided by `damp`
    (mirrors sim._grow_gillespie; new code path, frozen sim
    untouched)."""
    n = n.copy()
    c = beta @ n
    total = int(n.sum())
    events = 0
    while total < sim.NMAX:
        join, leave = sim.event_rates(n, c, total)
        rates = np.concatenate([join, leave / damp])
        mu = sim._sample_categorical(rates, rng)
        if mu < sim.NG:
            n[mu] += 1
            c += beta[:, mu]
            total += 1
        else:
            k = mu - sim.NG
            n[k] -= 1
            c -= beta[:, k]
            total -= 1
            if total == 0:
                return n, True
        events += 1
        if events >= 40 * sim.MAXSTEPS:
            break
    return n, False


def biased_grow_poisson(n, beta, rng, damp):
    """Candidate 03 growth with damped leave rates (mirrors
    sim._grow_poisson)."""
    n = n.copy()
    total = int(n.sum())
    steps = 0
    while total < sim.NMAX and steps < sim.MAXSTEPS:
        c = beta @ n
        join, leave = sim.event_rates(n, c, total)
        leave = leave / damp
        s = join.sum() + leave.sum()
        dt = sim.EVENTS_PER_STEP / s
        joins = rng.poisson(join * dt)
        leaves = np.minimum(rng.poisson(leave * dt), n)
        n = n + joins - leaves
        total = int(n.sum())
        steps += 1
        if total == 0:
            return n, True
    return n, False


def kinetic_unit(args):
    m, cand, lam = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    rng = cohort._rng(RI._ENT, 24, 5, cand_i, m, int(lam * 10))
    if lam == 0.0:
        tr = sim.run_fissions(n0, beta, cand, STEER, rng)
        inh = tr["inherited"]
        return {"matrix": m, "candidate": cand, "lam": lam,
                "inherit": float(np.mean(inh)) if len(inh) else np.nan}
    n = n0.copy()
    hs = []
    for _ in range(STEER):
        x = n / max(n.sum(), 1)
        pct = np.argsort(np.argsort(x @ beta)) / (sim.NG - 1)
        damp = 1.0 + lam * pct
        if cand == "02":
            grown, dead = biased_grow_gillespie(n, beta, rng, damp)
        else:
            grown, dead = biased_grow_poisson(n, beta, rng, damp)
        if dead or grown.sum() < 2:
            break
        parent = grown
        if cand == "02":
            ca, cb = sim._split_equal(parent, rng)
            d = ca
        else:
            ca, cb = sim._split_binomial(parent, rng)
            d = ca if rng.random() < 0.5 else cb
            if d.sum() == 0:
                d = ca if ca.sum() > 0 else cb
        hs.append(sim.cosine_h(parent.astype(float), d.astype(float)))
        n = d
    inh = np.array(hs) > sim.H_THRESH
    return {"matrix": m, "candidate": cand, "lam": lam,
            "inherit": float(np.mean(inh)) if len(inh) else np.nan}


def mat_boot_diff(units, key_a, key_b, get, n_boot=2048, seed=7):
    """Matrix-bootstrap CI on mean(get(u,key_a) - get(u,key_b))."""
    per = {}
    for u in units:
        a, b = get(u, key_a), get(u, key_b)
        if np.isfinite(a) and np.isfinite(b):
            per.setdefault(u["matrix"], []).append(a - b)
    means = {mm: np.mean(v) for mm, v in per.items()}
    mats = list(means)
    rng = np.random.default_rng(seed)
    boots = [np.mean([means[mm] for mm in
                      rng.choice(mats, size=len(mats), replace=True)])
             for _ in range(n_boot)]
    return (float(np.mean(list(means.values()))),
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)))


def main():
    global _TREES
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    _TREES = distill_trees()
    with open(os.path.join(OUT, "g5_trees.pkl"), "wb") as f:
        pickle.dump(_TREES, f, protocol=4)
    print(f"L3 trees distilled+frozen in {time.time()-t0:.0f}s",
          flush=True)

    t0 = time.time()
    jobs = [(m, c, r) for c in cohort.CANDIDATES for m in range(N_MAT)
            for r in REPS]
    with Pool(12) as pool:
        home = pool.map(home_unit, jobs)
    print(f"home campaign in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "g5_home_units.pkl"), "wb") as f:
        pickle.dump(home, f, protocol=4)

    t0 = time.time()
    jobs = [(ri, m, c) for ri in range(len(REGIMES))
            for c in cohort.CANDIDATES for m in range(12)]
    with Pool(12) as pool:
        reg = pool.map(regime_unit, jobs)
    print(f"regime campaign in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    jobs = [(m, c, lam) for c in cohort.CANDIDATES for m in range(N_MAT)
            for lam in [0.0] + LAMBDAS]
    with Pool(12) as pool:
        kin = pool.map(kinetic_unit, jobs)
    print(f"kinetic prototype in {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "g5_aux_units.pkl"), "wb") as f:
        pickle.dump({"regime": reg, "kinetic": kin}, f, protocol=4)

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in home if u["candidate"] == cand]
        entry = {"home": {}, "contrasts": {}}
        for p in POLICIES:
            entry["home"][p] = {
                "maintenance": float(np.nanmean(
                    [u[p]["maint"]["inherit"] for u in cu])),
                "post_perturb": float(np.nanmean(
                    [u[p]["recov"]["post_inherit"] for u in cu]))}
        for p in ("v2_down", "L0", "L1", "L2", "L3", "random"):
            for cond, get in (
                ("maint", lambda u, k: u[k]["maint"]["inherit"]),
                ("recov", lambda u, k: u[k]["recov"]["post_inherit"])):
                d, lo, hi = mat_boot_diff(cu, p, "noop", get)
                entry["contrasts"][f"{p}-noop:{cond}"] = {
                    "diff": d, "ci": [lo, hi]}
        v2g = entry["home"]["v2_down"]["maintenance"] \
            - entry["home"]["noop"]["maintenance"]
        entry["ladder_fraction_of_v2"] = {
            p: float((entry["home"][p]["maintenance"]
                      - entry["home"]["noop"]["maintenance"])
                     / max(v2g, 1e-9))
            for p in ("L0", "L1", "L2", "L3")}
        entry["regimes"] = {}
        for ri in range(len(REGIMES)):
            rows = [r for r in reg if r["candidate"] == cand
                    and r["regime"] == ri]
            entry["regimes"][str(REGIMES[ri])] = {
                p: float(np.nanmean([r[p] for r in rows]))
                for p in ("L0", "L3", "v2_down", "noop")}
        krows = [k for k in kin if k["candidate"] == cand]
        entry["kinetic"] = {str(lam): float(np.nanmean(
            [k["inherit"] for k in krows if k["lam"] == lam]))
            for lam in [0.0] + LAMBDAS}
        pairs = {}
        for k in krows:
            pairs.setdefault(k["matrix"], {})[k["lam"]] = k["inherit"]
        for lam in LAMBDAS:
            ku = [{"matrix": mm, "a": v.get(lam, np.nan),
                   "b": v.get(0.0, np.nan)} for mm, v in pairs.items()]
            d, lo, hi = mat_boot_diff(ku, "a", "b", lambda u, k: u[k])
            entry["contrasts"][f"kinetic{lam}-baseline"] = {
                "diff": d, "ci": [lo, hi]}
        results[cand] = entry

        print(f"\n=== G5 candidate {cand} ===")
        print(f"{'policy':8s} {'maint':>7s} {'post-k8':>8s} "
              f"{'frac-of-v2':>11s}")
        for p in POLICIES:
            e = entry["home"][p]
            fr = entry["ladder_fraction_of_v2"].get(p)
            print(f"{p:8s} {e['maintenance']:7.3f} "
                  f"{e['post_perturb']:8.3f} "
                  + (f"{fr:11.2f}" if fr is not None else " " * 11))
        for ck, cv in entry["contrasts"].items():
            print(f"  {ck}: {cv['diff']:+.4f} "
                  f"CI [{cv['ci'][0]:+.4f},{cv['ci'][1]:+.4f}]")
        for rk, rv in entry["regimes"].items():
            print(f"regime {rk}: " + " ".join(
                f"{p} {rv[p]:.3f}" for p in
                ("L0", "L3", "v2_down", "noop")))
        print("kinetic prototype inherit: " + " ".join(
            f"lam={k} {v:.3f}" for k, v in entry["kinetic"].items()))

    with open(os.path.join(OUT, "g5_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g5_results.json"))


if __name__ == "__main__":
    main()
