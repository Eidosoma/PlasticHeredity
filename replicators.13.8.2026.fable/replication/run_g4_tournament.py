"""Phase G4: mechanism disagreement tournament (PHASE_G.md; domain 23).

Which explanation actually drives intervention effects: the frozen v2
model, the outgoing-catalytic-influence rule, or flux alignment (R_Q)?

1. MINING (dev matrices only): sample candidate swaps; compute
   dv2 (risk change; stabilizing = negative), rule sign
   (out_infl[add] - out_infl[remove]; stabilizing = positive),
   dR_Q (stabilizing = positive). Assign the reviewer's four
   disagreement classes:
     A: v2 stab, rule stab, R_Q destab
     B: v2 stab, rule destab, R_Q stab
     C: v2 destab, rule stab, R_Q destab
     D: v2 destab, rule destab, R_Q stab
   Noise floors (registered): |dv2| > 0.01, |dR_Q| > 0.002.
   Target 100 edits/class/candidate; shortfalls reported.
2. BRANCH TEST: 16 edited + 16 no-op CRN branches x 12 fissions per
   edit; realized dq = q_edit - q_noop; per class, which predictor's
   stabilization sign matches the realized mean (matrix-bootstrapped).
3. TRANSPLANTATION: 40 strong native edits per candidate; the same
   (i, j) applied to the most composition-similar state under a
   DIFFERENT matrix; does the realized sign follow the local v2
   prediction at the applied state (relational) or the native edit's
   effect (molecular identity)?
4. BETA SURGERY: at fixed composition, multiplicative present-present
   edge changes raising / lowering the composition-weighted outgoing
   influence, plus random-edge arms of equal total Frobenius change;
   does web-tightness alone move heredity?
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

import sim
import features as Ft
import cohort
import growth_trace as GT
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
DEV_TAG = "25x-2026-08-13"
CONF_TAG = "steering-2026-08-13"
N_MINE_MAT = 60
N_SWAPS_PER_STATE = 40
TARGET_PER_CLASS = 100
N_BR = 16
HOR = 12
N_TRANS = 40
N_SURG_MAT = 24
CLASSES = ["A", "B", "C", "D"]


def out_infl(n, beta):
    x = n / max(n.sum(), 1)
    return x @ beta               # c_t = sum_i x_i beta[i, t]


def mine_unit(args):
    m, cand = args
    ent = cohort.domain_entropy("dev", DEV_TAG)
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(ent, m)
    rng = cohort._rng(ent, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, 70, rng)
    out = []
    for lm in (30, 60):
        if lm > traj["n_done"]:
            continue
        s = traj["daughters"][lm - 1]
        hs = traj["H"][:lm]
        X9 = Ft.direct9(lm, 100, hs, int(s.sum()))
        base_v2 = float(RI.score_states(
            RI._BUNDLES[cand], X9, [Ft.graph_state_195(s, beta)])[0])
        base_rq = GT.r_q(s, beta)
        infl = out_infl(s, beta)
        rr = cohort._rng(ent, 23, cand_i, m, lm)
        present = np.where(s > 0)[0]
        cands, feats = [], []
        for _ in range(N_SWAPS_PER_STATE):
            i = int(present[rr.integers(len(present))])
            j = int(rr.integers(sim.NG - 1))
            j = j + 1 if j >= i else j
            ne = RI.apply_swap(s, (i, j))
            cands.append((i, j, ne))
            feats.append(Ft.graph_state_195(ne, beta))
        v2s = RI.score_states(RI._BUNDLES[cand],
                              X9, np.array(feats))
        for (i, j, ne), v2e in zip(cands, v2s):
            dv2 = float(v2e - base_v2)
            drq = GT.r_q(ne, beta) - base_rq
            rule = float(infl[j] - infl[i])
            if abs(dv2) < 0.01 or abs(drq) < 0.002:
                continue
            v2_stab, rule_stab, rq_stab = dv2 < 0, rule > 0, drq > 0
            if v2_stab and rule_stab and not rq_stab:
                cl = "A"
            elif v2_stab and not rule_stab and rq_stab:
                cl = "B"
            elif not v2_stab and rule_stab and not rq_stab:
                cl = "C"
            elif not v2_stab and not rule_stab and rq_stab:
                cl = "D"
            else:
                continue
            out.append({"cl": cl, "m": m, "lm": lm, "swap": (i, j),
                        "dv2": dv2, "drq": float(drq), "rule": rule,
                        "state": s.copy(), "beta_m": m,
                        "abund": int(s[i]),
                        "disp": float(1 - sim.cosine_h(
                            ne.astype(float), s.astype(float)))})
    return cand, out


def branch_test(args):
    cand, e, eid = args
    ent = cohort.domain_entropy("dev", DEV_TAG)
    cand_i = cohort.CANDIDATES.index(cand)
    beta, _ = cohort.matrix_and_init(ent, e["beta_m"])
    s = e["state"]
    ne = RI.apply_swap(s, tuple(e["swap"]))
    q_e, q_n = [], []
    for b in range(N_BR):
        rb = cohort._rng(ent, 23, 5, cand_i, eid, b)
        q_e.append(float(Ft.joint_break_run3(
            sim.run_fissions(ne, beta, cand, HOR, rb)["inherited"])))
        rb2 = cohort._rng(ent, 23, 5, cand_i, eid, b)
        q_n.append(float(Ft.joint_break_run3(
            sim.run_fissions(s, beta, cand, HOR, rb2)["inherited"])))
    return cand, e["cl"], e["m"], float(np.mean(q_e) - np.mean(q_n)), e


def transplant_unit(args):
    cand, e, eid, lib = args
    ent = cohort.domain_entropy("dev", DEV_TAG)
    cand_i = cohort.CANDIDATES.index(cand)
    i, j = e["swap"]
    best = None
    for (m2, s2) in lib:
        if m2 == e["beta_m"] or s2[i] < 1:
            continue
        h = sim.cosine_h(s2.astype(float), e["state"].astype(float))
        if best is None or h > best[0]:
            best = (h, m2, s2)
    if best is None:
        return None
    _, m2, s2 = best
    beta2, _ = cohort.matrix_and_init(ent, m2)
    hs_dummy = np.full(30, 0.95)
    X9 = Ft.direct9(30, 100, hs_dummy, int(s2.sum()))
    base = float(RI.score_states(RI._BUNDLES[cand], X9,
                                 [Ft.graph_state_195(s2, beta2)])[0])
    ne2 = RI.apply_swap(s2, (i, j))
    loc = float(RI.score_states(RI._BUNDLES[cand], X9,
                                [Ft.graph_state_195(ne2, beta2)])[0])
    dv2_local = loc - base
    dq = []
    for b in range(N_BR):
        rb = cohort._rng(ent, 23, 6, cand_i, eid, b)
        qe = float(Ft.joint_break_run3(
            sim.run_fissions(ne2, beta2, cand, HOR, rb)["inherited"]))
        rb2 = cohort._rng(ent, 23, 6, cand_i, eid, b)
        qn = float(Ft.joint_break_run3(
            sim.run_fissions(s2, beta2, cand, HOR, rb2)["inherited"]))
        dq.append(qe - qn)
    return {"cand": cand, "native_dv2": e["dv2"],
            "local_dv2": float(dv2_local), "realized": float(np.mean(dq))}


def surgery_unit(args):
    m, cand = args
    ent = cohort.domain_entropy("confirmation", CONF_TAG)
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(ent, m)
    rng = cohort._rng(ent, 7, cand_i, m, 0)
    traj = sim.run_fissions(n0, beta, cand, 60, rng)
    s = traj["daughters"][59]
    present = np.where(s > 0)[0]
    pp = np.ix_(present, present)
    delta = 0.5                       # multiplicative surgery strength
    arms = {}
    B = beta.copy()
    B[pp] = beta[pp] * (1 + delta)
    arms["raise"] = B
    B = beta.copy()
    B[pp] = beta[pp] / (1 + delta)
    arms["lower"] = B
    fro = float(np.linalg.norm(arms["raise"][pp] - beta[pp]))
    rr = cohort._rng(ent, 23, 7, cand_i, m)
    B = beta.copy()
    k = len(present) ** 2
    ii = rr.integers(0, sim.NG, k)
    jj = rr.integers(0, sim.NG, k)
    pert = B[ii, jj] * delta * rr.choice([-0.5, 0.5], k)
    scale = fro / max(np.linalg.norm(pert), 1e-12)
    B[ii, jj] = np.maximum(B[ii, jj] + pert * scale, 1e-12)
    arms["random"] = B
    arms["none"] = beta
    rec = {}
    for name, Bm in arms.items():
        qs = []
        for b in range(N_BR):
            rb = cohort._rng(ent, 23, 8, cand_i, m, b)
            qs.append(float(Ft.joint_break_run3(
                sim.run_fissions(s, Bm, cand, HOR, rb)["inherited"])))
        rec[name] = float(np.mean(qs))
    return {"matrix": m, "candidate": cand, **rec}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)

    # ---------------- mining ------------------------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MINE_MAT)]
    with Pool(12) as pool:
        mined = pool.map(mine_unit, jobs)
    pools = {c: {cl: [] for cl in CLASSES} for c in cohort.CANDIDATES}
    lib = {c: [] for c in cohort.CANDIDATES}
    for cand, edits in mined:
        for e in edits:
            if len(pools[cand][e["cl"]]) < TARGET_PER_CLASS:
                pools[cand][e["cl"]].append(e)
        for e in edits[:1]:
            lib[cand].append((e["beta_m"], e["state"]))
    print(f"mining in {time.time()-t0:.0f}s | counts: " + " | ".join(
        f"{c}:" + ",".join(f"{cl}={len(pools[c][cl])}"
                           for cl in CLASSES)
        for c in cohort.CANDIDATES))

    # ---------------- branch tournament -------------------------------
    t0 = time.time()
    jobs = [(c, e, eid)
            for c in cohort.CANDIDATES
            for eid, e in enumerate(
                [e for cl in CLASSES for e in pools[c][cl]])]
    with Pool(12) as pool:
        tested = pool.map(branch_test, jobs)
    print(f"tournament branches in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        entry = {"classes": {}}
        for cl in CLASSES:
            rows = [(m, dq) for c, cl2, m, dq, e in tested
                    if c == cand and cl2 == cl]
            if not rows:
                entry["classes"][cl] = None
                continue
            mats = np.array([m for m, _ in rows])
            dqs = np.array([d for _, d in rows])
            rng = np.random.default_rng(17)
            ci = RI.boot_lower(dqs, mats, rng, n=1024)
            entry["classes"][cl] = {"n": len(rows),
                                    "mean_dq": float(dqs.mean()),
                                    "ci": ci}
        # who wins: predicted stabilization sign per class
        # (dq < 0 = realized stabilization of the joint event)
        pred = {"v2": {"A": -1, "B": -1, "C": +1, "D": +1},
                "rule": {"A": -1, "B": +1, "C": -1, "D": +1},
                "rq": {"A": +1, "B": -1, "C": +1, "D": -1}}
        score = {}
        for name in pred:
            hits = 0
            tot = 0
            for cl in CLASSES:
                e = entry["classes"][cl]
                if e is None or not (e["ci"][0] > 0 or e["ci"][1] < 0):
                    continue
                tot += 1
                realized = -1 if e["mean_dq"] < 0 else +1
                hits += int(realized == pred[name][cl])
            score[name] = {"hits": hits, "decided_classes": tot}
        entry["winner_score"] = score
        results[cand] = entry
        print(f"\n=== G4 tournament candidate {cand} ===")
        for cl in CLASSES:
            e = entry["classes"][cl]
            if e:
                print(f"class {cl}: n={e['n']} realized dq "
                      f"{e['mean_dq']:+.4f} CI [{e['ci'][0]:+.4f},"
                      f"{e['ci'][1]:+.4f}]")
        print("predictor scores (decided classes):", score)

    # ---------------- transplantation ---------------------------------
    t0 = time.time()
    jobs = []
    for cand in cohort.CANDIDATES:
        strong = sorted((e for cl in ("A", "B") for e in pools[cand][cl]),
                        key=lambda e: e["dv2"])[:N_TRANS]
        for eid, e in enumerate(strong):
            jobs.append((cand, e, 10_000 + eid, lib[cand]))
    with Pool(12) as pool:
        trans = [t for t in pool.map(transplant_unit, jobs)
                 if t is not None]
    print(f"\ntransplantation in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        rows = [t for t in trans if t["cand"] == cand]
        if not rows:
            continue
        local = np.mean([np.sign(t["realized"]) == np.sign(t["local_dv2"])
                         for t in rows if abs(t["realized"]) > 0.01])
        native = np.mean([np.sign(t["realized"])
                          == np.sign(t["native_dv2"])
                          for t in rows if abs(t["realized"]) > 0.01])
        results[cand]["transplant"] = {"n": len(rows),
                                       "follows_local_v2": float(local),
                                       "follows_native_edit":
                                       float(native)}
        print(f"{cand}: transplanted edits follow LOCAL v2 {local:.2f} "
              f"vs native effect {native:.2f} (n={len(rows)})")

    # ---------------- beta surgery ------------------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_SURG_MAT)]
    with Pool(12) as pool:
        surg = pool.map(surgery_unit, jobs)
    print(f"\nbeta surgery in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        rows = [s for s in surg if s["candidate"] == cand]
        mats = np.array([s["matrix"] for s in rows])
        d_rl = np.array([s["raise"] - s["lower"] for s in rows])
        d_rn = np.array([s["random"] - s["none"] for s in rows])
        rng = np.random.default_rng(23)
        ci_rl = RI.boot_lower(d_rl, mats, rng, n=1024)
        ci_rn = RI.boot_lower(d_rn, mats, rng, n=1024)
        sep = bool((ci_rl[0] > 0 or ci_rl[1] < 0)
                   and ci_rn[0] <= 0 <= ci_rn[1])
        results[cand]["surgery"] = {
            "means": {k: float(np.mean([s[k] for s in rows]))
                      for k in ("raise", "lower", "random", "none")},
            "raise_minus_lower": {"mean": float(d_rl.mean()),
                                  "ci": ci_rl},
            "random_minus_none": {"mean": float(d_rn.mean()),
                                  "ci": ci_rn},
            "sufficiency_pass": sep}
        s = results[cand]["surgery"]
        print(f"{cand}: q raise {s['means']['raise']:.3f} lower "
              f"{s['means']['lower']:.3f} random "
              f"{s['means']['random']:.3f} none {s['means']['none']:.3f}"
              f" | raise-lower {d_rl.mean():+.3f} CI "
              f"[{ci_rl[0]:+.3f},{ci_rl[1]:+.3f}] | random-none "
              f"{d_rn.mean():+.3f} -> sufficiency "
              f"{s['sufficiency_pass']}")

    with open(os.path.join(OUT, "g4_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g4_results.json"))


if __name__ == "__main__":
    main()
