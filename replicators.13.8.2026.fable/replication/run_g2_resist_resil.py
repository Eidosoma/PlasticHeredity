"""Phase G2: separate the knob into resistance and resilience
(preregistered in PHASE_G.md).

Stage A: frozen q_B(s) = P(break within 6 | s); edits screened on q_B;
outcome = break hazard alone.
Stage B: frozen q_R(s) = P(run3 within 8 | post-break daughter s);
the IDENTICAL post-break state restored across arms — an
unconditional causal test of recovery.
Stage C: staged controller (explore -> recover -> consolidate) vs
always-stabilize / always-explore / random / noop over 60 fissions.
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
import registry_v2 as R2
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
DEV_TAG = "25x-2026-08-13"
CONF_TAG = "g2-2026-08-14"
N_DEV = 400
N_MAT = 24
HB, HR = 6, 8
N_BR = 48
LANDMARKS = [30, 60]
STEER = 60
POLICIES = ["stabilize", "explore", "staged", "random", "noop"]

_QB = _QR = None


def g2_dev_unit(args):
    m, cand = args
    ent = cohort.domain_entropy("dev", DEV_TAG)
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(ent, m)
    rng = cohort._rng(ent, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, 100, rng)
    nd = traj["n_done"]
    hs, inh, daughters = traj["H"], traj["inherited"], traj["daughters"]
    A = {"X9": [], "X195": [], "y": []}
    B = {"X9": [], "X195": [], "y": []}
    for g in range(1, nd - HB + 1):
        s = daughters[g - 1]
        A["X9"].append(Ft.direct9(g, 100, hs[:g], int(s.sum())))
        A["X195"].append(Ft.graph_state_195(s, beta))
        A["y"].append(float((~inh[g:g + HB]).any()))
        if not inh[g - 1] and g + HR <= nd:      # post-break daughter
            run = best = 0
            for v in inh[g:g + HR]:
                run = run + 1 if v else 0
                best = max(best, run)
            B["X9"].append(A["X9"][-1])
            B["X195"].append(A["X195"][-1])
            B["y"].append(float(best >= 3))
    pack = lambda D: (np.array(D["X9"]).reshape(len(D["y"]), 9),
                      np.array(D["X195"]).reshape(len(D["y"]), 195),
                      np.array(D["y"]))
    return cand, pack(A), pack(B)


def predict_q(bundle, X9, X195):
    return R2.predict_v2(bundle, X9, X195)["v2"]


def screen_edit(bundle, n, beta, X9, sign):
    """Marginal (remove, add) swap moving the student's score in the
    given direction (+1 raise, -1 lower)."""
    present = np.where(n > 0)[0]
    eye = np.eye(sim.NG, dtype=np.int64)
    add = predict_q(bundle, np.tile(X9, (sim.NG, 1)),
                    np.array([Ft.graph_state_195(n + eye[j], beta)
                              for j in range(sim.NG)]))
    rem = predict_q(bundle, np.tile(X9, (len(present), 1)),
                    np.array([Ft.graph_state_195(n - eye[i], beta)
                              for i in present]))
    if sign > 0:
        j, i = int(np.argmax(add)), int(present[np.argmax(rem)])
    else:
        j, i = int(np.argmin(add)), int(present[np.argmin(rem)])
    if i == j:
        order = np.argsort(add)[::-1] if sign > 0 else np.argsort(add)
        j = int(order[1]) if int(order[0]) == i else int(order[0])
    return (i, j)


def stageAB_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    rng = cohort._rng(RI._ENT, 21, 0, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, 100, rng)
    nd, hs, inh = traj["n_done"], traj["H"], traj["inherited"]
    daughters = traj["daughters"]
    qb, qr = _QB[cand], _QR[cand]
    res = {"A": [], "B": None}

    for lm in LANDMARKS:
        if lm + HB > nd:
            continue
        s = daughters[lm - 1]
        X9 = Ft.direct9(lm, 100, hs[:lm], int(s.sum()))
        arms = {"noop": None,
                "bu": screen_edit(qb, s, beta, X9, +1),
                "bd": screen_edit(qb, s, beta, X9, -1)}
        rr = cohort._rng(RI._ENT, 21, 1, cand_i, m, lm)
        present = np.where(s > 0)[0]
        i = int(present[rr.integers(len(present))])
        j = int(rr.integers(sim.NG - 1))
        arms["random"] = (i, j + 1 if j >= i else j)
        rec = {}
        for name, swap in arms.items():
            s0 = RI.apply_swap(s, swap)
            brk = []
            for b in range(N_BR):
                rb = cohort._rng(RI._ENT, 21, 2, cand_i, m, lm, b)
                br = sim.run_fissions(s0, beta, cand, HB, rb)
                brk.append(float((~br["inherited"]).any())
                           if br["n_done"] else np.nan)
            rec[name] = float(np.nanmean(brk))
        res["A"].append({"lm": lm, **rec})

    # Stage B: first break at fission >= 10
    bidx = next((g for g in range(10, nd - HR)
                 if not inh[g - 1]), None)
    if bidx is not None:
        s = daughters[bidx - 1]
        X9 = Ft.direct9(bidx, 100, hs[:bidx], int(s.sum()))
        anchor = (daughters[bidx - 2] if bidx >= 2 else n0).astype(float)
        arms = {"noop": None,
                "ru": screen_edit(qr, s, beta, X9, +1),
                "rd": screen_edit(qr, s, beta, X9, -1)}
        rr = cohort._rng(RI._ENT, 21, 3, cand_i, m)
        present = np.where(s > 0)[0]
        i = int(present[rr.integers(len(present))])
        j = int(rr.integers(sim.NG - 1))
        arms["random"] = (i, j + 1 if j >= i else j)
        rec = {}
        for name, swap in arms.items():
            s0 = RI.apply_swap(s, swap)
            run3, run5, ttr, anc, cnt = [], [], [], [], []
            for b in range(N_BR):
                rb = cohort._rng(RI._ENT, 21, 4, cand_i, m, b)
                br = sim.run_fissions(s0, beta, cand, HR, rb)
                ib = br["inherited"]
                run = best = 0
                t3 = None
                for k, v in enumerate(ib):
                    run = run + 1 if v else 0
                    best = max(best, run)
                    if run == 3 and t3 is None:
                        t3 = k + 1
                run3.append(float(best >= 3))
                run5.append(float(best >= 5))
                ttr.append(t3 if t3 is not None else np.nan)
                cnt.append(float(np.sum(ib)))
                if t3 is not None:
                    anc.append(sim.cosine_h(
                        br["daughters"][t3 - 3].astype(float), anchor))
            rec[name] = {"run3": float(np.nanmean(run3)),
                         "run5": float(np.nanmean(run5)),
                         "ttr": float(np.nanmean(ttr)),
                         "anchor": float(np.nanmean(anc))
                         if anc else np.nan,
                         "inherited": float(np.nanmean(cnt))}
        res["B"] = rec
    return {"matrix": m, "candidate": cand, **res}


def stageC_unit(args):
    m, cand, rep = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    qb, qr = _QB[cand], _QR[cand]
    out = {}
    for policy in POLICIES:
        rng = cohort._rng(RI._ENT, 21, 5, cand_i, m, rep)
        n = n0.copy()
        hs, inh = [], []
        phase = "EXPLORE"
        run = 0
        ep_starts, breaks, ep_surv = [], 0, []
        cur_ep_len = None
        for f in range(1, STEER + 1):
            step = sim.run_fissions(n, beta, cand, 1, rng)
            if step["n_done"] < 1:
                break
            h = float(step["H"][0])
            hs.append(h)
            v = h > 0.9
            inh.append(v)
            n = step["final"]
            if not v:
                breaks += 1
                run = 0
                if cur_ep_len is not None:
                    ep_surv.append(cur_ep_len)
                    cur_ep_len = None
                if policy == "staged":
                    phase = "RECOVER"
            else:
                run += 1
                if run == 3 and (breaks > 0):
                    ep_starts.append(n.copy())
                    cur_ep_len = 0
                    if policy == "staged":
                        phase = "CONSOLIDATE"
                elif cur_ep_len is not None:
                    cur_ep_len += 1
            if f == STEER or policy == "noop":
                continue
            if policy == "random":
                rr = cohort._rng(RI._ENT, 21, 6, cand_i, m, rep, f)
                present = np.where(n > 0)[0]
                i = int(present[rr.integers(len(present))])
                j = int(rr.integers(sim.NG - 1))
                swap = (i, j + 1 if j >= i else j)
            else:
                X9 = Ft.direct9(f, 100, np.array(hs), int(n.sum()))
                if policy == "stabilize":
                    swap = screen_edit(qb, n, beta, X9, -1)
                elif policy == "explore":
                    swap = screen_edit(qb, n, beta, X9, +1)
                else:  # staged
                    if phase == "EXPLORE":
                        swap = screen_edit(qb, n, beta, X9, +1)
                    elif phase == "RECOVER":
                        swap = screen_edit(qr, n, beta, X9, +1)
                    else:
                        swap = screen_edit(qb, n, beta, X9, -1)
            n = RI.apply_swap(n, swap)
        if cur_ep_len is not None:
            ep_surv.append(cur_ep_len)
        distinct = 0
        prev = None
        for e in ep_starts:
            if prev is None or sim.cosine_h(e.astype(float),
                                            prev.astype(float)) < 0.9:
                distinct += 1
            prev = e
        out[policy] = {"episodes": len(ep_starts),
                       "distinct_episodes": distinct,
                       "breaks": breaks,
                       "mean_ep_survival": float(np.mean(ep_surv))
                       if ep_surv else np.nan,
                       "inherit_frac": float(np.mean(inh))
                       if inh else np.nan}
    return {"matrix": m, "candidate": cand, "rep": rep, "out": out}


def main():
    global _QB, _QR
    os.makedirs(OUT, exist_ok=True)

    # ---------------- train + freeze q_B, q_R -------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV)]
    with Pool(12) as pool:
        dev = pool.map(g2_dev_unit, jobs)
    _QB, _QR = {}, {}
    stats = {}
    for cand in cohort.CANDIDATES:
        cu = [d for d in dev if d[0] == cand]
        XA9 = np.vstack([d[1][0] for d in cu])
        XA195 = np.vstack([d[1][1] for d in cu])
        yA = np.concatenate([d[1][2] for d in cu])
        XB9 = np.vstack([d[2][0] for d in cu if len(d[2][2])])
        XB195 = np.vstack([d[2][1] for d in cu if len(d[2][2])])
        yB = np.concatenate([d[2][2] for d in cu if len(d[2][2])])
        _QB[cand] = R2.train_v2(XA9, XA195, yA)
        _QR[cand] = R2.train_v2(XB9, XB195, yB)
        stats[cand] = {"nA": int(len(yA)), "prevA": float(yA.mean()),
                       "nB": int(len(yB)), "prevB": float(yB.mean())}
        print(f"{cand}: q_B on {len(yA)} rows (prev {yA.mean():.3f}) | "
              f"q_R on {len(yB)} rows (prev {yB.mean():.3f})")
    with open(os.path.join(OUT, "g2_students.pkl"), "wb") as f:
        pickle.dump({"qb": _QB, "qr": _QR}, f, protocol=4)
    print(f"students trained+frozen in {time.time()-t0:.0f}s")

    RI._ENT = cohort.domain_entropy("confirmation", CONF_TAG)
    results = {"students": stats}

    # ---------------- Stages A + B ------------------------------------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MAT)]
    with Pool(12) as pool:
        ab = pool.map(stageAB_unit, jobs)
    print(f"Stages A+B in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        cu = [u for u in ab if u["candidate"] == cand]
        rowsA = [r for u in cu for r in u["A"]]
        matsA = np.array([u["matrix"] for u in cu for _ in u["A"]])
        rng = np.random.default_rng(11)
        dA = np.array([r["bu"] - r["bd"] for r in rowsA])
        rA = np.array([r["random"] - r["noop"] for r in rowsA])
        ciA = RI.boot_lower(dA, matsA, rng, n=1024)
        ciAr = RI.boot_lower(rA, matsA, rng, n=1024)
        rowsB = [u["B"] for u in cu if u["B"] is not None]
        matsB = np.array([u["matrix"] for u in cu if u["B"] is not None])
        dB = np.array([r["ru"]["run3"] - r["rd"]["run3"] for r in rowsB])
        rB = np.array([r["random"]["run3"] - r["noop"]["run3"]
                       for r in rowsB])
        ciB = RI.boot_lower(dB, matsB, rng, n=1024)
        ciBr = RI.boot_lower(rB, matsB, rng, n=1024)
        entry = {
            "A": {"arm_means": {k: float(np.mean([r[k] for r in rowsA]))
                                for k in ("bu", "bd", "random", "noop")},
                  "up_down": {"mean": float(dA.mean()), "ci": ciA},
                  "random_noop": {"mean": float(rA.mean()), "ci": ciAr},
                  "pass": bool(dA.mean() > 0 and ciA[0] > 0
                               and ciAr[0] <= 0 <= ciAr[1])},
            "B": {"n_states": len(rowsB),
                  "arm_run3": {k: float(np.mean([r[k]["run3"]
                                                 for r in rowsB]))
                               for k in ("ru", "rd", "random", "noop")},
                  "arm_run5": {k: float(np.mean([r[k]["run5"]
                                                 for r in rowsB]))
                               for k in ("ru", "rd", "random", "noop")},
                  "arm_ttr": {k: float(np.nanmean([r[k]["ttr"]
                                                   for r in rowsB]))
                              for k in ("ru", "rd", "random", "noop")},
                  "up_down": {"mean": float(dB.mean()), "ci": ciB},
                  "random_noop": {"mean": float(rB.mean()), "ci": ciBr},
                  "pass": bool(dB.mean() > 0 and ciB[0] > 0
                               and ciBr[0] <= 0 <= ciBr[1])},
        }
        results[cand] = entry
        print(f"\n=== G2 candidate {cand} ===")
        a = entry["A"]
        print(f"A break-hazard: bu {a['arm_means']['bu']:.3f} bd "
              f"{a['arm_means']['bd']:.3f} random "
              f"{a['arm_means']['random']:.3f} noop "
              f"{a['arm_means']['noop']:.3f} | up-down "
              f"{a['up_down']['mean']:+.3f} CI [{a['up_down']['ci'][0]:+.3f},"
              f"{a['up_down']['ci'][1]:+.3f}] -> pass {a['pass']}")
        b = entry["B"]
        print(f"B run3 (shared break state, n={b['n_states']}): ru "
              f"{b['arm_run3']['ru']:.3f} rd {b['arm_run3']['rd']:.3f} "
              f"random {b['arm_run3']['random']:.3f} noop "
              f"{b['arm_run3']['noop']:.3f} | up-down "
              f"{b['up_down']['mean']:+.3f} CI [{b['up_down']['ci'][0]:+.3f},"
              f"{b['up_down']['ci'][1]:+.3f}] -> pass {b['pass']}")
        print(f"B time-to-renewal: ru {b['arm_ttr']['ru']:.2f} vs rd "
              f"{b['arm_ttr']['rd']:.2f}")

    # ---------------- Stage C -----------------------------------------
    t0 = time.time()
    jobs = [(m, c, r) for c in cohort.CANDIDATES for m in range(N_MAT)
            for r in (0, 1)]
    with Pool(12) as pool:
        sc = pool.map(stageC_unit, jobs)
    print(f"\nStage C in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        cu = [u for u in sc if u["candidate"] == cand]
        entry = {}
        for p in POLICIES:
            entry[p] = {k: float(np.nanmean([u["out"][p][k]
                                             for u in cu]))
                        for k in ("episodes", "distinct_episodes",
                                  "breaks", "mean_ep_survival",
                                  "inherit_frac")}
        g1 = entry["staged"]["distinct_episodes"] \
            > entry["noop"]["distinct_episodes"]
        g2c = entry["staged"]["breaks"] < entry["explore"]["breaks"]
        g3 = entry["staged"]["mean_ep_survival"] \
            > entry["explore"]["mean_ep_survival"]
        results[f"C_{cand}"] = {**entry,
                                "gates": {"more_distinct_than_noop": bool(g1),
                                          "fewer_breaks_than_explore": bool(g2c),
                                          "longer_survival_than_explore": bool(g3)},
                                "pass": bool(g1 and g2c and g3)}
        print(f"\n=== G2 Stage C candidate {cand} ===")
        for p in POLICIES:
            e = entry[p]
            print(f"{p:10s} episodes {e['episodes']:.2f} distinct "
                  f"{e['distinct_episodes']:.2f} breaks {e['breaks']:.1f} "
                  f"ep-survival {e['mean_ep_survival']:.1f} inherit "
                  f"{e['inherit_frac']:.3f}")
        print(f"gates: {results[f'C_{cand}']['gates']} -> pass "
              f"{results[f'C_{cand}']['pass']}")

    with open(os.path.join(OUT, "g2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwritten:", os.path.join(OUT, "g2_results.json"))


if __name__ == "__main__":
    main()
