"""Phase D1: registered causal-outcome suite for the intervention
(reviewer check 3).

Regenerates the Phase A home campaign (same seeds, same persisted arm
selections) with extended per-branch outcomes:

  break            any break in the 12-fission window
  run3_gb          run-3 after the first break (given a break)
  persist5         run reaches 5 after renewal (given a certified
                   episode)
  inherited_count  number of inherited boundaries (of 12)
  survived         branch did not die
  updates_pf       mean growth updates per fission

Consistency assert: regenerated joint-event arm means must equal the
stored Phase A values.

Registered decomposition (identity-exact, midpoint convention):
  q = b*r per state (b = break prob, r = run3|break), so
  q_up - q_down = db*r_mid + b_mid*dr  (no remainder).
Adjudication: if the db*r_mid (break-hazard) term contributes > 50% of
the summed effect in BOTH candidates, the paper-facing framing becomes
"the knob primarily controls hereditary stability".
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
import run_intervention as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_d1")
TAG = "intervention-2026-08-13"
N_MATRICES = 40
N_BRANCHES = 64
N_WORKERS = 12
N_BOOT = 1024
ARMS = ["up", "down", "noop", "random"]

_SEL = None


def extended_branch_outcomes(inh, died):
    """Registered per-branch outcomes from the 12 inheritance flags."""
    n = len(inh)
    out = {"break": np.nan, "run3_gb": np.nan, "persist5": np.nan,
           "inherited_count": float(np.sum(inh)),
           "survived": float(not died)}
    brk = bool((~inh).any()) if n else False
    out["break"] = float(brk)
    if not brk:
        return out
    t = int(np.argmin(inh))
    run = best = 0
    for v in inh[t + 1:]:
        run = run + 1 if v else 0
        best = max(best, run)
    out["run3_gb"] = float(best >= 3)
    if best >= 3:
        out["persist5"] = float(best >= 5)
    return out


def d1_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    rng = cohort._rng(RI._ENT, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, cohort.N_FISSIONS, rng)
    states = []
    for lm in cohort.LANDMARKS:
        if lm > traj["n_done"]:
            continue
        n = traj["daughters"][lm - 1]
        sel = _SEL[(cand, m, lm)]
        arms = {"noop": None, "up": tuple(sel["up"]),
                "down": tuple(sel["down"]), "random": tuple(sel["random"])}
        res = {}
        for name, swap in arms.items():
            recs = {k: [] for k in ["break", "run3_gb", "persist5",
                                    "inherited_count", "survived",
                                    "updates_pf", "joint"]}
            state = RI.apply_swap(n, swap)
            for b in range(N_BRANCHES):
                rb = cohort._rng(RI._ENT, 5, cand_i, m, lm, b)
                br = sim.run_fissions(state, beta, cand,
                                      cohort.HORIZON, rb)
                eo = extended_branch_outcomes(br["inherited"], br["died"])
                for k, v in eo.items():
                    recs[k].append(v)
                recs["updates_pf"].append(
                    float(np.mean(br["updates"])) if br["n_done"] else np.nan)
                recs["joint"].append(
                    float(Ft.joint_break_run3(br["inherited"])))
            res[name] = {k: float(np.nanmean(v)) for k, v in recs.items()}
        states.append({"matrix": m, "landmark": lm, "arms": res})
    return {"matrix": m, "candidate": cand, "states": states}


def main():
    global _SEL
    os.makedirs(OUT, exist_ok=True)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(HERE, "results_intervention",
                           "selections.pkl"), "rb") as f:
        _SEL = pickle.load(f)
    with open(os.path.join(HERE, "results_intervention",
                           "intervention_results.json")) as f:
        stored = json.load(f)["phase_a"]

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(d1_unit, jobs)
    print(f"D1 campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        rows = [s for u in units if u["candidate"] == cand
                for s in u["states"]]
        mats = np.array([r["matrix"] for r in rows])
        g = lambda arm, key: np.array([r["arms"][arm][key] for r in rows])

        # consistency: regenerated joint arm means == stored Phase A
        for arm in ARMS:
            regen = float(np.nanmean(g(arm, "joint")))
            ref = stored[cand]["arm_means"][arm]
            assert abs(regen - ref) < 1e-9, (cand, arm, regen, ref)
        print(f"cand {cand}: consistency with stored Phase A OK")

        entry = {"outcomes": {}}
        rng = np.random.default_rng(11235)
        for key in ["break", "run3_gb", "persist5", "inherited_count",
                    "survived", "updates_pf"]:
            u, d = g("up", key), g("down", key)
            ok = np.isfinite(u) & np.isfinite(d)
            diff = u[ok] - d[ok]
            ci = RI.boot_lower(diff, mats[ok], rng, n=N_BOOT)
            entry["outcomes"][key] = {
                "up": float(np.nanmean(u)), "down": float(np.nanmean(d)),
                "noop": float(np.nanmean(g("noop", key))),
                "random": float(np.nanmean(g("random", key))),
                "up_down": float(diff.mean()), "ci": ci,
                "n_states": int(ok.sum()),
            }

        # identity-exact midpoint decomposition per state, aggregated
        b_u, b_d = g("up", "break"), g("down", "break")
        r_u, r_d = g("up", "run3_gb"), g("down", "run3_gb")
        ok = np.isfinite(r_u) & np.isfinite(r_d)
        db = b_u[ok] - b_d[ok]
        dr = r_u[ok] - r_d[ok]
        b_mid = (b_u[ok] + b_d[ok]) / 2
        r_mid = (r_u[ok] + r_d[ok]) / 2
        term_b = db * r_mid
        term_r = b_mid * dr
        dq = g("up", "joint")[ok] - g("down", "joint")[ok]
        resid = float(np.abs(term_b + term_r - dq).max())
        share_b = float(term_b.sum() / (term_b.sum() + term_r.sum()))
        entry["decomposition"] = {
            "share_break_hazard": share_b,
            "share_renewal": 1 - share_b,
            "max_identity_residual": resid,
            "n_states": int(ok.sum()),
        }
        entry["stability_primary"] = bool(share_b > 0.5)
        results[cand] = entry

        print(f"\n=== D1 candidate {cand} ===")
        for key, v in entry["outcomes"].items():
            print(f"{key:16s} up {v['up']:8.3f} | noop {v['noop']:8.3f} | "
                  f"random {v['random']:8.3f} | down {v['down']:8.3f} | "
                  f"up-down {v['up_down']:+8.4f} "
                  f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
        d = entry["decomposition"]
        print(f"decomposition: break-hazard share {share_b:.3f}, renewal "
              f"share {1-share_b:.3f} (identity residual "
              f"{d['max_identity_residual']:.2e}, n={d['n_states']})")
        print(f"stability-primary (share > 0.5): {entry['stability_primary']}")

    results["adjudication_stability_primary_both"] = all(
        results[c]["stability_primary"] for c in cohort.CANDIDATES)
    print(f"\nADJUDICATION stability-primary in both candidates: "
          f"{results['adjudication_stability_primary_both']}")

    with open(os.path.join(OUT, "d1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("written:", os.path.join(OUT, "d1_results.json"))


if __name__ == "__main__":
    main()
