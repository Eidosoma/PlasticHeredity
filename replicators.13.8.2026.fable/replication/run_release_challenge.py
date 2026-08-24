"""Phase E: steer-release-challenge — basin test of the written state.

Adjudicates four registered outcomes for the controller-written
high-heredity concentrated state:
  (1) controller-maintained : decays during release
  (2) written-but-passive   : persists, no restoration after challenge
  (3) mode-attractor        : recovers the concentrated hereditary mode
                              at a different composition
  (4) composition-attractor : returns to the anchor itself

Registered design (plan of 2026-08-13; predictions recorded there):
- Write: regenerate model_down steered lineages (steering tag, 24
  matrices x 2 candidates x reps {0,1} = 96 states, 60 fissions).
- Precursor: frozen-v2 risk scores of written states, reported before
  challenge outcomes are examined.
- Release: 60 fissions of plain dynamics, spawn key (9, cand_i, m, rep).
  Mode-survival = release inheritance fraction >= 0.95;
  composition-hold = written-anchor H > 0.9 at release end.
- Natural controls: plain 120-fission noop lineages (steering stream
  domain 7); the full challenge protocol runs identically on them.
- Challenge anchor (registered refinement, fixed before running): the
  RELEASE-END composition (the state actually perturbed); similarity to
  the original written anchor is tracked as a secondary trace.
- Arms: none; random-k, k in {2,4,8,16} (remove k ~ multivariate
  hypergeometric of the composition, add k uniform over types; stream
  (11, cand_i, m, rep, k, is_natural)); adversarial (frozen model's
  score-maximizing single swap). 32 branches x 24 fissions per arm,
  stream (10, cand_i, m, rep, arm_i, b, is_natural).
- Per-branch classification: held (anchor H >= 0.7 throughout, > 0.9 at
  end); returned (departed < 0.7 then > 0.9 sustained >= 3 fissions);
  mode-recovered (neither, but final-6 inheritance >= 5/6 and final
  top-1 >= 0.45); lost.
- Matrix = resampling unit; 1,024 bootstraps. Basin radius = largest k
  with written (held + returned) >= 0.5.

Validity-check note (registered deviation): stored steering results
contain only 6-rep controller means, so the 2-rep write regeneration
cannot be asserted against them exactly; write validity rests on the
seed architecture (suite-tested bitwise determinism) plus a sanity
bound (mean written top-1 share >= 0.4, cf. the D2 panel value 0.50).
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
import registry_v2 as R2
import run_intervention as RI
import run_steering as RS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_release")
FIG = os.path.join(OUT, "figures")
TAG = "steering-2026-08-13"
N_MATRICES = 24
REPS = [0, 1]
RELEASE_FISSIONS = 60
CH_HORIZON = 24
N_BRANCHES = 32
N_WORKERS = 12
N_BOOT = 1024
DOSES = [2, 4, 8, 16]
ARMS = ["none", "k2", "k4", "k8", "k16", "adversarial"]
CLASSES = ["held", "returned", "mode_recovered", "lost"]

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
GREEN = "#3E7D5B"
INK = "#33383D"
CLASS_COLORS = {"held": BLUE, "returned": GREEN,
                "mode_recovered": "#B08A3E", "lost": GRAY}


def perturb(n, k, rng):
    """Registered recipe: remove k molecules ~ multivariate
    hypergeometric of the composition, add k uniform over types."""
    removed = rng.multivariate_hypergeometric(n, k)
    ne = n - removed
    add = np.bincount(rng.integers(0, sim.NG, k), minlength=sim.NG)
    ne = ne + add
    assert ne.sum() == n.sum() and (ne >= 0).all()
    return ne.astype(np.int64)


def classify_branch(anchor_h, inh, final_top1):
    """Registered four-outcome classification."""
    if len(anchor_h) == 0:
        return "lost"
    if (anchor_h >= 0.7).all() and anchor_h[-1] > 0.9:
        return "held"
    dep = np.where(anchor_h < 0.7)[0]
    if len(dep):
        run = 0
        for h in anchor_h[dep[0]:]:
            run = run + 1 if h > 0.9 else 0
            if run >= 3:
                return "returned"
    if len(inh) >= 6 and np.sum(inh[-6:]) >= 5 and final_top1 >= 0.45:
        return "mode_recovered"
    return "lost"


def challenge_state(state, anchor, beta, cand, cand_i, m, rep, hs_prefix,
                    natural):
    """Run all six arms x 32 branches from `state`; anchor = state
    itself (registered). Returns per-arm class counts."""
    nat = 1 if natural else 0
    bundle = RI._BUNDLES[cand]
    out = {}
    for arm_i, arm in enumerate(ARMS):
        if arm == "none":
            s0 = state
        elif arm == "adversarial":
            X9 = Ft.direct9(len(hs_prefix), 100, np.array(hs_prefix),
                            int(state.sum()))
            swap = RS.marginal_swap(state, beta, X9, bundle, +1)
            s0 = RI.apply_swap(state, swap)
        else:
            k = int(arm[1:])
            pr = cohort._rng(RI._ENT, 11, cand_i, m, rep, k, nat)
            s0 = perturb(state, k, pr)
        counts = {c: 0 for c in CLASSES}
        for b in range(N_BRANCHES):
            rb = cohort._rng(RI._ENT, 10, cand_i, m, rep, arm_i, b, nat)
            br = sim.run_fissions(s0, beta, cand, CH_HORIZON, rb)
            d = br["daughters"].astype(float)
            ah = np.array([sim.cosine_h(d[i], anchor.astype(float))
                           for i in range(len(d))])
            ft = (d[-1] / max(d[-1].sum(), 1)).max() if len(d) else 0.0
            counts[classify_branch(ah, br["inherited"], ft)] += 1
        out[arm] = counts
    return out


def unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    res = {"matrix": m, "candidate": cand, "written": [], "natural": []}
    for rep in REPS:
        # ---- write: regenerate the model_down steered lineage --------
        hs_w, final_holder = [], {}

        def wlog(f, n, swap, H, updates):
            hs_w.append(H)
            final_holder["n"] = n
        RS.steer_lineage(n0, beta, cand, cand_i, m, rep, "model_down",
                         log=wlog)
        written = final_holder["n"]
        w_top1 = float((written / written.sum()).max())
        X9w = Ft.direct9(len(hs_w), 100, np.array(hs_w),
                         int(written.sum()))
        risk = float(R2.predict_v2(
            RI._BUNDLES[cand], X9w[None, :],
            Ft.graph_state_195(written, beta)[None, :])["v2"][0])

        # ---- release: 60 free fissions ------------------------------
        rr = cohort._rng(RI._ENT, 9, cand_i, m, rep)
        rel = sim.run_fissions(written, beta, cand,
                               RELEASE_FISSIONS, rr)
        d = rel["daughters"].astype(float)
        wa = written.astype(float)
        rel_ah = np.array([sim.cosine_h(d[i], wa) for i in range(len(d))])
        rel_inh = rel["inherited"]
        release_end = rel["final"]
        hs_full = hs_w + list(rel["H"])
        rec = {
            "rep": rep, "risk": risk, "written_top1": w_top1,
            "rel_anchor_trace": rel_ah.tolist(),
            "rel_inherit_frac": float(np.mean(rel_inh)),
            "rel_end_anchor_h": float(rel_ah[-1]) if len(rel_ah) else 0.0,
            "mode_survival": bool(np.mean(rel_inh) >= 0.95),
            "composition_hold": bool(len(rel_ah)
                                     and rel_ah[-1] > 0.9),
            "challenge": challenge_state(release_end, release_end, beta,
                                         cand, cand_i, m, rep, hs_full,
                                         natural=False),
        }
        res["written"].append(rec)

        # ---- natural control: plain 120-fission lineage -------------
        rn = cohort._rng(RI._ENT, 7, cand_i, m, rep)
        nat = sim.run_fissions(n0, beta, cand, 120, rn)
        dn = nat["daughters"].astype(float)
        a60 = dn[59]
        nat_ah = np.array([sim.cosine_h(dn[i], a60)
                           for i in range(60, len(dn))])
        nat_state = nat["final"]
        res["natural"].append({
            "rep": rep,
            "rel_anchor_trace": nat_ah.tolist(),
            "rel_inherit_frac": float(np.mean(nat["inherited"][60:])),
            "challenge": challenge_state(nat_state, nat_state, beta,
                                         cand, cand_i, m, rep,
                                         list(nat["H"]), natural=True),
        })
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(unit, jobs)
    print(f"Phase E campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        wrecs = [r for u in cu for r in u["written"]]
        nrecs = [r for u in cu for r in u["natural"]]
        mats = np.array([u["matrix"] for u in cu for _ in u["written"]])

        # validity sanity
        assert float(np.mean([r["written_top1"] for r in wrecs])) >= 0.4

        entry = {
            "precursor_risk": {
                "mean": float(np.mean([r["risk"] for r in wrecs])),
                "q90": float(np.quantile([r["risk"] for r in wrecs], 0.9)),
            },
            "release": {
                "mode_survival_frac": float(np.mean(
                    [r["mode_survival"] for r in wrecs])),
                "composition_hold_frac": float(np.mean(
                    [r["composition_hold"] for r in wrecs])),
                "mean_inherit_frac": float(np.mean(
                    [r["rel_inherit_frac"] for r in wrecs])),
                "mean_end_anchor_h": float(np.mean(
                    [r["rel_end_anchor_h"] for r in wrecs])),
                "natural_mean_inherit_frac": float(np.mean(
                    [r["rel_inherit_frac"] for r in nrecs])),
            },
        }

        def frac(recs, arm, classes):
            tot = np.array([sum(r["challenge"][arm].values())
                            for r in recs], dtype=float)
            hit = np.array([sum(r["challenge"][arm][c] for c in classes)
                            for r in recs], dtype=float)
            return hit / tot

        arms_out = {}
        rng = np.random.default_rng(60606)
        for arm in ARMS:
            w_hr = frac(wrecs, arm, ["held", "returned"])
            n_hr = frac(nrecs, arm, ["held", "returned"])
            diff = w_hr - n_hr
            ci = RI.boot_lower(diff, mats, rng, n=N_BOOT)
            arms_out[arm] = {
                "written": {c: float(np.mean(frac(wrecs, arm, [c])))
                            for c in CLASSES},
                "natural": {c: float(np.mean(frac(nrecs, arm, [c])))
                            for c in CLASSES},
                "held_or_returned_written": float(np.mean(w_hr)),
                "held_or_returned_natural": float(np.mean(n_hr)),
                "diff_ci": ci,
            }
        entry["arms"] = arms_out

        radius = 0
        for k in DOSES:
            if arms_out[f"k{k}"]["held_or_returned_written"] >= 0.5:
                radius = k
        entry["basin_radius"] = radius

        # verdict adjudication (registered)
        if entry["release"]["mode_survival_frac"] < 0.5:
            verdict = "controller-maintained"
        else:
            comp_evidence = any(
                arms_out[f"k{k}"]["held_or_returned_written"] >= 0.5
                and arms_out[f"k{k}"]["diff_ci"][0] > 0 for k in DOSES)
            mode_evidence = any(
                arms_out[f"k{k}"]["held_or_returned_written"] < 0.5
                and arms_out[f"k{k}"]["written"]["mode_recovered"]
                >= 0.5 for k in DOSES)
            if comp_evidence and mode_evidence:
                verdict = "composition-attractor (finite basin) + mode-attractor beyond it"
            elif comp_evidence:
                verdict = "composition-attractor (finite basin)"
            elif mode_evidence:
                verdict = "mode-attractor"
            else:
                verdict = "written-but-passive"
        entry["verdict"] = verdict
        results[cand] = entry

        print(f"\n=== Phase E candidate {cand} ===")
        print(f"precursor risk mean {entry['precursor_risk']['mean']:.4f} "
              f"(q90 {entry['precursor_risk']['q90']:.4f})")
        r = entry["release"]
        print(f"release: mode-survival {r['mode_survival_frac']:.2f} | "
              f"comp-hold {r['composition_hold_frac']:.2f} | inherit "
              f"{r['mean_inherit_frac']:.3f} (natural "
              f"{r['natural_mean_inherit_frac']:.3f}) | end anchor-H "
              f"{r['mean_end_anchor_h']:.3f}")
        for arm in ARMS:
            a = arms_out[arm]
            w = a["written"]
            print(f"{arm:12s} W held {w['held']:.2f} ret "
                  f"{w['returned']:.2f} mode {w['mode_recovered']:.2f} "
                  f"lost {w['lost']:.2f} | H+R W "
                  f"{a['held_or_returned_written']:.2f} vs N "
                  f"{a['held_or_returned_natural']:.2f} | dCI "
                  f"[{a['diff_ci'][0]:+.2f},{a['diff_ci'][1]:+.2f}]")
        print(f"basin radius: k = {entry['basin_radius']} | "
              f"VERDICT: {verdict}")

    with open(os.path.join(OUT, "release_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures -----------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    # release traces
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for j, cand in enumerate(cohort.CANDIDATES):
        cu = [u for u in units if u["candidate"] == cand]
        ax = axes[j]
        W = np.array([r["rel_anchor_trace"] for u in cu
                      for r in u["written"]])
        N = np.array([r["rel_anchor_trace"] for u in cu
                      for r in u["natural"]])
        ax.plot(W.mean(axis=0), color=BLUE, lw=1.4,
                label="written state (anchor = written comp)")
        ax.plot(N.mean(axis=0), color=GRAY, lw=1.4,
                label="natural (anchor = comp at 60)")
        ax.axhline(0.9, color=AMBER, lw=0.8, ls="--")
        ax.set_title(f"Candidate {cand}")
        ax.set_xlabel("Release fission")
        if j == 0:
            ax.set_ylabel("Anchor similarity")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Release: the written state under free dynamics")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_release_traces.png"))
    plt.close(fig)

    # dose-response stacked classes
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ci_, cand in enumerate(cohort.CANDIDATES):
        for gi, (group, label) in enumerate(
                [("written", "written state"), ("natural", "natural")]):
            ax = axes[ci_, gi]
            bottoms = np.zeros(len(ARMS))
            for cls in CLASSES:
                vals = [results[cand]["arms"][a][group][cls]
                        for a in ARMS]
                ax.bar(range(len(ARMS)), vals, bottom=bottoms,
                       color=CLASS_COLORS[cls], width=0.65,
                       label=cls if (ci_ == 0 and gi == 0) else None)
                bottoms += np.array(vals)
            ax.set_xticks(range(len(ARMS)), ARMS, fontsize=7,
                          rotation=30)
            ax.set_title(f"cand {cand} · {label}", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Challenge outcome classes by perturbation dose")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_challenge_doses.png"))
    plt.close(fig)
    print("written:", os.path.join(OUT, "release_results.json"))


if __name__ == "__main__":
    main()
