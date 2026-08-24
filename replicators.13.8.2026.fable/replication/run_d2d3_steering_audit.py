"""Phase D2+D3: long-run cost panel and controller action audit.

Regenerates the steering campaign deterministically (same seeds; the
overall outcome means must reproduce results_steering/steering_results
.json exactly) with extended logging, plus a 120-fission horizon
extension for {model_down, noop} on reps {0, 1}.

D2 registered predictions (pass/fail scored):
  P1 down.entropy < noop.entropy          (final-10-fission mean)
  P2 down.occupied < noop.occupied
  P3 down.top1 > noop.top1
  P4 down cross-lineage final similarity > noop
  P5 up.entropy > noop.entropy
  P6 up.occupied > noop.occupied
  P7 extinction = 0 in all arms
  P8 extension: model_down inheritance fraction over fissions 61-120
     > 0.9 and all extension lineages survive
Growth updates and catalytic throughput: measured, no registered
direction (exploratory).

D3 (characterization, no gates): swap-type concentration, cycling and
repeat rates, off-manifold fraction (frozen v2 PCA coordinates outside
the natural-state [0.5%, 99.5%] envelope from the v2 confirmation
cohort), and model-vs-rule action agreement.
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
import run_intervention as RI
import run_steering as RS
import run_mechanism as RM
from run_ablation import BETA_IDX

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_d2d3")
FIG = os.path.join(OUT, "figures")
TAG = "steering-2026-08-13"
N_MATRICES = 24
N_REPS = 6
EXT_HORIZON = 120
N_WORKERS = 12
EPS = 1e-12

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
INK = "#33383D"

_ENV_LO = _ENV_HI = None      # PCA envelope, set in parent (fork)


class Logger:
    def __init__(self, beta, bundle, controller):
        self.beta = beta
        self.bundle = bundle
        self.controller = controller
        self.swaps = []
        self.ent = []
        self.occ = []
        self.top1 = []
        self.thr = []
        self.upd = []
        self.out_env = 0
        self.n_seen = 0
        self.rule_match = {"exact": 0, "add": 0, "rem": 0, "n": 0}
        self.final_n = None

    def __call__(self, f, n, swap, H, updates):
        N = max(int(n.sum()), 1)
        x = n / N
        xp = x[n > 0]
        self.ent.append(float(-np.sum(xp * np.log(xp + EPS))))
        self.occ.append(int((n > 0).sum()))
        self.top1.append(float(x.max()))
        bn = 1.0 + (self.beta @ n) / N
        self.thr.append(float(np.sum(sim.KF * sim.RHO * N * bn)))
        self.upd.append(updates)
        z = self.bundle["pca"].transform(
            self.bundle["scb"].transform(
                Ft.graph_state_195(n, self.beta)[BETA_IDX][None, :]))[0]
        self.n_seen += 1
        if ((z < _ENV_LO) | (z > _ENV_HI)).any():
            self.out_env += 1
        self.final_n = n.copy()
        if swap is not None:
            self.swaps.append(swap)
            if self.controller in ("model_up", "model_down"):
                up, dn = RM.rule_swaps(n, self.beta)
                rule = up if self.controller == "model_up" else dn
                self.rule_match["n"] += 1
                self.rule_match["exact"] += int(swap == rule)
                self.rule_match["rem"] += int(swap[0] == rule[0])
                self.rule_match["add"] += int(swap[1] == rule[1])


def audit_unit(args):
    m, cand = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta, n0 = cohort.matrix_and_init(RI._ENT, m)
    bundle = RI._BUNDLES[cand]
    out = {}
    for controller in RS.CONTROLLERS:
        recs = []
        for rep in range(N_REPS):
            lg = Logger(beta, bundle, controller)
            stats = RS.steer_lineage(n0, beta, cand, cand_i, m, rep,
                                     controller, log=lg)
            swaps = lg.swaps
            rec = {
                "stats": stats,
                "ent10": float(np.mean(lg.ent[-10:])),
                "occ10": float(np.mean(lg.occ[-10:])),
                "top1_10": float(np.mean(lg.top1[-10:])),
                "thr10": float(np.mean(lg.thr[-10:])),
                "upd_mean": float(np.mean(lg.upd)),
                "died": len(lg.ent) < RS.HORIZON,
                "final_n": lg.final_n,
                "out_env_frac": lg.out_env / max(lg.n_seen, 1),
                "n_swaps": len(swaps),
                "n_distinct": len(set(swaps)),
                "repeat_rate": float(np.mean(
                    [swaps[i] == swaps[i - 1]
                     for i in range(1, len(swaps))])) if len(swaps) > 1
                else np.nan,
                "cycle_rate": float(np.mean(
                    [any(swaps[i] == (swaps[j][1], swaps[j][0])
                         for j in range(max(0, i - 3), i))
                     for i in range(1, len(swaps))])) if len(swaps) > 1
                else np.nan,
                "rule_match": dict(lg.rule_match),
            }
            recs.append(rec)
        out[controller] = recs

    # horizon extension: reps {0,1}, {model_down, noop}, 120 fissions
    ext = {}
    for controller in ("model_down", "noop"):
        e = []
        for rep in range(2):
            hs_frac = {}
            inh_log = []
            def xlog(f, n, swap, H, updates, _log=inh_log):
                _log.append(H > sim.H_THRESH)
            RS.steer_lineage(n0, beta, cand, cand_i, m, rep, controller,
                             log=xlog, horizon=EXT_HORIZON)
            tail = inh_log[60:]
            e.append({"survived": len(inh_log) == EXT_HORIZON,
                      "tail_inherit_frac": float(np.mean(tail))
                      if tail else np.nan})
        ext[controller] = e
    return {"matrix": m, "candidate": cand, "recs": out, "ext": ext}


def cross_lineage_sim(finals):
    sims = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            sims.append(sim.cosine_h(finals[i].astype(float),
                                     finals[j].astype(float)))
    return float(np.mean(sims))


def main():
    global _ENV_LO, _ENV_HI
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", TAG)
    with open(os.path.join(HERE, "results_mechanism",
                           "mechanism_results.json")) as f:
        fr = json.load(f)["frozen_rule"]
    RM._RULE = (fr["quantity"], fr["orientation"])
    with open(os.path.join(HERE, "results_steering",
                           "steering_results.json")) as f:
        stored = json.load(f)

    # natural-state PCA envelope from the v2 confirmation cohort
    with open(os.path.join(HERE, "results_sensitivity",
                           "v2_cohort.pkl"), "rb") as f:
        vc = pickle.load(f)["table"]
    Z = {}
    for cand in cohort.CANDIDATES:
        X = np.stack([r["X195"] for r in vc if r["candidate"] == cand])
        b = RI._BUNDLES[cand]
        Z[cand] = b["pca"].transform(b["scb"].transform(X[:, BETA_IDX]))
    zall = np.vstack(list(Z.values()))
    _ENV_LO = np.quantile(zall, 0.005, axis=0)
    _ENV_HI = np.quantile(zall, 0.995, axis=0)

    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(audit_unit, jobs)
    print(f"audit campaign in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]

        # consistency with stored steering results
        for key, skey in [("episodes", "episodes"), ("breaks", "breaks"),
                          ("inherit_frac", "inherit_frac"),
                          ("longest_run", "longest_run")]:
            for ctrl in RS.CONTROLLERS:
                regen = float(np.mean([
                    np.mean([r["stats"][key] for r in u["recs"][ctrl]])
                    for u in cu]))
                ref = stored[cand]["outcome_means"][skey][ctrl]
                assert abs(regen - ref) < 1e-9, (cand, ctrl, key)
        print(f"cand {cand}: consistency with stored steering OK")

        panel = {}
        for ctrl in RS.CONTROLLERS:
            rr = [r for u in cu for r in u["recs"][ctrl]]
            panel[ctrl] = {
                "entropy": float(np.mean([r["ent10"] for r in rr])),
                "occupied": float(np.mean([r["occ10"] for r in rr])),
                "top1": float(np.mean([r["top1_10"] for r in rr])),
                "throughput": float(np.mean([r["thr10"] for r in rr])),
                "updates_pf": float(np.mean([r["upd_mean"] for r in rr])),
                "extinct": int(sum(r["died"] for r in rr)),
                "cross_lineage_sim": float(np.mean([
                    cross_lineage_sim([r["final_n"]
                                       for r in u["recs"][ctrl]])
                    for u in cu])),
                "out_env_frac": float(np.mean([r["out_env_frac"]
                                               for r in rr])),
            }
        ext_tail = [e["tail_inherit_frac"] for u in cu
                    for e in u["ext"]["model_down"]]
        ext_surv = all(e["survived"] for u in cu
                       for e in u["ext"]["model_down"])
        p = panel
        preds = {
            "P1_down_entropy_lower": p["model_down"]["entropy"]
            < p["noop"]["entropy"],
            "P2_down_occupied_lower": p["model_down"]["occupied"]
            < p["noop"]["occupied"],
            "P3_down_top1_higher": p["model_down"]["top1"]
            > p["noop"]["top1"],
            "P4_down_convergence_higher":
                p["model_down"]["cross_lineage_sim"]
                > p["noop"]["cross_lineage_sim"],
            "P5_up_entropy_higher": p["model_up"]["entropy"]
            > p["noop"]["entropy"],
            "P6_up_occupied_higher": p["model_up"]["occupied"]
            > p["noop"]["occupied"],
            "P7_no_extinction": all(p[c]["extinct"] == 0
                                    for c in RS.CONTROLLERS),
            "P8_extension_stable": bool(ext_surv
                                        and np.nanmean(ext_tail) > 0.9),
        }
        audit = {}
        for ctrl in ("model_up", "model_down"):
            rr = [r for u in cu for r in u["recs"][ctrl]]
            rm = {k: sum(r["rule_match"][k] for r in rr)
                  for k in ("exact", "add", "rem", "n")}
            audit[ctrl] = {
                "distinct_swaps_per_lineage": float(np.mean(
                    [r["n_distinct"] for r in rr])),
                "repeat_rate": float(np.nanmean(
                    [r["repeat_rate"] for r in rr])),
                "cycle_rate": float(np.nanmean(
                    [r["cycle_rate"] for r in rr])),
                "rule_agreement": {k: rm[k] / max(rm["n"], 1)
                                   for k in ("exact", "add", "rem")},
                "out_env_frac": panel[ctrl]["out_env_frac"],
            }
        results[cand] = {"panel": panel, "predictions": preds,
                         "ext_tail_inherit": float(np.nanmean(ext_tail)),
                         "audit": audit}

        print(f"\n=== D2 panel candidate {cand} (final-10-fission means) ===")
        hdr = ["entropy", "occupied", "top1", "throughput", "updates_pf",
               "cross_lineage_sim", "out_env_frac", "extinct"]
        for ctrl in RS.CONTROLLERS:
            print(f"{ctrl:11s} " + " | ".join(
                f"{k} {panel[ctrl][k]:.3f}" if not isinstance(
                    panel[ctrl][k], int) else f"{k} {panel[ctrl][k]}"
                for k in hdr))
        print("predictions:", preds,
              f"| ext tail inherit {np.nanmean(ext_tail):.3f}")
        print("D3 audit:", json.dumps(audit[cand] if cand in audit
                                      else audit, default=str)[:400]
              if False else "")
        for ctrl in ("model_up", "model_down"):
            a = audit[ctrl]
            print(f"audit {ctrl}: distinct {a['distinct_swaps_per_lineage']:.1f} "
                  f"| repeat {a['repeat_rate']:.3f} | cycle "
                  f"{a['cycle_rate']:.3f} | rule agree exact "
                  f"{a['rule_agreement']['exact']:.3f} add "
                  f"{a['rule_agreement']['add']:.3f} rem "
                  f"{a['rule_agreement']['rem']:.3f} | out-env "
                  f"{a['out_env_frac']:.3f}")

    results["all_predictions_pass"] = all(
        all(results[c]["predictions"].values())
        for c in cohort.CANDIDATES)
    print(f"\nD2 PREDICTION SCORECARD: "
          f"{'ALL PASS' if results['all_predictions_pass'] else 'MIXED'}")

    with open(os.path.join(OUT, "d2d3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # cost-panel figure
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    metrics = [("entropy", "composition entropy"),
               ("occupied", "occupied species"),
               ("top1", "top-1 share"),
               ("cross_lineage_sim", "cross-lineage similarity")]
    order = ["model_down", "random", "noop", "model_up"]
    colors = [AMBER, GRAY, GRAY, BLUE]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for ci, cand in enumerate(cohort.CANDIDATES):
        for mi, (key, label) in enumerate(metrics):
            ax = axes[ci, mi]
            vals = [results[cand]["panel"][c][key] for c in order]
            ax.bar(range(4), vals, color=colors, width=0.6)
            ax.set_xticks(range(4), ["down", "rand", "noop", "up"],
                          fontsize=7)
            ax.set_title(f"cand {cand} · {label}", fontsize=8)
    fig.suptitle("Long-run costs of steering (final-10-fission means)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cost_panel.png"))
    plt.close(fig)
    print("written:", os.path.join(OUT, "d2d3_results.json"))


if __name__ == "__main__":
    main()
