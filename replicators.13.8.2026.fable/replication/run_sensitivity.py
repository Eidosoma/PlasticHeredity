"""Step 2: target-definition sensitivity grid, v2 cohort persistence,
calibration report, and 4,096-permutation upgrade.

Registered grid: H_thr in {0.85, 0.90, 0.95} x run in {2, 3, 4} x
horizon in {8, 12, 16->12}. The F12 branches are 12 fissions long, so
horizon 16 is not evaluable and the grid uses {8, 10, 12}; 0.90/3/12 is
the registered primary and must reproduce the stored v2 numbers.

Frozen v2 and direct-8 predictions are evaluated by RANKING only
(overall and matrix-centered Spearman) against the branch-measured q at
each grid point — the frozen model was trained at the registered point,
so off-target calibration is neither expected nor claimed.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cohort
import registry_v2 as R2
from run_ablation import center_by

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_sensitivity")
FIG = os.path.join(OUT, "figures")
V2_TAG = "v2-conf-2026-08-13"
N_MATRICES = 200
N_WORKERS = 12
HALF = 32

THRESHOLDS = [0.85, 0.90, 0.95]
RUNS = [2, 3, 4]
HORIZONS = [8, 10, 12]
PRIMARY = (0.90, 3, 12)

BLUE = "#4878A8"
INK = "#33383D"


def break_then_run_q(H64, lens, thr, run_len, hor):
    """Per-branch event indicator for the generalized target."""
    B = len(H64)
    out = np.zeros(B, dtype=bool)
    f = H64[:, :hor] > thr
    for b in range(B):
        L = min(int(lens[b]), hor)
        flags = f[b, :L]
        if flags.all():
            continue
        t = int(np.argmin(flags))
        run = 0
        for v in flags[t + 1:]:
            run = run + 1 if v else 0
            if run >= run_len:
                out[b] = True
                break
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", V2_TAG)

    # ---- regenerate cohort: H sequences + features, persist ----------
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        hunits = pool.map(cohort.conf_h_sequences_unit, jobs)
    print(f"H sequences in {time.time()-t0:.0f}s")
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        funits = pool.map(cohort.conf_features_unit, jobs)
    print(f"features in {time.time()-t0:.0f}s")
    fmap = {}
    for u in funits:
        for s in u["states"]:
            fmap[(s["candidate"], s["matrix"], s["landmark"])] = \
                (s["X9"], s["X195"])
    table = []
    for u in hunits:
        for s in u["states"]:
            X9, X195 = fmap[(s["candidate"], s["matrix"], s["landmark"])]
            table.append({**s, "X9": X9, "X195": X195})
    with open(os.path.join(OUT, "v2_cohort.pkl"), "wb") as f:
        pickle.dump({"table": table, "tag": V2_TAG}, f, protocol=4)
    print(f"persisted v2 cohort: {len(table)} states")

    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        bundles = pickle.load(f)
    with open(os.path.join(HERE, "results_v2", "v2_results.json")) as f:
        stored_v2 = json.load(f)

    results = {"grid": [], "tag": V2_TAG,
               "note_horizon16": "branches are 12 fissions; 16 -> 12, "
                                 "grid uses horizons {8,10,12}"}
    for cand in cohort.CANDIDATES:
        rows = [r for r in table if r["candidate"] == cand]
        mats = np.array([r["matrix"] for r in rows])
        X9 = np.stack([r["X9"] for r in rows])
        X195 = np.stack([r["X195"] for r in rows])
        p = R2.predict_v2(bundles[cand], X9, X195)
        p_v2, p_d8 = p["v2"], p["direct8"]

        for thr in THRESHOLDS:
            for run_len in RUNS:
                for hor in HORIZONS:
                    qA = np.empty(len(rows))
                    qB = np.empty(len(rows))
                    for i, r in enumerate(rows):
                        ev = break_then_run_q(r["H64"], r["lens"],
                                              thr, run_len, hor)
                        qA[i] = ev[:HALF].mean()
                        qB[i] = ev[HALF:].mean()
                    q = (qA + qB) / 2
                    cA, cB = center_by(qA, mats), center_by(qB, mats)
                    entry = {
                        "candidate": cand, "thr": thr, "run": run_len,
                        "horizon": hor,
                        "prevalence": float(q.mean()),
                        "reliability": float(spearmanr(qA, qB).correlation),
                        "v2_overall": float(np.mean([
                            spearmanr(p_v2, qA).correlation,
                            spearmanr(p_v2, qB).correlation])),
                        "v2_centered": float(np.mean([
                            spearmanr(center_by(p_v2, mats), cA).correlation,
                            spearmanr(center_by(p_v2, mats), cB).correlation])),
                        "d8_overall": float(np.mean([
                            spearmanr(p_d8, qA).correlation,
                            spearmanr(p_d8, qB).correlation])),
                        "d8_centered": float(np.mean([
                            spearmanr(center_by(p_d8, mats), cA).correlation,
                            spearmanr(center_by(p_d8, mats), cB).correlation])),
                    }
                    results["grid"].append(entry)

        # ---- primary-point consistency check -------------------------
        prim = [e for e in results["grid"]
                if e["candidate"] == cand
                and (e["thr"], e["run"], e["horizon"]) == PRIMARY][0]
        stored_ov = np.mean(stored_v2[cand]["v2"]["overall"])
        assert abs(prim["v2_overall"] - stored_ov) < 5e-3, \
            (prim["v2_overall"], stored_ov)
        print(f"cand {cand}: primary-point consistency OK "
              f"({prim['v2_overall']:.4f} vs stored {stored_ov:.4f})")

        # ---- calibration at the registered point ---------------------
        qA = np.empty(len(rows)); qB = np.empty(len(rows))
        for i, r in enumerate(rows):
            ev = break_then_run_q(r["H64"], r["lens"], *PRIMARY)
            qA[i], qB[i] = ev[:HALF].mean(), ev[HALF:].mean()
        q = (qA + qB) / 2
        results[f"calibration_{cand}"] = {"p": p_v2.tolist(),
                                          "q": q.tolist()}

        # ---- 4096-permutation upgrade --------------------------------
        landmarks = np.array([r["landmark"] for r in rows])
        results[f"perm4096_v2_{cand}"] = R2.permutation_p(
            p_v2, q, mats, landmarks, n_perm=4096)
        print(f"cand {cand}: 4096-perm p (v2 cohort) = "
              f"{results[f'perm4096_v2_{cand}']:.6f}")

    # ---- 4096 permutations for v1-full on the stored 25x cohort ------
    with open(os.path.join(HERE, "results_25x", "conf_data.pkl"), "rb") as f:
        t25 = pickle.load(f)["table"]
    for cand in cohort.CANDIDATES:
        rows = [r for r in t25 if r["candidate"] == cand]
        p_full = np.array([r["p_full"] for r in rows])
        q = np.array([r["q"] for r in rows])
        mats = np.array([r["matrix"] for r in rows])
        landmarks = np.array([r["landmark"] for r in rows])
        results[f"perm4096_v1_25x_{cand}"] = R2.permutation_p(
            p_full, q, mats, landmarks, n_perm=4096)
        print(f"cand {cand}: 4096-perm p (25x, v1) = "
              f"{results[f'perm4096_v1_25x_{cand}']:.6f}")

    with open(os.path.join(OUT, "sensitivity_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures -----------------------------------------------------
    plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                         "axes.titlecolor": INK})
    for metric, fname in [("v2_overall", "fig_sensitivity_overall.png"),
                          ("v2_centered", "fig_sensitivity_centered.png")]:
        fig, axes = plt.subplots(2, len(HORIZONS),
                                 figsize=(3.2 * len(HORIZONS), 5.6))
        for ci, cand in enumerate(cohort.CANDIDATES):
            for hi, hor in enumerate(HORIZONS):
                ax = axes[ci, hi]
                M = np.zeros((len(THRESHOLDS), len(RUNS)))
                for e in results["grid"]:
                    if e["candidate"] == cand and e["horizon"] == hor:
                        M[THRESHOLDS.index(e["thr"]),
                          RUNS.index(e["run"])] = e[metric]
                im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
                for a in range(len(THRESHOLDS)):
                    for b in range(len(RUNS)):
                        ax.text(b, a, f"{M[a, b]:.2f}", ha="center",
                                va="center", fontsize=8,
                                color="white" if M[a, b] > 0.6 else INK)
                ax.set_xticks(range(len(RUNS)), [f"run {r}" for r in RUNS])
                ax.set_yticks(range(len(THRESHOLDS)),
                              [f"H>{t}" for t in THRESHOLDS])
                ax.set_title(f"cand {cand} · horizon {hor}")
        fig.suptitle(f"Frozen v2 {'overall' if 'overall' in metric else 'within-matrix'}"
                     " Spearman across target definitions")
        fig.colorbar(im, ax=axes, shrink=0.7, label="Spearman")
        fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
        plt.close(fig)

    # calibration figure with isotonic overlay
    from sklearn.isotonic import IsotonicRegression
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for j, cand in enumerate(cohort.CANDIDATES):
        c = results[f"calibration_{cand}"]
        p = np.array(c["p"]); q = np.array(c["q"])
        ax = axes[j]
        bins = np.clip((p * 10).astype(int), 0, 9)
        bx = [p[bins == b].mean() for b in range(10) if (bins == b).any()]
        by = [q[bins == b].mean() for b in range(10) if (bins == b).any()]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, q)
        xs = np.linspace(0, 1, 200)
        ax.plot([0, 1], [0, 1], "--", color=INK, lw=0.9,
                label="identity")
        ax.plot(bx, by, "o-", color=BLUE, lw=1.4, ms=5,
                label="10-bin reliability")
        ax.plot(xs, iso.predict(xs), color="#A8641E", lw=1.2,
                label="isotonic recalibration")
        ax.set_title(f"Candidate {cand}")
        ax.set_xlabel("Frozen v2 prediction")
        if j == 0:
            ax.set_ylabel("Branch-measured q")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Calibration of the frozen v2 coordinate (registered target)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_calibration_v2.png"))
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
