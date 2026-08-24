"""C7 campaign under code-faithful Phi_R (PHIR_C7_PREREGISTRATION.md).

Four arms x 100 matched seeds, coarse universe: control / max-Phi_R /
min-Phi_R / random-edit. Scoring via phi_r_point.WindowFit on the
trailing WINDOW=100 steps; candidates = all single-molecule adds +
deletes (as interventions.py). The equality gate runs before the
campaign and aborts on failure.

Usage: interventions_phir.py [smoke]   (smoke = seeds 0-2 only)
Outputs: results/interv_phir_rows.pkl, results/interv_phir_summary.json
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, simulate
from interventions import replicator_metrics
from phi import phi_r_local
from phi_r_code import phi_r_code_local
from phi_r_point import WindowFit, test_phi_r_point

ROOT = Path(__file__).parent.parent
WINDOW = 100
ARMS = ("control", "max", "min", "random")
SMOKE = "smoke" in sys.argv[1:]
N_RUNS = 3 if SMOKE else 100


def enumerate_candidates(counts, n_types):
    cands, edits = [], []
    for i in range(n_types):
        c = counts.copy(); c[i] += 1
        cands.append(c); edits.append(f"add{i}")
        if counts[i] > 0:
            c = counts.copy(); c[i] -= 1
            cands.append(c); edits.append(f"del{i}")
    return np.array(cands), edits


def make_intervention(arm, seed, edit_log):
    if arm == "control":
        return None

    def intervene(counts, history, beta, params, rng, gen):
        if len(history) < WINDOW:
            return counts
        cands, edits = enumerate_candidates(counts, params.n_types)
        if arm == "random":
            k = int(np.random.default_rng([seed, 7919, gen])
                    .integers(len(cands)))
        else:
            fit = WindowFit(np.array(history[-WINDOW:]))
            if not fit.ok:
                return counts
            scores = fit.score_candidates(counts, cands)
            if not np.isfinite(scores).any():
                return counts
            k = int(np.nanargmax(scores) if arm == "max"
                    else np.nanargmin(scores))
        edit_log.append((gen, edits[k]))
        return cands[k]

    return intervene


def one_job(args):
    arm, seed = args
    p = GardParams(dt=0.4, max_events_per_step=24.0)
    edit_log = []
    traj = simulate(seed=seed, p=p,
                    intervention=make_intervention(arm, seed, edit_log))
    m = replicator_metrics(traj.counts, traj.fission_steps, seed=seed)
    sr = m.pop("sr")
    gen = traj.generation
    m["gen_prob"] = [float(sr[gen == g].mean()) for g in range(p.n_gen)]
    m["phi_r_mean"] = float(np.nanmean(
        phi_r_code_local(traj.counts.astype(float), "full")))
    psi, _ = phi_r_local(traj.counts, mib_seed=seed)
    m["psi_mean"] = float(np.nanmean(psi))
    m["n_edits"] = len(edit_log)
    m["edits"] = edit_log
    m["arm"], m["seed"] = arm, seed
    return m


def mw(rows, a, b, key, alternative):
    va = np.array([r[key] for r in rows if r["arm"] == a], float)
    vb = np.array([r[key] for r in rows if r["arm"] == b], float)
    return float(stats.mannwhitneyu(va, vb, alternative=alternative,
                                    nan_policy="omit").pvalue)


def arm_mean(rows, arm, key):
    v = np.array([r[key] for r in rows if r["arm"] == arm], float)
    return f"{np.nanmean(v):.3f}±{np.nanstd(v):.3f}"


def summarize(rows):
    out = {"arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {k: arm_mean(rows, arm, k) for k in
                            ("persistence", "probability",
                             "consistency_1k", "episode_mean",
                             "phi_r_mean", "psi_mean", "n_edits")}
    t1 = {"max_gt_min": mw(rows, "max", "min", "phi_r_mean", "greater"),
          "max_gt_control": mw(rows, "max", "control", "phi_r_mean",
                               "greater"),
          "min_lt_control": mw(rows, "min", "control", "phi_r_mean",
                               "less")}
    t1["gate_passes"] = bool(t1["max_gt_min"] < 0.05)
    out["T1_manipulation_check"] = t1
    t2 = {}
    for key in ("persistence", "probability"):
        t2[f"{key}_max_gt_control"] = mw(rows, "max", "control", key,
                                         "greater")
        t2[f"{key}_min_lt_control"] = mw(rows, "min", "control", key,
                                         "less")
    t2["all_four_pass"] = bool(all(v < 0.05 for v in t2.values()))
    out["T2_primary_sr_outcomes"] = t2
    out["T3_secondary"] = {
        k: {"max_vs_control": mw(rows, "max", "control", k, "two-sided"),
            "min_vs_control": mw(rows, "min", "control", k, "two-sided")}
        for k in ("consistency_1k", "episode_mean")}
    out["T4_random_validity"] = {
        k: mw(rows, "random", "control", k, "two-sided")
        for k in ("persistence", "probability")}
    out["T4_valid"] = bool(all(v > 0.05
                               for v in out["T4_random_validity"].values()))
    out["T5_cross_instrument_psi"] = {
        "max_vs_min": mw(rows, "max", "min", "psi_mean", "two-sided")}
    if t1["gate_passes"] and t2["all_four_pass"]:
        verdict = "C7 REPLICATES under the code instrument"
    elif t1["gate_passes"]:
        verdict = ("Phi_R steerable but SR outcomes do not follow — "
                   "C7 not replicated; Phi_R->SR causal arrow constrained")
    else:
        verdict = ("manipulation-check gate FAILED — no working Phi_R "
                   "manipulation at one-molecule granularity; C7 "
                   "untestable at this intervention strength")
    out["frozen_table_verdict"] = verdict
    return out


def main():
    with np.load(ROOT / "results" / "runs_coarse" / "run_000.npz") as d:
        gate_diff = test_phi_r_point(d["counts"].astype(float)[:150])
    print(f"equality gate: max diff {gate_diff:.2e} — PASS", flush=True)

    jobs = [(arm, seed) for arm in ARMS for seed in range(N_RUNS)]
    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(one_job, jobs)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s", flush=True)

    tag = "_smoke" if SMOKE else ""
    with open(ROOT / "results" / f"interv_phir_rows{tag}.pkl", "wb") as f:
        pickle.dump(rows, f, protocol=4)
    summary = summarize(rows)
    (ROOT / "results" / f"interv_phir_summary{tag}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
