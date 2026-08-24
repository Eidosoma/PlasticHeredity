"""Phase H — STRICT_BREAK_COHERENT8_DISTINCT occurrence replication.

Preregistered in STRICT8_REPLICATION_PREREGISTRATION.md (sealed with
this file's SHA-256 in results_strict8_occurrence/SEAL.json BEFORE any
scientific matrix was generated). Occurrence experiment only: no
prediction, no intervention, no model fitting, no Phase G outputs, no
external-agent code or data.

Usage:
    python3 run_strict8_occurrence_replication.py --smoke   # I/O+replay only
    python3 run_strict8_occurrence_replication.py           # full module
"""

import hashlib
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

import sim
import cohort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_strict8_occurrence")

# ---- frozen registration constants -----------------------------------
TAG = "2026-08-14"
DOMAIN = 25
N_MAT = 200
LANDMARKS = [20, 35, 50, 65, 80]
N_BRANCH = 128
HALF_A = range(0, 64)          # prospective half assignment by index
HALF_B = range(64, 128)
HORIZON = 32
MAIN_FISSIONS = 100
INHERIT_T = 0.9                # H > 0.9 inherited; H <= 0.9 break
COHERE_T = 0.9                 # pairwise strictly > 0.9
DISTINCT_T = 0.85              # anchor inclusively <= 0.85
AUDIT_MOD = 997                # outcome-blind audit sample rule
BOOT_N = 4096
BOOT_SEED = 41
EXTERNAL = {("02", "A"): 0.01869, ("02", "B"): 0.01809,
            ("03", "A"): 0.02089, ("03", "B"): 0.02109}

_ENT = cohort.domain_entropy("strict8", TAG)


# ---- endpoint predicates (threshold semantics live ONLY here) --------

def is_inherited(h):
    return h > INHERIT_T


def pair_coherent(h):
    return h > COHERE_T


def anchor_distinct(h):
    return h <= DISTINCT_T


def _norm_rows(M):
    M = np.asarray(M, dtype=np.float64)
    nrm = np.sqrt((M * M).sum(axis=1))
    nrm = np.where(nrm < 1e-7, np.inf, nrm)   # zero vector -> cosine 0
    return M / nrm[:, None]


def classify_future(hs, parents, daughters, horizon=HORIZON):
    """Frozen STRICT_BREAK_COHERENT8_DISTINCT classifier.

    hs: float64 H per realized boundary (t = 1..k, 0-indexed array);
    parents/daughters: per-boundary full compositions (selected
    daughter per the candidate contract). Returns the registered
    per-future record; eligible windows are ALL 8-consecutive-
    inherited windows strictly after the first break certified within
    the horizon, in increasing start order (overlaps included).
    """
    hs = np.asarray(hs, dtype=np.float64)
    k = int(len(hs))
    rec = {"k": k, "first_break": 0, "break_within": False,
           "run8_after": False, "eligible": [], "positive": False,
           "primary_r": 0, "cert": 0, "run_len": 0,
           "coh_first": None, "dis_first": None,
           "minpair_first": np.nan, "maxanchor_first": np.nan}
    inh = np.array([is_inherited(h) for h in hs], dtype=bool)
    if inh.all() or k == 0:
        return rec                       # no-break future -> negative
    b = int(np.argmax(~inh)) + 1         # first break boundary (1-idx)
    rec["first_break"] = b
    rec["break_within"] = True
    lim = min(k, horizon)
    if b + 8 > lim:                      # no room to certify
        return rec
    p_old = np.asarray(parents[b - 1], dtype=np.float64)
    n_old = float(np.sqrt(p_old @ p_old))
    D = _norm_rows(daughters[:lim])
    if n_old < 1e-7:
        anchor = np.zeros(lim)
    else:
        anchor = np.clip(D @ (p_old / n_old), 0.0, 1.0)
    G = np.clip(D @ D.T, 0.0, 1.0)
    iu = np.triu_indices(8, 1)
    for r in range(b + 1, lim - 7 + 1):  # 1-indexed start, cert r+7<=lim
        if not inh[r - 1:r + 7].all():
            continue
        rec["run8_after"] = True
        sub = G[r - 1:r + 7, r - 1:r + 7]
        minpair = float(sub[iu].min())
        maxanch = float(anchor[r - 1:r + 7].max())
        rec["eligible"].append((r, minpair, maxanch))
        if rec["coh_first"] is None:
            rec["coh_first"] = bool(all(pair_coherent(v)
                                        for v in sub[iu]))
            rec["dis_first"] = bool(all(anchor_distinct(v)
                                        for v in anchor[r - 1:r + 7]))
            rec["minpair_first"] = minpair
            rec["maxanchor_first"] = maxanch
        ok = all(pair_coherent(v) for v in sub[iu]) and \
            all(anchor_distinct(v) for v in anchor[r - 1:r + 7])
        if ok and not rec["positive"]:   # first qualifying = primary
            rec["positive"] = True
            rec["primary_r"] = r
            rec["cert"] = r + 7
            s = r
            while s - 1 >= 1 and inh[s - 2]:
                s -= 1
            e = r + 7
            while e < k and inh[e]:
                e += 1
            rec["run_len"] = e - s + 1   # observable run containing it
    return rec


# ---- cohort unit -----------------------------------------------------

def _rng(*key):
    return cohort._rng(_ENT, DOMAIN, *key)


def unit(args):
    """(matrix, candidate, mode, n_mat, n_branch): one full unit.
    mode 'full' returns records + archives; mode 'hash' returns only
    the unit hash (used by the deterministic replay pass)."""
    m, cand, mode, n_mat, n_branch = args
    cand_i = cohort.CANDIDATES.index(cand)
    beta = sim.make_beta(_rng(0, m))
    n0 = sim.make_initial_state(_rng(1, m))
    traj = sim.run_fissions(n0, beta, cand, MAIN_FISSIONS,
                            _rng(2, cand_i, m))
    h = hashlib.sha256()
    states, futures, htraces, archives = [], [], [], []
    for li, lm in enumerate(LANDMARKS):
        if lm > traj["n_done"]:
            continue                     # frozen contract: drop, no retry
        restored = traj["daughters"][lm - 1]
        if mode == "full":
            states.append({"landmark": lm, "restored":
                           restored.astype(np.int16)})
        for b in range(n_branch):
            rb = _rng(3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, HORIZON, rb)
            rec = classify_future(br["H"], br["parents"],
                                  br["daughters"])
            h.update(np.ascontiguousarray(br["H"]).tobytes())
            h.update(repr((lm, b, rec["k"], rec["first_break"],
                           rec["positive"], rec["primary_r"],
                           rec["cert"], len(rec["eligible"])))
                     .encode())
            if mode != "full":
                continue
            flat = ((cand_i * n_mat + m) * 5 + li) * n_branch + b
            row = {"landmark": lm, "branch": b, **rec}
            futures.append(row)
            htraces.append(np.asarray(br["H"], dtype=np.float32))
            arch = None
            if rec["positive"]:
                r = rec["primary_r"]
                arch = {"kind": "positive", "landmark": lm, "branch": b,
                        "p_old": br["parents"][rec["first_break"] - 1]
                        .astype(np.int16),
                        "daughters": br["daughters"][r - 1:r + 7]
                        .astype(np.int16)}
            if flat % AUDIT_MOD == 0:
                arch2 = {"kind": "audit", "landmark": lm, "branch": b,
                         "parents": br["parents"].astype(np.int16),
                         "daughters": br["daughters"].astype(np.int16)}
                archives.append(arch2)
            if arch is not None:
                archives.append(arch)
    out = {"matrix": m, "candidate": cand,
           "unit_hash": h.hexdigest(), "n_done": int(traj["n_done"])}
    if mode == "full":
        out.update({"states": states, "futures": futures,
                    "H": htraces, "archives": archives})
    return out


def campaign_hash(units):
    h = hashlib.sha256()
    for u in sorted(units, key=lambda u: (u["candidate"], u["matrix"])):
        h.update(u["unit_hash"].encode())
    return h.hexdigest()


# ---- analysis --------------------------------------------------------

def analyze(units, n_branch):
    half_of = {b: ("A" if b in HALF_A else "B")
               for b in range(n_branch)}
    if n_branch != N_BRANCH:             # smoke only: first half A
        half_of = {b: ("A" if b < n_branch // 2 else "B")
                   for b in range(n_branch)}
    results = {"cells": {}, "descriptive": {}, "components": {}}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        for half in ("A", "B"):
            per_mat = {}
            per_state = {}
            timing = {"first_break": [], "run_start": [], "cert": []}
            for u in cu:
                ev = tot = 0
                for f in u["futures"]:
                    if half_of[f["branch"]] != half:
                        continue
                    tot += 1
                    key = (u["matrix"], f["landmark"])
                    st = per_state.setdefault(key, [0, 0])
                    st[1] += 1
                    if f["positive"]:
                        ev += 1
                        st[0] += 1
                        timing["first_break"].append(f["first_break"])
                        timing["run_start"].append(f["primary_r"])
                        timing["cert"].append(f["cert"])
                per_mat[u["matrix"]] = (ev, tot)
            events = sum(e for e, _ in per_mat.values())
            futs = sum(t for _, t in per_mat.values())
            rate = events / futs if futs else np.nan
            rng = np.random.default_rng(BOOT_SEED)
            mats = sorted(per_mat)
            boots = []
            for _ in range(BOOT_N):
                pick = rng.choice(mats, size=len(mats), replace=True)
                e = sum(per_mat[mm][0] for mm in pick)
                t = sum(per_mat[mm][1] for mm in pick)
                boots.append(e / t if t else np.nan)
            lo, hi = (float(np.nanquantile(boots, 0.025)),
                      float(np.nanquantile(boots, 0.975)))
            ev_mats = [mm for mm in mats if per_mat[mm][0] > 0]
            ev_states = [k for k, v in per_state.items() if v[0] > 0]
            counts = [per_mat[mm][0] for mm in mats]
            cell = {
                "rate": rate, "ci95": [lo, hi], "events": events,
                "futures": futs,
                "matrices_with_event": len(ev_mats),
                "matrices_total": len(mats),
                "matrix_fraction": len(ev_mats) / max(len(mats), 1),
                "states_with_event": len(ev_states),
                "states_total": len(per_state),
                "state_fraction": len(ev_states)
                / max(len(per_state), 1),
                "median_events_per_matrix": float(np.median(counts)),
                "max_events_per_matrix": int(max(counts)) if counts
                else 0,
                "timing_median": {k: float(np.median(v)) if v else None
                                  for k, v in timing.items()},
                "timing_range": {k: [int(min(v)), int(max(v))] if v
                                 else None for k, v in timing.items()},
            }
            results["cells"][f"{cand}/{half}"] = cell
        # descriptive (per candidate, both halves)
        st_rate = {}
        mat_rate = {}
        stA, stB = {}, {}
        comp = {"break_within": 0, "run8_after": 0, "coh_first": 0,
                "dis_first": 0, "positive": 0, "n": 0,
                "minpair": [], "maxanchor": [], "run_len": []}
        for u in cu:
            for f in u["futures"]:
                key = (u["matrix"], f["landmark"])
                st_rate.setdefault(key, [0, 0])
                mat_rate.setdefault(u["matrix"], [0, 0])
                st_rate[key][1] += 1
                mat_rate[u["matrix"]][1] += 1
                tgt = stA if half_of[f["branch"]] == "A" else stB
                tt = tgt.setdefault(key, [0, 0])
                tt[1] += 1
                comp["n"] += 1
                for c in ("break_within", "run8_after", "positive"):
                    comp[c] += int(bool(f[c]))
                for c in ("coh_first", "dis_first"):
                    comp[c] += int(bool(f[c]))
                if np.isfinite(f["minpair_first"]):
                    comp["minpair"].append(f["minpair_first"])
                    comp["maxanchor"].append(f["maxanchor_first"])
                if f["positive"]:
                    st_rate[key][0] += 1
                    mat_rate[u["matrix"]][0] += 1
                    tt[0] += 1
                    comp["run_len"].append(f["run_len"])
        keys = sorted(set(stA) & set(stB))
        ra = np.array([stA[k][0] / stA[k][1] for k in keys])
        rb = np.array([stB[k][0] / stB[k][1] for k in keys])
        agree = float(np.corrcoef(ra, rb)[0, 1]) \
            if len(keys) > 2 and ra.std() > 0 and rb.std() > 0 else None
        results["descriptive"][cand] = {
            "equal_state_macro_rate": float(np.mean(
                [v[0] / v[1] for v in st_rate.values()])),
            "equal_matrix_macro_rate": float(np.mean(
                [v[0] / v[1] for v in mat_rate.values()])),
            "half_state_agreement_pearson_descriptive": agree,
        }
        n = comp.pop("n")
        results["components"][cand] = {
            "n_futures": n,
            "frac_break_within": comp["break_within"] / n,
            "frac_run8_after_break": comp["run8_after"] / n,
            "frac_coh_first_window": comp["coh_first"] / n,
            "frac_dis_first_window": comp["dis_first"] / n,
            "frac_full_endpoint": comp["positive"] / n,
            "median_minpair_first": float(np.median(comp["minpair"]))
            if comp["minpair"] else None,
            "median_coherence_margin": float(np.median(
                np.array(comp["minpair"]) - COHERE_T))
            if comp["minpair"] else None,
            "median_distinctness_margin": float(np.median(
                DISTINCT_T - np.array(comp["maxanchor"])))
            if comp["maxanchor"] else None,
            "median_run_len_positives": float(np.median(
                comp["run_len"])) if comp["run_len"] else None,
            "max_run_len_positives": int(max(comp["run_len"]))
            if comp["run_len"] else None,
        }
    return results


def adjudicate(results):
    cells = results["cells"]
    gate = all(c["rate"] > 0 and c["ci95"][0] > 0
               for c in cells.values())
    compatible = all(
        cells[f"{cand}/{half}"]["ci95"][0] <= EXTERNAL[(cand, half)]
        <= cells[f"{cand}/{half}"]["ci95"][1]
        for cand in cohort.CANDIDATES for half in ("A", "B"))
    if not gate:
        return "C: partial or failed replication"
    if compatible:
        return ("A: phenomenon and rate numerically compatible "
                "with the external result")
    return ("B: phenomenon replicated, rate contract-sensitive "
            "(legitimate; contracts differ between clean rooms)")


# ---- driver ----------------------------------------------------------

def run(smoke=False):
    global _ENT
    os.makedirs(OUT, exist_ok=True)
    if smoke:
        _ENT = cohort.domain_entropy("strict8-smoke", TAG)
        n_mat, n_branch = 2, 8
    else:
        n_mat, n_branch = N_MAT, N_BRANCH

    t0 = time.time()
    jobs = [(m, c, "full", n_mat, n_branch)
            for c in cohort.CANDIDATES for m in range(n_mat)]
    with Pool(12) as pool:
        units = pool.map(unit, jobs)
    t_pass1 = time.time() - t0
    h1 = campaign_hash(units)

    t0 = time.time()
    jobs = [(m, c, "hash", n_mat, n_branch)
            for c in cohort.CANDIDATES for m in range(n_mat)]
    with Pool(12) as pool:
        replay = pool.map(unit, jobs)
    t_pass2 = time.time() - t0
    h2 = campaign_hash(replay)
    replay_equal = (h1 == h2)

    if smoke:
        report = {"io": "ok", "replay_equal": bool(replay_equal),
                  "n_units": len(units),
                  "n_records": int(sum(len(u["futures"])
                                       for u in units))}
        with open(os.path.join(OUT, "smoke_check.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("SMOKE: io ok; replay_equal =", replay_equal,
              "; records =", report["n_records"])
        print("(no event counts or rates inspected, per registration)")
        return

    with open(os.path.join(OUT, "strict8_units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=4)

    results = analyze(units, n_branch)
    realized_states = sum(len(u["states"]) for u in units)
    realized_futures = sum(len(u["futures"]) for u in units)
    results["cohort"] = {
        "seed_entropy_string":
            f"replication-strict8-domain-{TAG}",
        "domain": DOMAIN, "n_matrices": N_MAT,
        "landmarks": LANDMARKS, "branches_per_state": N_BRANCH,
        "half_A": [0, 63], "half_B": [64, 127],
        "nominal_states": 2 * N_MAT * len(LANDMARKS),
        "realized_states": realized_states,
        "nominal_futures": 2 * N_MAT * len(LANDMARKS) * N_BRANCH,
        "realized_futures": realized_futures,
        "main_path_extinction_disclosure":
            "landmarks beyond a main trajectory's realized fissions "
            "are dropped, never retried (frozen contract)",
    }
    results["replay"] = {"hash_pass1": h1, "hash_pass2": h2,
                         "exact_replay_equal": bool(replay_equal)}
    results["runtime_s"] = {"pass1": round(t_pass1),
                            "replay": round(t_pass2)}
    results["gate"] = {
        "all_cells_positive": all(c["rate"] > 0 for c in
                                  results["cells"].values()),
        "all_lower_bounds_positive": all(c["ci95"][0] > 0 for c in
                                         results["cells"].values()),
        "exact_replay": bool(replay_equal),
    }
    results["external_benchmark"] = {f"{c}/{h}": EXTERNAL[(c, h)]
                                     for c in cohort.CANDIDATES
                                     for h in ("A", "B")}
    results["conclusion"] = adjudicate(results) if replay_equal \
        else "C: replay gate failed"

    with open(os.path.join(OUT, "strict8_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"pass1 {t_pass1:.0f}s, replay {t_pass2:.0f}s, "
          f"replay_equal={replay_equal}")
    print(f"realized states {realized_states}/2000, futures "
          f"{realized_futures}/256000")
    for name, c in results["cells"].items():
        print(f"cell {name}: rate {c['rate']:.5f} "
              f"CI [{c['ci95'][0]:.5f},{c['ci95'][1]:.5f}] "
              f"events {c['events']}/{c['futures']} | matrices w/ event "
              f"{c['matrices_with_event']}/{c['matrices_total']} | "
              f"states w/ event {c['states_with_event']}"
              f"/{c['states_total']} | med/max per-matrix "
              f"{c['median_events_per_matrix']:.1f}/"
              f"{c['max_events_per_matrix']} | timing med "
              f"break {c['timing_median']['first_break']} start "
              f"{c['timing_median']['run_start']} cert "
              f"{c['timing_median']['cert']}")
    for cand in cohort.CANDIDATES:
        d = results["descriptive"][cand]
        k = results["components"][cand]
        print(f"{cand} descriptive: state-macro "
              f"{d['equal_state_macro_rate']:.5f} matrix-macro "
              f"{d['equal_matrix_macro_rate']:.5f} half-agreement "
              f"{d['half_state_agreement_pearson_descriptive']}")
        print(f"{cand} components: break {k['frac_break_within']:.3f} "
              f"run8 {k['frac_run8_after_break']:.4f} "
              f"coh(first) {k['frac_coh_first_window']:.4f} "
              f"dis(first) {k['frac_dis_first_window']:.4f} "
              f"full {k['frac_full_endpoint']:.5f}")
    print("GATE:", results["gate"])
    print("CONCLUSION:", results["conclusion"])
    print("written:", os.path.join(OUT, "strict8_results.json"))


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
