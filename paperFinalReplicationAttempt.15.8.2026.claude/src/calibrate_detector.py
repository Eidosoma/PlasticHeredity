"""Reverse-engineer the paper's self-replicator detector by calibrating against
Table 1's control column: probability 0.88±0.03, persistence 716±198 (= total
SR steps), consistency 0.38±0.06 (lag-1 Pearson autocorr of the binary SR
trajectory), time-to-first ≈ 36±26 steps. Sweep similarity metric × reference
composition × threshold rule on the saved coarse-universe runs; rank configs by
distance to targets. The ±3% probability spread across runs strongly suggests a
per-run adaptive threshold, so quantile/mean-based rules are included.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")
from composomes import nondrift_mask, compotypes, _unit

ROOT = Path(__file__).parent.parent
TARGETS = {"probability": (0.88, 0.03), "persistence": (716, 198),
           "consistency": (0.38, 0.06), "time_to_first": (36, 26)}


def references(counts, fission_steps):
    pre = counts[fission_steps].astype(float)
    refs = {}
    mask = nondrift_mask(pre, 0.9)
    if mask.sum() >= 2:
        cents, labels = compotypes(pre[mask], seed=0)
        refs["dominant_compotype"] = cents[np.bincount(labels).argmax()]
    else:
        refs["dominant_compotype"] = pre[0]
    u = _unit(pre)
    sim = u @ u.T
    refs["modal_prefission"] = pre[int(np.argmax(sim.sum(axis=1)))]
    refs["mean_prefission"] = pre.mean(axis=0)
    return refs


def similarities(counts, ref):
    c = counts.astype(float)
    rel = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
    rref = ref / max(ref.sum(), 1e-12)
    cos = _unit(c) @ _unit(ref)
    euc_rel = -np.linalg.norm(rel - rref, axis=1)        # higher = more similar
    return {"cosine": cos, "euclid_rel": euc_rel}


def label(sim, rule):
    kind, val = rule
    if kind == "abs":                     # absolute threshold (cosine only)
        return sim >= val
    if kind == "quantile":                # per-run: top (1-q) fraction
        return sim >= np.quantile(sim, val)
    if kind == "mean_k_std":              # per-run: mean + k*std
        return sim >= sim.mean() + val * sim.std()
    raise ValueError(kind)


def metrics(sr):
    n = len(sr)
    x = sr.astype(float)
    if x[:-1].std() < 1e-12 or x[1:].std() < 1e-12:
        cons = np.nan
    else:
        cons = stats.pearsonr(x[:-1], x[1:])[0]
    first = int(np.argmax(sr)) if sr.any() else n
    return dict(probability=float(sr.mean()), persistence=int(sr.sum()),
                consistency=cons, time_to_first=first)


def score(mrows):
    s = 0.0
    for key, (mu, sd) in TARGETS.items():
        vals = np.array([m[key] for m in mrows], float)
        s += abs(np.nanmean(vals) - mu) / mu          # match the mean
        s += abs(np.nanstd(vals) - sd) / max(sd, 1)   # and the spread
    return s


def main():
    runs = []
    for f in sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz")):
        d = np.load(f)
        runs.append((d["counts"].astype(float), d["fission_steps"]))
    print(f"{len(runs)} coarse runs")

    rules = ([("abs", v) for v in (0.5, 0.6, 0.7, 0.8, 0.9)]
             + [("quantile", q) for q in (0.05, 0.12, 0.2, 0.3)]
             + [("mean_k_std", k) for k in (-1.5, -1.0, -0.5, 0.0)])
    metrics_cache = {}
    for (counts, fs) in runs:
        refs = references(counts, fs)
        for rname, ref in refs.items():
            for sname, sim in similarities(counts, ref).items():
                for rule in rules:
                    if sname != "cosine" and rule[0] == "abs":
                        continue
                    key = (rname, sname, rule)
                    metrics_cache.setdefault(key, []).append(
                        metrics(label(sim, rule)))

    rows = []
    for (rname, sname, rule), mrows in metrics_cache.items():
        agg = {k: f"{np.nanmean([m[k] for m in mrows]):.2f}"
                  f"±{np.nanstd([m[k] for m in mrows]):.2f}"
               for k in TARGETS}
        rows.append((score(mrows), rname, sname, str(rule), agg))
    rows.sort(key=lambda r: r[0])
    for r in rows[:12]:
        print(f"score={r[0]:.2f}  ref={r[1]:<19s} sim={r[2]:<10s} rule={r[3]:<22s} {r[4]}")
    (ROOT / "results" / "detector_calibration.json").write_text(
        json.dumps([{"score": r[0], "ref": r[1], "sim": r[2], "rule": r[3],
                     "metrics": r[4]} for r in rows[:30]], indent=2))


if __name__ == "__main__":
    main()
