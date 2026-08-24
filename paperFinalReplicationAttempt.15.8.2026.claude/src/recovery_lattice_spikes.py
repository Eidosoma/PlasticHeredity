"""C5 rescue routes 1 & 2 (2026-08-17).

Route 1 — FULL-LATTICE CLOSURE TEST: feed the MLP all 16 local PhiID atom
trajectories of the code-faithful macro system (phi_r_code pipeline) instead
of any scalar. Every scalar "Phi" definable on this lattice (printed-Psi-on-
macro, code Phi_R, emergence capacity, any atom reweighting) is a fixed
linear projection of these features, so if the full stack cannot beat the
causal baselines, no scalar in the ΦID-on-macro-halves family can.
Rows: atoms16 (whole-run moments, the authors' leaky convention) and
atoms16_leakfree (ALL estimation restricted to the input window).

Route 2 — SPIKE STATISTICS: low-dimensional episode features from the
first-25% Phi trajectory (spike counts at 2/3 sigma, max excursion, first-
spike time, mean inter-spike gap, above-2sigma fraction, window mean/sd).
Rows: spikes_phir (leak-free code Phi_R), spikes_psi (leak-free printed Psi),
spikes_phir_wholefit (trajectory + threshold from whole-run moments, for
diagnosis of how much the paper-style version owes to leakage).

Same harness as recovery_c5/recovery_phir: success = beats dcomp, raw, flux
AND dummy (MW one-sided p<0.05); both label rules; 10 rep splits.
Writes results/recovery_lattice_spikes.json.
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from recovery_c5 import resample, quantile_labels, SPLIT, CONFIGS
from phi_r_code import macro_halves, local_phi_id, ORDER, phi_r_code_local
from phi import phi_r_local

ROOT = Path(__file__).parent.parent
ROWS = ["atoms16", "atoms16_leakfree", "spikes_phir", "spikes_psi",
        "spikes_phir_wholefit"]


def atom_stack(counts, fit):
    """(T-1, 16) local atom trajectories; zeros if degenerate."""
    edge, _ = macro_halves(counts, fit)
    T = (np.asarray(counts).shape[0] if fit == "full"
         else min(int(fit), np.asarray(counts).shape[0]))
    if edge is None:
        return np.zeros((max(T - 1, 0), 16))
    pi = local_phi_id(edge)
    return np.stack([np.asarray(pi[a], float) for a in ORDER], axis=1)


def spike_features(x):
    """9 episode statistics of one Phi window (z relative to the window)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.zeros(9)
    mu, sd = x.mean(), x.std() + 1e-12
    z = (x - mu) / sd
    n = len(z)

    def first(th):
        idx = np.flatnonzero(z > th)
        return idx[0] / n if len(idx) else 1.0

    def mean_gap(th):
        idx = np.flatnonzero(z > th)
        return float(np.diff(idx).mean()) / n if len(idx) > 1 else 1.0

    return np.array([(z > 3).sum(), (z > 2).sum(), z.max(), first(3),
                     first(2), mean_gap(2), (z > 2).mean(), mu, x.std()])


def spike_features_wholefit(traj_full, cut):
    """Paper-style: z against WHOLE-run moments, stats on the first-25%."""
    x = np.asarray(traj_full, float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.zeros(9)
    mu, sd = x[ok].mean(), x[ok].std() + 1e-12
    z = (x[:cut] - mu) / sd
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return np.zeros(9)
    n = len(z)

    def first(th):
        idx = np.flatnonzero(z > th)
        return idx[0] / n if len(idx) else 1.0

    def mean_gap(th):
        idx = np.flatnonzero(z > th)
        return float(np.diff(idx).mean()) / n if len(idx) > 1 else 1.0

    return np.array([(z > 3).sum(), (z > 2).sum(), z.max(), first(3),
                     first(2), mean_gap(2), (z > 2).mean(), mu,
                     x[:cut][np.isfinite(x[:cut])].std()])


def _one_run(f):
    d = np.load(f)
    counts = d["counts"].astype(float)
    seed = int(Path(f).stem.split("_")[1])
    cut = int((counts.shape[0] - 1) * SPLIT)
    phir_lf = phi_r_code_local(counts, fit=cut + 1)
    psi_lf, _ = phi_r_local(counts[:cut + 1], mib_seed=seed)
    return {
        "seed": seed, "counts": counts,
        "fission_steps": d["fission_steps"], "sr": d["sr"],
        "atoms16": atom_stack(counts, "full"),
        "atoms16_leakfree": atom_stack(counts, cut + 1),
        "spikes_phir": spike_features(phir_lf),
        "spikes_psi": spike_features(np.asarray(psi_lf, float)),
        "spikes_phir_wholefit": spike_features_wholefit(
            phi_r_code_local(counts, fit="full"), cut),
    }


def load_runs():
    cache = ROOT / "results" / "lattice_features.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            print("features loaded from cache", flush=True)
            return pickle.load(fh)
    files = [str(f) for f in
             sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz"))]
    with Pool(12) as pool:
        runs = pool.map(_one_run, files)
    with open(cache, "wb") as fh:
        pickle.dump(runs, fh, protocol=4)
    print(f"features computed for {len(runs)} runs (cached)", flush=True)
    return runs


def experiment(runs, variant, sr_key, cfg, n_reps=10):
    n_in, n_out, hidden = cfg["n_in"], cfg["n_out"], cfg["hidden"]
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for r in runs:
        counts, sr = r["counts"], r[sr_key]
        full_len = counts.shape[0] - 1
        cut = int(full_len * SPLIT)
        v = r[variant]
        if variant.startswith("atoms16"):
            seg = v if variant.endswith("leakfree") else v[:cut]
            phi_feat = resample(np.nan_to_num(seg), n_in).ravel()
        else:
            phi_feat = np.nan_to_num(v)          # 9-dim, no resampling
        rel = counts / counts.sum(axis=1, keepdims=True)
        feats["phi"].append(phi_feat)
        feats["dcomp"].append(resample(
            np.linalg.norm(np.diff(rel, axis=0), axis=1)[:cut], n_in).ravel())
        feats["raw"].append(resample(rel[:cut], n_in).ravel())
        feats["flux"].append(resample(
            np.abs(np.diff(counts, axis=0)).sum(axis=1)[:cut], n_in).ravel())
        srr = sr[1:]
        targets.append(resample(srr[int(len(srr) * SPLIT):].astype(float),
                                n_out) > 0.5)
    feats = {k: np.array(v) for k, v in feats.items()}
    targets = np.array(targets)
    accs = {k: [] for k in list(feats) + ["dummy"]}
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        order = rng.permutation(len(targets))
        n_test = max(1, int(0.2 * len(targets)))
        test, train = order[:n_test], order[n_test:]
        y_tr = targets[train].reshape(len(train), -1)
        y_te = targets[test].reshape(len(test), -1)
        maj = y_tr.mean(axis=0) > 0.5
        accs["dummy"].append(float((y_te == maj).mean()))
        for k, X in feats.items():
            mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
            clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=600,
                                random_state=rep)
            clf.fit((X[train] - mu) / sd, y_tr)
            accs[k].append(
                float((clf.predict((X[test] - mu) / sd) == y_te).mean()))
    row = {k: round(float(np.mean(v)), 4) for k, v in accs.items()}
    beats = []
    for k in ("dcomp", "raw", "flux", "dummy"):
        p = float(stats.mannwhitneyu(accs["phi"], accs[k],
                                     alternative="greater").pvalue)
        row[f"p>{k}"] = round(p, 4)
        beats.append(p < 0.05)
    row["beats_all"] = all(beats)
    return row


_RUNS = None


def _init(runs):
    global _RUNS
    _RUNS = runs


def _one_experiment(job):
    label_rule, variant, cfg = job
    name = f"{label_rule}/{variant}/in{cfg['n_in']}"
    return name, experiment(_RUNS, variant, f"sr_{label_rule}", cfg)


def main():
    runs = load_runs()
    for r in runs:
        r["sr_classic09"] = r["sr"]
        r["sr_quantile12"] = quantile_labels(
            r["counts"], r["fission_steps"], seed=r["seed"])
    jobs = [(lr, v, cfg) for lr in ("classic09", "quantile12")
            for v in ROWS for cfg in CONFIGS]
    results = {}
    with Pool(10, initializer=_init, initargs=(runs,)) as pool:
        for name, row in pool.imap(_one_experiment, jobs):
            results[name] = row
            flag = "  *** BEATS ALL ***" if row["beats_all"] else ""
            print(name, json.dumps(row), flag, flush=True)
    (ROOT / "results" / "recovery_lattice_spikes.json").write_text(
        json.dumps(results, indent=2))
    print("written results/recovery_lattice_spikes.json")


if __name__ == "__main__":
    main()
