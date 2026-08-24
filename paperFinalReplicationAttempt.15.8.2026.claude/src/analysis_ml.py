"""C5: predict the self-replication trajectory (last 75% of each run) from the
first 25% of the Phi_r trajectory, vs. baselines (delta-composition, raw
composition, fluxes, dummy). MLP, 80/20 run split, 10 repetitions.

Feature construction (paper does not specify): every input channel is
resampled to N_BINS time bins over the first 25% of steps; the target is the
SR label majority per bin over the last 75% (N_BINS bins). Binary accuracy
averaged over bins and test runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier

np.seterr(all="ignore")
ROOT = Path(__file__).parent.parent
N_BINS_IN = 32
N_BINS_OUT = 64
SPLIT = 0.25


def resample_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean over n_bins equal time bins; x is (steps,) or (steps, ch)."""
    edges = np.linspace(0, len(x), n_bins + 1).astype(int)
    return np.stack([x[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


def load_features():
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for f in sorted((ROOT / "results" / "runs").glob("run_*.npz")):
        d = np.load(f)
        counts = d["counts"].astype(float)
        phi = d["phi"].astype(float)
        sr = d["sr"][1:]
        n = len(phi)
        cut = int(n * SPLIT)
        rel = counts / counts.sum(axis=1, keepdims=True)
        dcomp = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        flux = (d["counts"][1:] - d["counts"][:-1]).astype(float)
        feats["phi"].append(resample_bins(phi[:cut], N_BINS_IN).ravel())
        feats["dcomp"].append(resample_bins(dcomp[:cut], N_BINS_IN).ravel())
        feats["raw"].append(resample_bins(rel[:cut], N_BINS_IN).ravel())
        feats["flux"].append(resample_bins(flux[:cut], N_BINS_IN).ravel())
        targets.append(resample_bins(sr[cut:].astype(float), N_BINS_OUT) > 0.5)
    feats = {k: np.array(v) for k, v in feats.items()}
    return feats, np.array(targets)


def run_experiment(feats, targets, n_reps=10):
    n_runs = len(targets)
    accs = {k: [] for k in list(feats) + ["dummy"]}
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        order = rng.permutation(n_runs)
        n_test = max(1, int(0.2 * n_runs))
        test, train = order[:n_test], order[n_test:]
        y_tr = targets[train].reshape(len(train), -1)
        y_te = targets[test].reshape(len(test), -1)
        maj = y_tr.mean(axis=0) > 0.5
        accs["dummy"].append(float((y_te == maj).mean()))
        for k, X in feats.items():
            mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                random_state=rep)
            clf.fit((X[train] - mu) / sd, y_tr)
            pred = clf.predict((X[test] - mu) / sd)
            accs[k].append(float((pred == y_te).mean()))
    return accs


def main():
    feats, targets = load_features()
    print(f"{len(targets)} runs; feature dims:",
          {k: v.shape[1] for k, v in feats.items()})
    accs = run_experiment(feats, targets)
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for k, v in accs.items()}
    for k in ("dcomp", "raw", "flux", "dummy"):
        u = stats.mannwhitneyu(accs["phi"], accs[k], alternative="greater")
        out[k]["p_phi_greater"] = float(u.pvalue)
    (ROOT / "results" / "ml_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
