"""C5 robustness grid: the paper's MLP setup is underspecified, so test several
defensible configurations and report every one (no cherry-picking).
Axes: input bins, per-run z-scoring of Phi_r, output bins, hidden size.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier

np.seterr(all="ignore")
ROOT = Path(__file__).parent.parent
SPLIT = 0.25
RUNS_SUB = "runs_coarse" if "coarse" in sys.argv[1:] else "runs"


def resample_bins(x, n_bins):
    """Resample to n_bins points: bin means when there are enough samples,
    linear interpolation otherwise (short runs)."""
    x = np.asarray(x, float)
    if len(x) >= 2 * n_bins:
        edges = np.linspace(0, len(x), n_bins + 1).astype(int)
        return np.stack([x[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    grid = np.linspace(0, len(x) - 1, n_bins)
    if x.ndim == 1:
        return np.interp(grid, np.arange(len(x)), x)
    return np.stack([np.interp(grid, np.arange(len(x)), x[:, j])
                     for j in range(x.shape[1])], axis=1)


def load(n_in, n_out, zscore_phi):
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for f in sorted((ROOT / "results" / RUNS_SUB).glob("run_*.npz")):
        d = np.load(f)
        counts = d["counts"].astype(float)
        phi = d["phi"].astype(float)
        if zscore_phi:
            phi = (phi - phi.mean()) / (phi.std() + 1e-12)
        sr = d["sr"][1:]
        cut = int(len(phi) * SPLIT)
        rel = counts / counts.sum(axis=1, keepdims=True)
        dcomp = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        flux = np.abs(np.diff(counts, axis=0)).sum(axis=1)
        feats["phi"].append(resample_bins(phi[:cut], n_in).ravel())
        feats["dcomp"].append(resample_bins(dcomp[:cut], n_in).ravel())
        feats["raw"].append(resample_bins(rel[:cut], n_in).ravel())
        feats["flux"].append(resample_bins(flux[:cut], n_in).ravel())
        targets.append(resample_bins(sr[cut:].astype(float), n_out) > 0.5)
    return {k: np.array(v) for k, v in feats.items()}, np.array(targets)


def run(feats, targets, hidden, n_reps=10):
    accs = {k: [] for k in list(feats) + ["dummy"]}
    n_runs = len(targets)
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
            clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=600,
                                random_state=rep)
            clf.fit((X[train] - mu) / sd, y_tr)
            accs[k].append(float((clf.predict((X[test] - mu) / sd) == y_te).mean()))
    return accs


def main():
    results = {}
    configs = [
        dict(n_in=32, n_out=64, zscore_phi=False, hidden=(64,)),
        dict(n_in=32, n_out=64, zscore_phi=True, hidden=(64,)),
        dict(n_in=128, n_out=64, zscore_phi=True, hidden=(64,)),
        dict(n_in=250, n_out=250, zscore_phi=True, hidden=(128, 64)),
        dict(n_in=64, n_out=16, zscore_phi=True, hidden=(32,)),
    ]
    for cfg in configs:
        name = f"in{cfg['n_in']}_out{cfg['n_out']}_z{int(cfg['zscore_phi'])}_h{cfg['hidden']}"
        feats, targets = load(cfg["n_in"], cfg["n_out"], cfg["zscore_phi"])
        accs = run(feats, targets, cfg["hidden"])
        row = {k: round(float(np.mean(v)), 4) for k, v in accs.items()}
        for k in ("dcomp", "raw", "flux", "dummy"):
            row[f"p_phi>{k}"] = round(float(stats.mannwhitneyu(
                accs["phi"], accs[k], alternative="greater").pvalue), 4)
        results[name] = row
        print(name, json.dumps(row), flush=True)
    tag = "_coarse" if RUNS_SUB.endswith("coarse") else ""
    (ROOT / "results" / f"ml_grid{tag}.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
