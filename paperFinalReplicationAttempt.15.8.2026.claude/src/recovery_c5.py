"""C5 recovery: MLP prediction with every Phi_r estimator variant x SR label
rule, on the coarse universe. Success criterion = the paper's: Phi_r model
beats delta-composition, raw, flux AND dummy (MW p<0.05, one-sided).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")
from composomes import label_self_replication
from phi import clr  # noqa: F401  (parity with variant pipeline)

ROOT = Path(__file__).parent.parent
SPLIT = 0.25
VARIANTS = ["local", "win100", "scalar2", "phyid_mmi", "tau2", "tau4"]
CONFIGS = [dict(n_in=64, n_out=16, hidden=(32,)),
           dict(n_in=32, n_out=64, hidden=(64,))]


def resample(x, n_bins):
    x = np.asarray(x, float)
    if len(x) >= 2 * n_bins:
        edges = np.linspace(0, len(x), n_bins + 1).astype(int)
        return np.stack([x[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    grid = np.linspace(0, len(x) - 1, n_bins)
    if x.ndim == 1:
        return np.interp(grid, np.arange(len(x)), x)
    return np.stack([np.interp(grid, np.arange(len(x)), x[:, j])
                     for j in range(x.shape[1])], axis=1)


def quantile_labels(counts, fission_steps, q=0.12, seed=0):
    from composomes import nondrift_mask, compotypes, _unit, cosine_to
    pre = counts[fission_steps].astype(float)
    mask = nondrift_mask(pre, 0.9)
    if mask.sum() >= 2:
        cents, labels = compotypes(pre[mask], seed=seed)
        dom = cents[np.bincount(labels).argmax()]
    else:
        dom = pre[0]
    sims = cosine_to(counts.astype(float), dom)
    return sims >= np.quantile(sims, q)


def load(label_rule):
    files = sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz"))
    vdir = ROOT / "results" / "phi_variants"
    data = []
    for f in files:
        d = np.load(f)
        v = np.load(vdir / f.name)
        counts = d["counts"].astype(float)
        seed = int(f.stem.split("_")[1])
        if label_rule == "classic09":
            sr = d["sr"]
        else:
            sr = quantile_labels(d["counts"], d["fission_steps"], seed=seed)
        rel = counts / counts.sum(axis=1, keepdims=True)
        base = {
            "dcomp": np.linalg.norm(np.diff(rel, axis=0), axis=1),
            "raw": rel,
            "flux": np.abs(np.diff(counts, axis=0)).sum(axis=1),
        }
        phis = {k: np.asarray(v[k], float) for k in VARIANTS}
        data.append((phis, base, sr))
    return data


def experiment(data, variant, cfg, n_reps=10):
    n_in, n_out, hidden = cfg["n_in"], cfg["n_out"], cfg["hidden"]
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for phis, base, sr in data:
        phi = phis[variant]
        cut = int(len(phi) * SPLIT)
        feats["phi"].append(resample(phi[:cut], n_in).ravel())
        feats["dcomp"].append(resample(base["dcomp"][:cut], n_in).ravel())
        feats["raw"].append(resample(base["raw"][:cut], n_in).ravel())
        feats["flux"].append(resample(base["flux"][:cut], n_in).ravel())
        srr = sr[1:]
        targets.append(resample(srr[int(len(srr) * SPLIT):].astype(float),
                                n_out) > 0.5)
    feats = {k: np.array(vv) for k, vv in feats.items()}
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
            accs[k].append(float((clf.predict((X[test] - mu) / sd) == y_te).mean()))
    row = {k: round(float(np.mean(vv)), 4) for k, vv in accs.items()}
    beats = []
    for k in ("dcomp", "raw", "flux", "dummy"):
        p = float(stats.mannwhitneyu(accs["phi"], accs[k],
                                     alternative="greater").pvalue)
        row[f"p>{k}"] = round(p, 4)
        beats.append(p < 0.05)
    row["beats_all"] = all(beats)
    return row


def main():
    results = {}
    for label_rule in ("classic09", "quantile12"):
        data = load(label_rule)
        for variant in VARIANTS:
            for cfg in CONFIGS:
                name = f"{label_rule}/{variant}/in{cfg['n_in']}"
                row = experiment(data, variant, cfg)
                results[name] = row
                flag = "  *** BEATS ALL ***" if row["beats_all"] else ""
                print(name, json.dumps(row), flag, flush=True)
    (ROOT / "results" / "recovery_c5.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
