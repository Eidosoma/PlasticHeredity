"""Re-run headline stats (C2 spikes, C3 correlation, C4 Ljung-Box, consistency
band, C5 prediction) with the authors-pipeline emergence-capacity Phi_r."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")
from recovery_c5 import resample

ROOT = Path(__file__).parent.parent
SPLIT = 0.25


def load(sub):
    rows = []
    for f in sorted((ROOT / "results" / sub).glob("run_*.npz")):
        d = np.load(f)
        v = np.load(ROOT / "results" / "phi_authors" / sub / f.name)
        rows.append({"phi": v["phi"].astype(float), "sr": d["sr"][1:],
                     "counts": d["counts"].astype(float)})
    return rows


def headline(rows):
    out = {}
    spike_runs = pos = sig = hi = lb = n = 0
    mw_ps, autocorrs = [], []
    for r in rows:
        phi, sr = r["phi"], r["sr"]
        m = min(len(phi), len(sr))
        phi, sr = phi[:m], sr[:m]
        ok = np.isfinite(phi)
        phi, sr = phi[ok], sr[ok]
        if len(phi) < 50:
            continue
        if (phi > phi.mean() + 3 * phi.std()).any():
            spike_runs += 1
        autocorrs.append(stats.pearsonr(phi[:-1], phi[1:])[0])
        if sr.min() != sr.max():
            n += 1
            rho, p = stats.spearmanr(phi, sr.astype(float))
            pos += rho > 0
            sig += (rho > 0) and (p < 0.05)
            mw = stats.mannwhitneyu(phi[sr], phi[~sr], alternative="greater")
            mw_ps.append(mw.pvalue)
            hi += (np.median(phi[sr]) > np.median(phi[~sr])) and (mw.pvalue < 0.001)
        if acorr_ljungbox(phi, lags=[20], return_df=True)["lb_pvalue"].iloc[0] < 0.05:
            lb += 1
    fisher = stats.chi2.sf(-2 * np.sum(np.log(np.clip(mw_ps, 1e-300, 1))),
                           2 * len(mw_ps))
    out["C2_spike_runs"] = spike_runs
    out["C3"] = {"positive": pos, "positive_sig": sig, "higher_p001": hi,
                 "n": n, "fisher_p": float(fisher)}
    out["C4_ljungbox_reject"] = lb
    out["consistency_phi_autocorr"] = (f"{np.nanmean(autocorrs):.3f}"
                                       f"±{np.nanstd(autocorrs):.3f}")
    return out


def c5(rows, n_in=64, n_out=16, hidden=(32,), n_reps=10):
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for r in rows:
        phi, sr, counts = r["phi"], r["sr"], r["counts"]
        cut = int(len(phi) * SPLIT)
        rel = counts / counts.sum(axis=1, keepdims=True)
        feats["phi"].append(resample(phi[:cut], n_in).ravel())
        feats["dcomp"].append(
            resample(np.linalg.norm(np.diff(rel, axis=0), axis=1)[:cut], n_in).ravel())
        feats["raw"].append(resample(rel[:cut], n_in).ravel())
        feats["flux"].append(
            resample(np.abs(np.diff(counts, axis=0)).sum(axis=1)[:cut], n_in).ravel())
        targets.append(resample(sr[int(len(sr) * SPLIT):].astype(float),
                                n_out) > 0.5)
    feats = {k: np.array(v) for k, v in feats.items()}
    targets = np.array(targets)
    accs = {k: [] for k in list(feats) + ["dummy"]}
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        order = rng.permutation(len(targets))
        nt = max(1, int(0.2 * len(targets)))
        test, train = order[:nt], order[nt:]
        y_tr = targets[train].reshape(len(train), -1)
        y_te = targets[test].reshape(nt, -1)
        maj = y_tr.mean(axis=0) > 0.5
        accs["dummy"].append(float((y_te == maj).mean()))
        for k, X in feats.items():
            mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
            clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=600,
                                random_state=rep)
            clf.fit((X[train] - mu) / sd, y_tr)
            accs[k].append(float((clf.predict((X[test] - mu) / sd) == y_te).mean()))
    row = {k: round(float(np.mean(v)), 4) for k, v in accs.items()}
    for k in ("dcomp", "raw", "flux", "dummy"):
        row[f"p>{k}"] = round(float(stats.mannwhitneyu(
            accs["phi"], accs[k], alternative="greater").pvalue), 4)
    return row


def main():
    out = {}
    for sub in ("runs_coarse", "runs"):
        rows = load(sub)
        out[sub] = {"headline": headline(rows), "C5_in64": c5(rows)}
        print(sub, json.dumps(out[sub], indent=2, default=float), flush=True)
    (ROOT / "results" / "authors_pipeline_stats.json").write_text(
        json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
