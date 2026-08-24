"""C5/C2-C4 under the CODE-FAITHFUL Phi_R (phi_r_code.py) — the quantity
the authors' public implementation computes, never previously tested in
this replication (all prior variants were Psi-flavored: redundancy
subtracted or excluded).

Outputs results/recovery_phir.json with:
  sign_regime   — run-mean Phi_R vs run-mean printed-Psi distributions
  phyid_xcheck  — Pearson r between our port and phyid's nine-atom sum
  headline      — C2 spikes / C3 correlation / C4 Ljung-Box / consistency
                  battery on the Phi_R trajectory (classic SR labels)
  c5            — MLP prediction harness (same as recovery_c5.py):
                  variants psi_local / phir / phir_leakfree
                  x label rules classic09 / quantile12 x in64 / in32.
phir uses whole-run moments (the authors' convention — leaks future
info); phir_leakfree restricts ALL estimation to the input window.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from recovery_c5 import resample, quantile_labels, SPLIT, CONFIGS
from reanalyze_authors import headline
from phi_r_code import phi_r_code_local, phi_r_phyid_local

ROOT = Path(__file__).parent.parent
PHI_ROWS = ["psi_local", "phir", "phir_leakfree"]


def load_runs():
    files = sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz"))
    runs = []
    for i, f in enumerate(files):
        d = np.load(f)
        counts = d["counts"].astype(float)
        cut = int((counts.shape[0] - 1) * SPLIT)
        runs.append({
            "seed": int(f.stem.split("_")[1]),
            "counts": counts,
            "fission_steps": d["fission_steps"],
            "sr": d["sr"],
            "psi_local": d["phi"].astype(float),
            "phir": phi_r_code_local(counts, fit="full"),
            "phir_leakfree": phi_r_code_local(counts, fit=cut + 1),
        })
        if (i + 1) % 20 == 0:
            print(f"phi_R computed {i + 1}/{len(files)}", flush=True)
    return runs


def experiment(runs, variant, sr_key, cfg, n_reps=10):
    """recovery_c5.experiment, with support for window-only phi arrays."""
    n_in, n_out, hidden = cfg["n_in"], cfg["n_out"], cfg["hidden"]
    feats = {"phi": [], "dcomp": [], "raw": [], "flux": []}
    targets = []
    for r in runs:
        counts, sr = r["counts"], r[sr_key]
        full_len = counts.shape[0] - 1
        cut = int(full_len * SPLIT)
        phi = r[variant]
        seg = phi if variant == "phir_leakfree" else phi[:cut]
        seg = np.nan_to_num(np.asarray(seg, float))
        rel = counts / counts.sum(axis=1, keepdims=True)
        feats["phi"].append(resample(seg, n_in).ravel())
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


def main():
    runs = load_runs()

    # ---- sign regime -------------------------------------------------
    mean_phir = np.array([np.nanmean(r["phir"]) for r in runs])
    mean_psi = np.array([np.nanmean(r["psi_local"]) for r in runs])
    sign = {
        "phir_run_mean": f"{np.nanmean(mean_phir):+.4f}±{np.nanstd(mean_phir):.4f}",
        "phir_runs_positive": int((mean_phir > 0).sum()),
        "psi_run_mean": f"{np.nanmean(mean_psi):+.4f}±{np.nanstd(mean_psi):.4f}",
        "psi_runs_positive": int((mean_psi > 0).sum()),
        "n_runs": len(runs),
    }
    ok = np.isfinite(mean_phir) & np.isfinite(mean_psi)
    sign["runmean_pearson_phir_vs_psi"] = round(
        float(np.corrcoef(mean_phir[ok], mean_psi[ok])[0, 1]), 3)
    print("SIGN REGIME:", json.dumps(sign, indent=2), flush=True)

    # ---- phyid cross-check on 5 runs ---------------------------------
    xs = []
    for r in runs[:5]:
        a, b = r["phir"], phi_r_phyid_local(r["counts"])
        m = min(len(a), len(b))
        good = np.isfinite(a[:m]) & np.isfinite(b[:m])
        if good.sum() > 50:
            xs.append(float(np.corrcoef(a[:m][good], b[:m][good])[0, 1]))
    xcheck = {"pearson_per_run": [round(v, 4) for v in xs]}
    print("PHYID CROSS-CHECK:", xcheck, flush=True)

    # ---- C2-C4 headline battery on Phi_R -----------------------------
    hl_rows = [{"phi": np.asarray(r["phir"], float), "sr": r["sr"][1:],
                "counts": r["counts"]} for r in runs]
    hl = headline(hl_rows)
    print("HEADLINE (code-faithful Phi_R):",
          json.dumps(hl, indent=2, default=float), flush=True)

    # ---- C5 prediction harness ---------------------------------------
    for r in runs:
        r["sr_classic09"] = r["sr"]
        r["sr_quantile12"] = quantile_labels(
            r["counts"], r["fission_steps"], seed=r["seed"])
    c5 = {}
    for label_rule in ("classic09", "quantile12"):
        for variant in PHI_ROWS:
            for cfg in CONFIGS:
                name = f"{label_rule}/{variant}/in{cfg['n_in']}"
                row = experiment(runs, variant, f"sr_{label_rule}", cfg)
                c5[name] = row
                flag = "  *** BEATS ALL ***" if row["beats_all"] else ""
                print(name, json.dumps(row), flag, flush=True)

    out = {"sign_regime": sign, "phyid_xcheck": xcheck,
           "headline_phir": hl, "c5": c5}
    (ROOT / "results" / "recovery_phir.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("written results/recovery_phir.json")


if __name__ == "__main__":
    main()
