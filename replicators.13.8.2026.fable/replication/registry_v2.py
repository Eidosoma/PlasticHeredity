"""Part B: registry v2 — the deduplicated frozen coordinate.

v2 = PCA-12 of the 142 beta-conditioned graph/state coordinates
     + the 8 unique direct history/phase variables
     (fissionsSinceLatestBreak dropped: exact duplicate of
     trailingInheritanceRun; regimeDuration retained — informative when
     the current boundary is a break — with its conditional identity
     documented), ridge logistic C=0.1.

By construction v2 has no cross-block duplication: the beta-conditioned
block contains neither mass nor any raw-composition coordinate.

Train on the regenerated 25x development cohort, freeze, then confirm
on a FRESH untouched cohort (tag v2-conf-2026-08-13, 200 matrices,
2,000 states, 64 branches in halves of 32) against v1-full (the frozen
25x model) and the direct baselines.
"""

import hashlib
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cohort
import models
from run_ablation import (BETA_IDX, SINCE_BREAK_COL, logit, evaluate,
                          collect_conf)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_v2")
DEV_TAG = "25x-2026-08-13"
N_DEV = 1000
V2_TAG = "v2-conf-2026-08-13"
N_V2 = 200
N_WORKERS = 12
N_PERM = 512
EPS = 1e-7

DIRECT8_COLS = [i for i in range(9) if i != SINCE_BREAK_COL]


def train_v2(X9, X195, y):
    sc8 = StandardScaler().fit(X9[:, DIRECT8_COLS])
    Z8 = sc8.transform(X9[:, DIRECT8_COLS])
    scb = StandardScaler().fit(X195[:, BETA_IDX])
    pca = PCA(n_components=12, svd_solver="full", random_state=0)
    Z = pca.fit_transform(scb.transform(X195[:, BETA_IDX]))
    return {
        "sc8": sc8, "scb": scb, "pca": pca,
        "v2": logit().fit(np.hstack([Z, Z8]), y),
        "direct8": logit().fit(Z8, y),
    }


def predict_v2(bundle, X9, X195):
    Z8 = bundle["sc8"].transform(X9[:, DIRECT8_COLS])
    Z = bundle["pca"].transform(bundle["scb"].transform(X195[:, BETA_IDX]))
    clip = lambda p: np.clip(p, EPS, 1 - EPS)
    return {
        "v2": clip(bundle["v2"].predict_proba(np.hstack([Z, Z8]))[:, 1]),
        "direct8": clip(bundle["direct8"].predict_proba(Z8)[:, 1]),
    }


def permutation_p(p, q, mats, landmarks, n_perm=N_PERM):
    counts = {m: np.sum(mats == m) for m in np.unique(mats)}
    full = [m for m, c in counts.items() if c == 5]
    keep = np.isin(mats, full)
    order = np.lexsort((landmarks[keep], mats[keep]))
    qb = q[keep][order].reshape(len(full), 5)
    pb = p[keep][order].reshape(len(full), 5)
    obs = float(spearmanr(pb.ravel(), qb.ravel()).correlation)
    prng = np.random.default_rng(913)
    exceed = sum(
        float(spearmanr(pb[prng.permutation(len(full))].ravel(),
                        qb.ravel()).correlation) >= obs
        for _ in range(n_perm))
    return (1 + exceed) / (n_perm + 1)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- train on regenerated 25x dev --------------------------------
    cohort.DEV_ENTROPY = cohort.domain_entropy("dev", DEV_TAG)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV)]
    with Pool(N_WORKERS) as pool:
        dev_units = pool.map(cohort.dev_unit, jobs)
    print(f"dev regenerated in {time.time()-t0:.0f}s")
    bundles = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in dev_units if u["candidate"] == cand
              and len(u["y"]) > 0]
        X9 = np.vstack([u["X9"] for u in cu])
        X195 = np.vstack([u["X195"] for u in cu])
        y = np.concatenate([u["y"] for u in cu])
        bundles[cand] = train_v2(X9, X195, y)
    del dev_units
    blob = pickle.dumps(bundles, protocol=4)
    with open(os.path.join(OUT, "frozen_models_v2.pkl"), "wb") as f:
        f.write(blob)
    v2_hash = hashlib.sha256(blob).hexdigest()
    print("v2 frozen:", v2_hash[:16])

    # ---- fresh untouched confirmation cohort -------------------------
    v1_bundles, v1_hash = models.thaw(
        os.path.join(HERE, "results_25x", "frozen_models.pkl"))
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", V2_TAG)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_V2)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(cohort.conf_unit, jobs)
    print(f"v2 confirmation cohort simulated in {time.time()-t0:.0f}s")

    results = {"v2_models_sha256": v2_hash, "v1_models_sha256": v1_hash,
               "v2_entropy_hex": hex(cohort.CONF_ENTROPY)}
    for cand in cohort.CANDIDATES:
        rows = []
        for u in units:
            if u["candidate"] != cand:
                continue
            for s in u["states"]:
                rows.append({**s, "Xbeta": u["Xbeta"]})
        X9, X195, Xb, qA, qB, Y, mats = collect_conf(rows)
        landmarks = np.array([r["landmark"] for r in rows])

        p2 = predict_v2(bundles[cand], X9, X195)
        p1 = models.predict(v1_bundles[cand], X9, X195, Xb)
        preds = {"direct": p1["direct"], "direct8": p2["direct8"],
                 "v1_full": p1["full"], "v2": p2["v2"]}
        r = evaluate(preds, qA, qB, Y, mats, n_boot=2048)
        # v2-vs-v1 log-loss delta with bootstrap
        ll = {}
        for name, p in preds.items():
            ll[name] = (-(Y * np.log(p[:, None])
                          + (1 - Y) * np.log(1 - p[:, None]))).mean(axis=1)
        d21 = ll["v1_full"] - ll["v2"]
        rng = np.random.default_rng(777)
        u_mats = np.unique(mats)
        idx_map = {m: np.where(mats == m)[0] for m in u_mats}
        boot = np.empty(2048)
        for i in range(2048):
            pick = rng.choice(u_mats, size=len(u_mats), replace=True)
            boot[i] = d21[np.concatenate([idx_map[m] for m in pick])].mean()
        r["_v2_minus_v1_logloss_gain"] = {
            "mean": float(d21.mean()),
            "ci": [float(np.quantile(boot, 0.025)),
                   float(np.quantile(boot, 0.975))]}
        r["_reliability"] = float(spearmanr(qA, qB).correlation)
        r["_perm_p_v2"] = permutation_p(preds["v2"], (qA + qB) / 2,
                                        mats, landmarks)
        results[cand] = {k: v for k, v in r.items()}
        results[f"{cand}_n_states"] = len(rows)

        print(f"\n=== v2 confirmation, candidate {cand} "
              f"({len(rows)} states) ===")
        print(f"reliability qA~qB: {r['_reliability']:.3f} | "
              f"perm p (v2): {r['_perm_p_v2']:.6f}")
        for name in ["direct", "direct8", "v1_full", "v2"]:
            v = r[name]
            print(f"{name:8s} overall [{v['overall'][0]:.3f},"
                  f"{v['overall'][1]:.3f}] centered "
                  f"[{v['centered'][0]:.3f},{v['centered'][1]:.3f}] "
                  f"logloss {v['logloss']:.4f} gain-vs-direct "
                  f"{v['gain_vs_direct']:+.4f} (low {v['gain_lower95']:+.4f})")
        d = r["_v2_minus_v1_logloss_gain"]
        print(f"v2 - v1_full logloss gain: {d['mean']:+.5f} "
              f"[{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]")

    with open(os.path.join(OUT, "v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten:", os.path.join(OUT, "v2_results.json"))


if __name__ == "__main__":
    main()
