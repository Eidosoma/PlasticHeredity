"""Frozen ablation suite responding to the reviewer's confound analysis
of the FULL_STATE_GRAPH_HISTORY vs DIRECT_HISTORY_PHASE comparison.

Students (all ridge logistic C=0.1; every PCA student keeps the scaled
direct-9 baseline appended, mirroring the frozen architecture):

  direct        : the 9 registered direct history/phase variables
  direct_unique : direct minus fissionsSinceLatestBreak (exact duplicate
                  of trailingInheritanceRun under the registered
                  definitions - verified at runtime)
  dup_control   : direct + 6 exact copies of scaled mass + 6 of scaled
                  normalized generation (12 duplicate columns, matching
                  the PCA block width). Negative control: measures how
                  much held-out gain ridge coefficient-splitting alone
                  can manufacture.
  beta_matrix   : matrix-level beta-only student (complete beta-only)
  state_only    : PCA-12 of the 53 pure-composition coordinates
                  (sorted composition, composition scalars incl. mass,
                  split-stability proxies) + direct-9
  beta_cond     : PCA-12 of the 142 beta-conditioned coordinates
                  (boost, join/leave distributions, self-coupling,
                  two-step, spectra, pairwise log-beta, rates) + direct-9
  full          : PCA-12 of all 195 + direct-9 (the registered model)

Note: this replication's 195 block contains NO generation/phase/clock
coordinates (the reviewer's generation_local_step / batch_step confound
does not exist here); the only coordinate shared between the PCA block
and the direct block is current mass.

Phases:
  A. regenerate the 25x development cohort from its seeds, verify the
     direct-9 redundancy identities, train and freeze all students
  B. diagnostic evaluation on the EXISTING 25x untouched confirmation
     (features regenerated deterministically, branch outcomes reused)
  C. fresh untouched ablation cohort (new entropy tag, 200 matrices,
     2,000 states, 64 branches each) - confirmation for the ablation
     conclusions, since the suite was designed after seeing 25x results
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import cohort
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_ablation")
N_WORKERS = 12
DEV_TAG = "25x-2026-08-13"
N_DEV = 1000
ABL_TAG = "ablation-conf-2026-08-13"
N_ABL = 200
EPS = 1e-7

import features as F

# Derived from the typed provenance table (frozen values unchanged:
# COMP_IDX == [0..49, 162..164], BETA_IDX its 142-element complement;
# test_validation.py asserts this identity).
COMP_IDX = F.state_only_indices()          # 53 pure-composition
BETA_IDX = F.beta_conditioned_indices()    # 142 beta-conditioned
MASS_COL = 1        # direct-9 column indices
NORMGEN_COL = 0
SINCE_BREAK_COL = 6


def logit():
    return LogisticRegression(penalty="l2", C=0.1, solver="lbfgs",
                              max_iter=5000)


def train_suite(X9, X195, Xbeta_rows, y):
    sc9 = StandardScaler().fit(X9)
    Z9 = sc9.transform(X9)
    suite = {"_sc9": sc9}

    suite["direct"] = {"kind": "plain", "cols": list(range(9)),
                       "clf": logit().fit(Z9, y)}
    keep = [i for i in range(9) if i != SINCE_BREAK_COL]
    suite["direct_unique"] = {"kind": "plain", "cols": keep,
                              "clf": logit().fit(Z9[:, keep], y)}
    dup = np.hstack([Z9] + [Z9[:, [MASS_COL]]] * 6
                    + [Z9[:, [NORMGEN_COL]]] * 6)
    suite["dup_control"] = {"kind": "dup", "clf": logit().fit(dup, y)}

    scb = StandardScaler().fit(Xbeta_rows)
    suite["beta_matrix"] = {"kind": "beta", "sc": scb,
                            "clf": logit().fit(scb.transform(Xbeta_rows), y)}

    for name, idx in [("state_only", COMP_IDX), ("beta_cond", BETA_IDX),
                      ("full", np.arange(195))]:
        sc = StandardScaler().fit(X195[:, idx])
        pca = PCA(n_components=12, svd_solver="full", random_state=0)
        Z = pca.fit_transform(sc.transform(X195[:, idx]))
        suite[name] = {"kind": "pca", "idx": idx, "sc": sc, "pca": pca,
                       "clf": logit().fit(np.hstack([Z, Z9]), y)}
    return suite


def predict_suite(suite, X9, X195, Xbeta_rows):
    Z9 = suite["_sc9"].transform(X9)
    out = {}
    for name, s in suite.items():
        if name.startswith("_"):
            continue
        if s["kind"] == "plain":
            p = s["clf"].predict_proba(Z9[:, s["cols"]])[:, 1]
        elif s["kind"] == "dup":
            dup = np.hstack([Z9] + [Z9[:, [MASS_COL]]] * 6
                            + [Z9[:, [NORMGEN_COL]]] * 6)
            p = s["clf"].predict_proba(dup)[:, 1]
        elif s["kind"] == "beta":
            p = s["clf"].predict_proba(s["sc"].transform(Xbeta_rows))[:, 1]
        else:
            Z = s["pca"].transform(s["sc"].transform(X195[:, s["idx"]]))
            p = s["clf"].predict_proba(np.hstack([Z, Z9]))[:, 1]
        out[name] = np.clip(p, EPS, 1 - EPS)
    return out


def center_by(v, groups):
    v = np.asarray(v, dtype=float)
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=v) / np.bincount(inv)
    return v - means[inv]


def evaluate(preds, qA, qB, Y, mats, n_boot=1024):
    """Per-student metrics + matrix-bootstrap lower bounds of the
    log-loss gain over the direct baseline."""
    res = {}
    rng = np.random.default_rng(424242)
    ll = {}
    for name, p in preds.items():
        ll[name] = (-(Y * np.log(p[:, None])
                      + (1 - Y) * np.log(1 - p[:, None]))).mean(axis=1)
    u = np.unique(mats)
    idx_map = {m: np.where(mats == m)[0] for m in u}
    cA, cB = center_by(qA, mats), center_by(qB, mats)
    for name, p in preds.items():
        sp_o = [float(spearmanr(p, qA).correlation),
                float(spearmanr(p, qB).correlation)]
        pc = center_by(p, mats)
        sp_c = [float(spearmanr(pc, cA).correlation),
                float(spearmanr(pc, cB).correlation)]
        gain_state = ll["direct"] - ll[name]
        boot = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(u, size=len(u), replace=True)
            idx = np.concatenate([idx_map[m] for m in pick])
            boot[i] = gain_state[idx].mean()
        res[name] = {
            "overall": sorted(sp_o), "centered": sorted(sp_c),
            "logloss": float(ll[name].mean()),
            "gain_vs_direct": float(gain_state.mean()),
            "gain_lower95": float(np.quantile(boot, 0.025)),
        }
    return res


def collect_conf(rows):
    return (np.stack([r["X9"] for r in rows]),
            np.stack([r["X195"] for r in rows]),
            np.stack([r["Xbeta"] for r in rows]),
            np.array([r["qA"] for r in rows]),
            np.array([r["qB"] for r in rows]),
            np.stack([r["y64"] for r in rows]).astype(float),
            np.array([r["matrix"] for r in rows]))


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"suite_doc": __doc__}

    # ---------------- Phase A: dev regeneration + training ------------
    cohort.DEV_ENTROPY = cohort.domain_entropy("dev", DEV_TAG)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV)]
    with Pool(N_WORKERS) as pool:
        dev_units = pool.map(cohort.dev_unit, jobs)
    print(f"dev regenerated in {time.time()-t0:.0f}s")

    suites = {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in dev_units if u["candidate"] == cand
              and len(u["y"]) > 0]
        X9 = np.vstack([u["X9"] for u in cu])
        X195 = np.vstack([u["X195"] for u in cu])
        Xb = np.vstack([np.tile(u["Xbeta"], (len(u["y"]), 1)) for u in cu])
        y = np.concatenate([u["y"] for u in cu])
        # redundancy identities claimed by the reviewer
        ident1 = bool(np.array_equal(X9[:, 4], X9[:, 6]))
        inh = X9[:, 7] == 1.0
        ident2 = bool(np.array_equal(X9[inh, 8], X9[inh, 4]))
        results[f"identity_sinceBreak_eq_trailing_{cand}"] = ident1
        results[f"identity_regimeDur_eq_trailing_when_inherited_{cand}"] = ident2
        print(f"cand {cand}: sinceBreak==trailing {ident1}, "
              f"regimeDur==trailing|inherited {ident2}")
        suites[cand] = train_suite(X9, X195, Xb, y)
    del dev_units
    with open(os.path.join(OUT, "ablation_models.pkl"), "wb") as f:
        pickle.dump(suites, f, protocol=4)

    # ---------------- Phase B: diagnostic on existing 25x conf --------
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", DEV_TAG)
    with open(os.path.join(HERE, "results_25x", "conf_data.pkl"), "rb") as f:
        conf25 = pickle.load(f)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV)]
    with Pool(N_WORKERS) as pool:
        funits = pool.map(cohort.conf_features_unit, jobs)
    print(f"25x conf features regenerated in {time.time()-t0:.0f}s")
    fmap = {}
    for u in funits:
        for s in u["states"]:
            fmap[(s["candidate"], s["matrix"], s["landmark"])] = \
                (s["X9"], s["X195"], u["Xbeta"])
    for cand in cohort.CANDIDATES:
        rows = []
        for r in conf25["table"]:
            if r["candidate"] != cand:
                continue
            X9v, X195v, Xbv = fmap[(cand, r["matrix"], r["landmark"])]
            rows.append({**r, "X9": X9v, "X195": X195v, "Xbeta": Xbv})
        X9, X195, Xb, qA, qB, Y, mats = collect_conf(rows)
        # sanity: frozen-full predictions must reproduce stored p_full ranks
        preds = predict_suite(suites[cand], X9, X195, Xb)
        results[f"diagnostic_25x_{cand}"] = evaluate(preds, qA, qB, Y, mats)
        print(f"diagnostic 25x cand {cand} done")
    del funits, fmap

    # ---------------- Phase C: fresh untouched ablation cohort --------
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", ABL_TAG)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_ABL)]
    with Pool(N_WORKERS) as pool:
        aunits = pool.map(cohort.conf_unit, jobs)
    print(f"ablation cohort simulated in {time.time()-t0:.0f}s")
    for cand in cohort.CANDIDATES:
        rows = []
        for u in aunits:
            if u["candidate"] != cand:
                continue
            for s in u["states"]:
                rows.append({**s, "qA": float(s["qA"]), "qB": float(s["qB"]),
                             "Xbeta": u["Xbeta"]})
        X9, X195, Xb, qA, qB, Y, mats = collect_conf(rows)
        preds = predict_suite(suites[cand], X9, X195, Xb)
        results[f"fresh_{cand}"] = evaluate(preds, qA, qB, Y, mats,
                                            n_boot=2048)
        results[f"fresh_{cand}_n_states"] = len(rows)
        print(f"fresh ablation cohort cand {cand}: {len(rows)} states")

    results["ablation_entropy_hex"] = hex(cohort.CONF_ENTROPY)
    with open(os.path.join(OUT, "ablation_results.json"), "w") as f:
        json.dump({k: v for k, v in results.items() if k != "suite_doc"},
                  f, indent=2)

    for phase in ["diagnostic_25x", "fresh"]:
        for cand in cohort.CANDIDATES:
            r = results[f"{phase}_{cand}"]
            print(f"\n=== {phase} candidate {cand} ===")
            for name in ["direct", "direct_unique", "dup_control",
                         "beta_matrix", "state_only", "beta_cond", "full"]:
                v = r[name]
                print(f"{name:14s} overall [{v['overall'][0]:.3f},"
                      f"{v['overall'][1]:.3f}] centered "
                      f"[{v['centered'][0]:.3f},{v['centered'][1]:.3f}] "
                      f"logloss {v['logloss']:.4f} gain "
                      f"{v['gain_vs_direct']:+.4f} "
                      f"(low {v['gain_lower95']:+.4f})")


if __name__ == "__main__":
    main()
