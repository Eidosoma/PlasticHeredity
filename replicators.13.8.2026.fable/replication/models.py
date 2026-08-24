"""Frozen student models (per candidate).

Students, mirroring the paper's registry:
  prior  : development base-rate constant
  direct : ridge logistic on the 9 direct history/phase variables
  beta   : ridge logistic on matrix-level beta-only features
  full   : development-only scaling of the 195 graph/state block,
           12-component PCA, concatenated with the scaled 9 direct
           variables, ridge logistic regression with C=0.1

Everything is fit on development data only and then frozen (pickled with
a SHA-256 identity); the confirmation stage loads and applies without
refitting.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C_RIDGE = 0.1
PCA_DIM = 12
EPS = 1e-7


def _logistic():
    return LogisticRegression(penalty="l2", C=C_RIDGE, solver="lbfgs",
                              max_iter=5000)


def train_students(X9, X195, Xbeta_rows, y):
    prior = float(np.mean(y))

    sc9 = StandardScaler().fit(X9)
    direct = _logistic().fit(sc9.transform(X9), y)

    scb = StandardScaler().fit(Xbeta_rows)
    beta_m = _logistic().fit(scb.transform(Xbeta_rows), y)

    sc195 = StandardScaler().fit(X195)
    pca = PCA(n_components=PCA_DIM, svd_solver="full", random_state=0)
    Z = pca.fit_transform(sc195.transform(X195))
    Xfull = np.hstack([Z, sc9.transform(X9)])
    full = _logistic().fit(Xfull, y)

    return {
        "prior": prior,
        "sc9": sc9, "direct": direct,
        "scb": scb, "beta": beta_m,
        "sc195": sc195, "pca": pca, "full": full,
    }


def predict(bundle, X9, X195, Xbeta_rows):
    p_prior = np.full(len(X9), bundle["prior"])
    p_direct = bundle["direct"].predict_proba(
        bundle["sc9"].transform(X9))[:, 1]
    p_beta = bundle["beta"].predict_proba(
        bundle["scb"].transform(Xbeta_rows))[:, 1]
    Z = bundle["pca"].transform(bundle["sc195"].transform(X195))
    Xfull = np.hstack([Z, bundle["sc9"].transform(X9)])
    p_full = bundle["full"].predict_proba(Xfull)[:, 1]
    clip = lambda p: np.clip(p, EPS, 1 - EPS)
    return {"prior": clip(p_prior), "direct": clip(p_direct),
            "beta": clip(p_beta), "full": clip(p_full)}


def freeze(bundles: dict, path: str) -> str:
    blob = pickle.dumps(bundles, protocol=4)
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def thaw(path: str):
    with open(path, "rb") as f:
        blob = f.read()
    return pickle.loads(blob), hashlib.sha256(blob).hexdigest()
