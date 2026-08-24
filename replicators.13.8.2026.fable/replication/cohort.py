"""Cohort generation: development (L53-analog) and untouched
confirmation (L54-analog) units.

Seed architecture (domain-separated, documented):
  DEV_ENTROPY  = sha256("replication-dev-domain-2026-08-13")
  CONF_ENTROPY = sha256("replication-confirmation-domain-2026-08-13")
  spawn keys:
    (0, m)                 -> catalytic matrix beta for matrix index m
    (1, m)                 -> matched initial state for matrix index m
    (2, cand_i, m)         -> main trajectory stream
    (3, cand_i, m, lm, b)  -> branch b at landmark lm

Matrices and initial states are shared between candidates. The
confirmation campaign is regenerated exactly (same seeds) a second time
and compared by hash (replay gate).
"""

from __future__ import annotations

import hashlib

import numpy as np

import sim
from sim import NG, H_THRESH
import features as F

def domain_entropy(kind: str, tag: str = "2026-08-13") -> int:
    return int(hashlib.sha256(
        f"replication-{kind}-domain-{tag}".encode()).hexdigest(), 16)


# Defaults (the 1x campaign). Run scripts may override these module
# globals with a different tag BEFORE creating worker pools; workers
# inherit the override via fork.
DEV_ENTROPY = domain_entropy("dev")
CONF_ENTROPY = domain_entropy("confirmation")

N_FISSIONS = 100
HORIZON = 12
LANDMARKS = [20, 35, 50, 65, 80]
N_BRANCHES = 64
HALF = 32

CANDIDATES = ["02", "03"]


def _rng(entropy, *key):
    return np.random.default_rng(np.random.SeedSequence(entropy=entropy, spawn_key=key))


def matrix_and_init(entropy: int, m: int):
    beta = sim.make_beta(_rng(entropy, 0, m))
    n0 = sim.make_initial_state(_rng(entropy, 1, m))
    return beta, n0


# ----------------------------------------------------------------------
# Development unit: one (matrix, candidate) trajectory -> training rows
# ----------------------------------------------------------------------

def dev_unit(args):
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(DEV_ENTROPY, m)
    rng = _rng(DEV_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, inh, daughters = traj["H"], traj["inherited"], traj["daughters"]
    xb = F.beta_only(beta)
    rows9, rows195, ys, gs = [], [], [], []
    for g in range(1, nd - HORIZON + 1):          # need full 12-fission future
        state = daughters[g - 1]
        rows9.append(F.direct9(g, N_FISSIONS, hs[:g], int(state.sum())))
        rows195.append(F.graph_state_195(state, beta))
        ys.append(float(F.joint_break_run3(inh[g:g + HORIZON])))
        gs.append(g)
    return {
        "matrix": m, "candidate": cand,
        "X9": np.array(rows9).reshape(len(rows9), 9),
        "X195": np.array(rows195).reshape(len(rows195), 195),
        "Xbeta": xb, "y": np.array(ys), "g": np.array(gs),
        "died": traj["died"], "n_done": nd,
    }


# ----------------------------------------------------------------------
# Confirmation unit: one (matrix, candidate) trajectory + 5 landmark
# states x 64 branches
# ----------------------------------------------------------------------

PROC_KEYS = ["break", "resume2", "episode3", "persist5",
             "old_return", "pos_gain", "gain", "repeat"]


def conf_unit(args):
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(CONF_ENTROPY, m)
    rng = _rng(CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, daughters = traj["H"], traj["daughters"]
    xb = F.beta_only(beta)
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        x9 = F.direct9(lm, N_FISSIONS, hs[:lm], int(restored.sum()))
        x195 = F.graph_state_195(restored, beta)
        y64 = np.zeros(N_BRANCHES)
        proc = {k: [] for k in PROC_KEYS}
        for b in range(N_BRANCHES):
            rb = _rng(CONF_ENTROPY, 3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, HORIZON, rb)
            inh = br["inherited"]
            y64[b] = float(F.joint_break_run3(inh))
            po = F.process_outcomes(inh, br["daughters"], restored)
            for k in PROC_KEYS:
                proc[k].append(po[k])
        states.append({
            "matrix": m, "candidate": cand, "landmark": lm,
            "X9": x9, "X195": x195, "y64": y64,
            "qA": y64[:HALF].mean(), "qB": y64[HALF:].mean(),
            "proc": {k: np.array(v, dtype=float) for k, v in proc.items()},
        })
    return {"matrix": m, "candidate": cand, "Xbeta": xb,
            "states": states, "n_done": nd, "died": traj["died"]}


def conf_sequences_unit(args):
    """Regenerate a confirmation unit's branches and return each
    branch's raw inheritance-flag sequence (12 fissions) per landmark
    state. Deterministic from the same seeds as conf_unit, so sequences
    correspond exactly to the published branch outcomes."""
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(CONF_ENTROPY, m)
    rng = _rng(CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    daughters = traj["daughters"]
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        seqs = np.zeros((N_BRANCHES, HORIZON), dtype=bool)
        lens = np.zeros(N_BRANCHES, dtype=np.int64)
        for b in range(N_BRANCHES):
            rb = _rng(CONF_ENTROPY, 3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, HORIZON, rb)
            inh = br["inherited"]
            seqs[b, :len(inh)] = inh
            lens[b] = len(inh)
        states.append({"matrix": m, "candidate": cand, "landmark": lm,
                       "seqs": seqs, "lens": lens})
    return {"matrix": m, "candidate": cand, "states": states}


def conf_coherence_unit(args):
    """Regenerate a confirmation unit's branches computing, per branch,
    the registered coherence indicators (features.coherence_outcomes):
    joint / coherent / distinct booleans plus span and anchor
    similarities. Same seeds as conf_unit — q for the joint target must
    reproduce the stored cohort exactly."""
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(CONF_ENTROPY, m)
    rng = _rng(CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    daughters = traj["daughters"]
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        rec = {k: np.zeros(N_BRANCHES) for k in
               ["joint", "coherent", "distinct"]}
        span = np.full(N_BRANCHES, np.nan)
        anch = np.full(N_BRANCHES, np.nan)
        for b in range(N_BRANCHES):
            rb = _rng(CONF_ENTROPY, 3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, HORIZON, rb)
            co = F.coherence_outcomes(br["inherited"], br["daughters"],
                                      restored)
            for k in ("joint", "coherent", "distinct"):
                rec[k][b] = float(co[k])
            span[b] = co["span_sim"]
            anch[b] = co["anchor_sim"]
        states.append({"matrix": m, "candidate": cand, "landmark": lm,
                       "y": rec, "span": span, "anchor": anch})
    return {"matrix": m, "candidate": cand, "states": states}


def conf_h_sequences_unit(args):
    """Like conf_sequences_unit but returns each branch's raw
    parent->daughter H values (float32) per fission, enabling
    re-thresholding for target-sensitivity analysis."""
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(CONF_ENTROPY, m)
    rng = _rng(CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    daughters = traj["daughters"]
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        hvals = np.zeros((N_BRANCHES, HORIZON), dtype=np.float32)
        lens = np.zeros(N_BRANCHES, dtype=np.int64)
        for b in range(N_BRANCHES):
            rb = _rng(CONF_ENTROPY, 3, cand_i, m, lm, b)
            br = sim.run_fissions(restored, beta, cand, HORIZON, rb)
            h = br["H"]
            hvals[b, :len(h)] = h
            lens[b] = len(h)
        states.append({"matrix": m, "candidate": cand, "landmark": lm,
                       "H64": hvals, "lens": lens})
    return {"matrix": m, "candidate": cand, "states": states}


def conf_features_unit(args):
    """Regenerate ONLY the trajectory and landmark features of a
    confirmation unit (no branches). Deterministic from the same seeds,
    so it reattaches exactly to stored branch outcomes."""
    m, cand = args
    cand_i = CANDIDATES.index(cand)
    beta, n0 = matrix_and_init(CONF_ENTROPY, m)
    rng = _rng(CONF_ENTROPY, 2, cand_i, m)
    traj = sim.run_fissions(n0, beta, cand, N_FISSIONS, rng)
    nd = traj["n_done"]
    hs, daughters = traj["H"], traj["daughters"]
    states = []
    for lm in LANDMARKS:
        if lm > nd:
            continue
        restored = daughters[lm - 1]
        states.append({
            "matrix": m, "candidate": cand, "landmark": lm,
            "X9": F.direct9(lm, N_FISSIONS, hs[:lm], int(restored.sum())),
            "X195": F.graph_state_195(restored, beta),
        })
    return {"matrix": m, "candidate": cand, "Xbeta": F.beta_only(beta),
            "states": states}


def campaign_hash(units) -> str:
    """Deterministic hash over all branch outcomes and q values."""
    h = hashlib.sha256()
    for u in sorted(units, key=lambda u: (u["candidate"], u["matrix"])):
        for s in u["states"]:
            h.update(np.ascontiguousarray(s["y64"]).tobytes())
            h.update(np.ascontiguousarray(s["X9"]).tobytes())
            h.update(np.ascontiguousarray(s["X195"]).tobytes())
    return h.hexdigest()
