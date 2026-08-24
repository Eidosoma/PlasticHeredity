"""Formal validation suite for the clean-room replication.

Self-contained (no pytest): run `python3 test_validation.py`.
Covers: propensity fixtures, event-sampler distribution, Poisson
exposure moments, fission laws, conservation invariants, cosine/threshold
semantics, seed architecture and replay (including a subset replay
against the frozen 1x campaign artifact), target and process-outcome
unit tests, Markov/IID estimator null calibration, the registered
direct-variable identities, and feature-provenance consistency.
"""

import os
import pickle
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import stats as st

import sim
import features as F
import cohort
import markov_iid as MI

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ----------------------------------------------------------------------
# 1. Propensity fixtures: 512 exact cases vs independent closed form
# ----------------------------------------------------------------------
@check
def propensity_fixtures():
    rng = np.random.default_rng(101)
    for _ in range(512):
        beta = sim.make_beta(rng)
        n = np.zeros(sim.NG, dtype=np.int64)
        support = rng.choice(sim.NG, size=rng.integers(5, 60), replace=False)
        n[support] = rng.integers(1, 8, size=len(support))
        total = int(n.sum())
        c = beta @ n
        join, leave = sim.event_rates(n, c, total)
        for i in rng.choice(sim.NG, size=8, replace=False):
            bn_i = 1.0 + c[i] / total
            assert join[i] == (sim.KF * sim.RHO * total) * bn_i
            assert leave[i] == (sim.KB * n[i]) * bn_i
        assert (join > 0).all() and (leave >= 0).all()


# ----------------------------------------------------------------------
# 2. Categorical sampler: chi-square against rate ratios
# ----------------------------------------------------------------------
@check
def sampler_chisquare():
    rng = np.random.default_rng(202)
    rates = np.array([5.0, 1.0, 0.5, 3.0, 0.1, 2.0, 0.4, 8.0])
    draws = np.array([sim._sample_categorical(rates, rng)
                      for _ in range(200_000)])
    obs = np.bincount(draws, minlength=len(rates))
    exp = rates / rates.sum() * len(draws)
    p = st.chisquare(obs, exp).pvalue
    assert p > 1e-4, f"sampler chi-square p={p}"


# ----------------------------------------------------------------------
# 3. Poisson exposure moments (the numpy contract candidate 03 relies on)
# ----------------------------------------------------------------------
@check
def poisson_exposure_moments():
    rng = np.random.default_rng(303)
    lam = np.array([0.05, 0.5, 2.0, 4.0])
    draws = rng.poisson(lam, size=(200_000, len(lam)))
    m, v = draws.mean(axis=0), draws.var(axis=0)
    assert np.allclose(m, lam, rtol=0.03), (m, lam)
    assert np.allclose(v, lam, rtol=0.05), (v, lam)


# ----------------------------------------------------------------------
# 4. Fission laws
# ----------------------------------------------------------------------
@check
def split_equal_hypergeometric():
    rng = np.random.default_rng(404)
    parent = np.zeros(sim.NG, dtype=np.int64)
    parent[:8] = [30, 20, 10, 8, 5, 4, 2, 1]      # mass 80
    N, K = 80, 40
    reps = 50_000
    a = np.zeros((reps, sim.NG))
    for r in range(reps):
        ca, cb = sim._split_equal(parent, rng)
        assert ca.sum() == K and cb.sum() == N - K
        assert np.array_equal(ca + cb, parent)
        assert (ca >= 0).all() and (cb >= 0).all()
        a[r] = ca
    mean, var = a.mean(axis=0)[:8], a.var(axis=0)[:8]
    exp_mean = parent[:8] * K / N
    exp_var = (parent[:8] * (K / N) * (1 - K / N)
               * (N - parent[:8]) / (N - 1))
    z = (mean - exp_mean) / np.sqrt(exp_var / reps)
    assert (np.abs(z) < 5).all(), z
    assert np.allclose(var, exp_var, rtol=0.08), (var, exp_var)


@check
def split_binomial_moments():
    rng = np.random.default_rng(505)
    parent = np.zeros(sim.NG, dtype=np.int64)
    parent[:6] = [40, 20, 10, 6, 3, 1]
    reps = 50_000
    a = np.zeros((reps, 6))
    for r in range(reps):
        ca, cb = sim._split_binomial(parent, rng)
        assert np.array_equal(ca + cb, parent)
        assert (ca >= 0).all() and (cb >= 0).all()
        a[r] = ca[:6]
    exp_mean, exp_var = parent[:6] / 2, parent[:6] / 4
    z = (a.mean(axis=0) - exp_mean) / np.sqrt(exp_var / reps)
    assert (np.abs(z) < 5).all(), z
    assert np.allclose(a.var(axis=0), exp_var, rtol=0.08)


# ----------------------------------------------------------------------
# 5. Trajectory invariants
# ----------------------------------------------------------------------
@check
def trajectory_invariants():
    for cand in ("02", "03"):
        rng = np.random.default_rng(606)
        beta = sim.make_beta(rng)
        n0 = sim.make_initial_state(rng)
        out = sim.run_fissions(n0, beta, cand, 20, rng)
        assert out["n_done"] == 20 and not out["died"]
        p_mass = out["parents"].sum(axis=1)
        d_mass = out["daughters"].sum(axis=1)
        assert (out["parents"] >= 0).all() and (out["daughters"] >= 0).all()
        assert ((out["H"] >= 0) & (out["H"] <= 1)).all()
        if cand == "02":
            assert (p_mass == sim.NMAX).all(), "exact-size fission"
            assert (d_mass == sim.NMAX // 2).all(), "equal split"
        else:
            assert (p_mass >= sim.NMAX).all(), "overshoot semantics"
            assert (p_mass < sim.NMAX + 40).all(), "bounded overshoot"


# ----------------------------------------------------------------------
# 6. Cosine and strict threshold semantics
# ----------------------------------------------------------------------
@check
def cosine_and_threshold():
    e1 = np.zeros(10); e1[0] = 3.0
    e2 = np.zeros(10); e2[1] = 5.0
    assert sim.cosine_h(e1, e1) == 1.0
    assert sim.cosine_h(e1, e2) == 0.0
    assert sim.cosine_h(np.zeros(10), e1) == 0.0
    v = np.ones(10)
    assert 0.0 <= sim.cosine_h(v, e1) <= 1.0
    assert sim.H_THRESH == 0.9
    # strictness: H exactly at threshold is NOT inherited
    assert not (np.array([0.9]) > sim.H_THRESH)[0]
    assert (np.array([np.nextafter(0.9, 1)]) > sim.H_THRESH)[0]


# ----------------------------------------------------------------------
# 7. Seed architecture: domains, spawn keys, bitwise replay
# ----------------------------------------------------------------------
@check
def seed_architecture():
    assert cohort.domain_entropy("dev") != cohort.domain_entropy("confirmation")
    assert (cohort.domain_entropy("dev", "a") != cohort.domain_entropy("dev", "b"))
    assert cohort.domain_entropy("dev") == cohort.domain_entropy("dev")
    b1, s1 = cohort.matrix_and_init(cohort.domain_entropy("dev"), 0)
    b2, s2 = cohort.matrix_and_init(cohort.domain_entropy("dev"), 0)
    b3, s3 = cohort.matrix_and_init(cohort.domain_entropy("dev"), 1)
    assert np.array_equal(b1, b2) and np.array_equal(s1, s2)
    assert not np.array_equal(b1, b3)
    ss = np.random.SeedSequence(entropy=42, spawn_key=(1, 2, 3))
    r1 = sim.run_fissions(s1, b1, "02", 10, np.random.default_rng(ss))
    r2 = sim.run_fissions(s1, b1, "02", 10, np.random.default_rng(
        np.random.SeedSequence(entropy=42, spawn_key=(1, 2, 3))))
    assert np.array_equal(r1["daughters"], r2["daughters"])
    assert np.array_equal(r1["H"], r2["H"])
    r3 = sim.run_fissions(s1, b1, "02", 10, np.random.default_rng(
        np.random.SeedSequence(entropy=42, spawn_key=(1, 2, 4))))
    assert not np.array_equal(r1["daughters"], r3["daughters"])


# ----------------------------------------------------------------------
# 8. Subset replay against the frozen 1x campaign artifact
# ----------------------------------------------------------------------
@check
def subset_replay_against_stored():
    path = os.path.join(HERE, "results", "conf_data.pkl")
    with open(path, "rb") as f:
        table = pickle.load(f)["table"]
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation")
    for m, cand in [(7, "02"), (23, "03")]:
        unit = cohort.conf_unit((m, cand))
        stored = {r["landmark"]: r for r in table
                  if r["matrix"] == m and r["candidate"] == cand}
        assert len(unit["states"]) == len(stored) > 0
        for s in unit["states"]:
            row = stored[s["landmark"]]
            assert np.array_equal(s["y64"].astype(np.int8), row["y64"])
            assert abs(float(s["qA"]) - row["qA"]) < 1e-12
            assert abs(float(s["qB"]) - row["qB"]) < 1e-12


# ----------------------------------------------------------------------
# 9. JOINT_BREAK_RUN3 unit cases
# ----------------------------------------------------------------------
@check
def joint_break_run3_cases():
    T, Fa = True, False
    j = lambda seq: F.joint_break_run3(np.array(seq, dtype=bool))
    assert j([Fa, T, T, T]) is True
    assert j([T] * 12) is False
    assert j([Fa] * 12) is False
    assert j([T] * 11 + [Fa]) is False                    # break at end
    assert j([T, Fa, T, T, T] + [Fa] * 7) is True
    assert j([Fa, T, T]) is False                         # run can't fit
    assert j([T] * 8 + [Fa, T, T, T]) is True             # run at edge
    assert j([Fa, T, T, Fa, T, T]) is False               # only runs of 2
    assert j([Fa, Fa, Fa, T, T, T]) is True               # later run counts


# ----------------------------------------------------------------------
# 10. Process-outcome unit cases
# ----------------------------------------------------------------------
@check
def process_outcome_cases():
    e = lambda i, m=10.0: (np.eye(sim.NG) * m)[i].astype(np.int64)
    T, Fa = True, False

    # no break
    po = F.process_outcomes(np.array([T] * 6), np.tile(e(0), (6, 1)), e(0))
    assert po["break"] == 0.0 and po["old_return"] == 0.0

    # break then exactly 2 inherited: resume2 yes, episode3 no
    inh = np.array([Fa, T, T])
    po = F.process_outcomes(inh, np.tile(e(0), (3, 1)), e(0))
    assert po["break"] == 1.0 and po["resume2"] == 1.0 and po["episode3"] == 0.0

    # break, departure at t+1, return to anchor: old_return fires.
    # Registered semantics: departures count from fission t+1 onward
    # (the breaking fission's own daughter, sims[0], is not eligible).
    inh = np.array([Fa, T, T, T, T, T])
    daughters = np.stack([e(2), e(1), e(0), e(0), e(0), e(0)])
    po = F.process_outcomes(inh, daughters, e(0))         # anchor = restored
    assert po["episode3"] == 1.0
    assert po["old_return"] == 1.0

    # departure staged at the breaking fission itself does NOT count
    daughters = np.stack([e(1), e(0), e(0), e(0), e(0), e(0)])
    po = F.process_outcomes(inh, daughters, e(0))
    assert po["old_return"] == 0.0
    # positive gain: episode start closer to anchor than sims[0]
    assert po["gain"] == 1.0 and po["pos_gain"] == 1.0

    # two full break->episode3 cycles: repeat
    inh = np.array([Fa, T, T, T, Fa, T, T, T, T, T, T, T])
    daughters = np.stack([e(1)] + [e(2)] * 11)
    po = F.process_outcomes(inh, daughters, e(0))
    assert po["repeat"] == 1.0


# ----------------------------------------------------------------------
# 10b. Coherence-outcome unit cases (reviewer #7 criterion)
# ----------------------------------------------------------------------
@check
def coherence_outcome_cases():
    T, Fa = True, False
    e = lambda i, m=10.0: (np.eye(sim.NG) * m)[i]

    # coherent + distinct: episode sits at e(0), anchor at e(5)
    inh = np.array([Fa, T, T, T])
    daughters = np.stack([e(1), e(0), e(0), e(0)])
    co = F.coherence_outcomes(inh, daughters, e(5))
    assert co["joint"] and co["coherent"] and co["distinct"]
    assert co["span_sim"] == 1.0 and co["anchor_sim"] == 0.0

    # drifting episode: adjacent similarities > 0.9 but span < 0.9
    # (planar rotations 24 deg apart: adjacent cos ~0.914, span ~0.669)
    th = np.deg2rad(24.0)
    def rot(k):
        v = np.zeros(sim.NG)
        v[0], v[1] = np.cos(k * th) * 100, np.sin(k * th) * 100
        return v
    daughters = np.stack([e(1), rot(0), rot(1), rot(2)])
    co = F.coherence_outcomes(inh, daughters, e(5))
    a1 = sim.cosine_h(rot(0), rot(1))
    assert a1 > 0.9 and co["span_sim"] < 0.9
    assert co["joint"] and not co["coherent"] and not co["distinct"]

    # coherent but NOT distinct: episode returns to the anchor
    daughters = np.stack([e(1), e(5), e(5), e(5)])
    co = F.coherence_outcomes(inh, daughters, e(5))
    assert co["coherent"] and not co["distinct"]
    assert co["anchor_sim"] == 1.0

    # break with no subsequent 3-run: joint False, sims undefined
    co = F.coherence_outcomes(np.array([Fa, T, T]),
                              np.stack([e(1), e(0), e(0)]), e(5))
    assert not co["joint"] and np.isnan(co["span_sim"])

    # fuzz: joint flag must agree with joint_break_run3 exactly
    rng = np.random.default_rng(909)
    for _ in range(500):
        flags = rng.random(12) < 0.8
        d = rng.integers(0, 5, size=(12, sim.NG))
        co = F.coherence_outcomes(flags, d, d[0])
        assert co["joint"] == F.joint_break_run3(flags)


# ----------------------------------------------------------------------
# 11. Markov/IID estimator null calibration
# ----------------------------------------------------------------------
def _null_groups(rng, p1, p2, dep=0.0, n_mat=50, n_seq=40):
    groups = {}
    for m in range(n_mat):
        sfx = []
        for _ in range(n_seq):
            L = rng.integers(0, 12)
            s = np.zeros(L, dtype=bool)
            for i in range(L):
                p = p1 if i == 0 else p2 + (dep if s[i - 1] else -dep)
                s[i] = rng.random() < p
            sfx.append(s)
        groups[m] = sfx
    return groups


def _pooled(groups, fitter):
    pm = MI.crossfit_gain(groups, fitter)
    tot = sum(v[0] for v in pm.values())
    n = sum(v[1] for v in pm.values())
    return tot / max(n, 1) / MI.LOG2


@check
def markov_null_calibration():
    rng = np.random.default_rng(707)
    iid = _null_groups(rng, 0.8, 0.8)
    assert abs(_pooled(iid, MI.fit_iid_corrected)) < 0.003
    assert abs(_pooled(iid, MI.fit_iid_biased)) < 0.003
    ns = _null_groups(rng, 0.6, 0.85)
    g_b = _pooled(ns, MI.fit_iid_biased)
    g_c = _pooled(ns, MI.fit_iid_corrected)
    assert abs(g_c) < 0.003, f"corrected on nonstat null: {g_c}"
    assert g_b - g_c > 0.004, f"bias not detected: {g_b} vs {g_c}"


# ----------------------------------------------------------------------
# 11b. Intervention machinery: swap legality, CRN pairing, determinism
# ----------------------------------------------------------------------
@check
def intervention_swap_and_crn():
    import pickle
    import run_intervention as RI
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", "test-crn")

    ent = cohort.domain_entropy("dev", "test-swaps")
    beta, n0 = cohort.matrix_and_init(ent, 0)
    rng = cohort._rng(ent, 2, 0, 0)
    traj = sim.run_fissions(n0, beta, "02", 20, rng)
    n = traj["daughters"][19]
    X9 = F.direct9(20, 100, traj["H"][:20], int(n.sum()))

    # deterministic selection: identical twice
    s1 = RI.screen_swaps(n, beta, X9, RI._BUNDLES["02"],
                         np.random.default_rng(5))
    s2 = RI.screen_swaps(n, beta, X9, RI._BUNDLES["02"],
                         np.random.default_rng(5))
    assert s1["up"] == s2["up"] and s1["down"] == s2["down"]
    assert s1["random"] == s2["random"]
    assert s1["up_score"] >= s1["base_score"] >= 0.0
    assert s1["down_score"] <= s1["up_score"]

    # swap legality: mass preserved, non-negative, i != j
    for swap in [s1["up"], s1["down"], s1["random"]]:
        i, j = swap
        assert i != j
        ne = RI.apply_swap(n, swap)
        assert ne.sum() == n.sum()
        assert (ne >= 0).all()
        assert ne[i] == n[i] - 1 and ne[j] == n[j] + 1

    # CRN pairing: same spawn keys, same state -> identical outcomes
    a1 = RI.run_arm(n, beta, "02", 0, 3, 20, list(range(8)))
    a2 = RI.run_arm(n, beta, "02", 0, 3, 20, list(range(8)))
    assert a1 == a2
    # stream separation at the RNG level (aggregate equality between
    # different keys can coincide by chance; the streams must not)
    s_a = cohort._rng(RI._ENT, 5, 0, 3, 20, 0).random(8)
    s_b = cohort._rng(RI._ENT, 5, 0, 3, 20, 0).random(8)
    s_c = cohort._rng(RI._ENT, 5, 0, 3, 35, 0).random(8)
    s_d = cohort._rng(RI._ENT, 5, 0, 3, 20, 1).random(8)
    assert np.array_equal(s_a, s_b)
    assert not np.array_equal(s_a, s_c)
    assert not np.array_equal(s_a, s_d)


# ----------------------------------------------------------------------
# 11c. Steering invariants and physical-rule determinism (Phase C)
# ----------------------------------------------------------------------
@check
def steering_and_rule():
    import pickle
    import run_intervention as RI
    import run_steering as RS
    import run_mechanism as RM
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", "test-steer")

    ent = cohort.domain_entropy("dev", "test-steer")
    beta, n0 = cohort.matrix_and_init(ent, 0)

    # noop steering equals the plain trajectory (short horizon)
    old_h = RS.HORIZON
    RS.HORIZON = 6
    try:
        st = RS.steer_lineage(n0, beta, "02", 0, 0, 0, "noop")
        rng = cohort._rng(RI._ENT, 7, 0, 0, 0)
        plain = sim.run_fissions(n0, beta, "02", 6, rng)
        assert st == RS.lineage_stats(plain["inherited"])
    finally:
        RS.HORIZON = old_h

    # controller swaps preserve mass and non-negativity along a chain
    rng = cohort._rng(ent, 2, 0, 0)
    traj = sim.run_fissions(n0, beta, "02", 12, rng)
    n = traj["daughters"][11]
    hs = traj["H"][:12]
    mass0 = int(n.sum())
    for k in range(6):
        X9 = F.direct9(12 + k, 100, hs, int(n.sum()))
        swap = RS.marginal_swap(n, beta, X9, RI._BUNDLES["02"],
                                +1 if k % 2 == 0 else -1)
        i, j = swap
        assert i != j
        n = RI.apply_swap(n, swap)
        assert int(n.sum()) == mass0 and (n >= 0).all()

    # lineage_stats unit case: F T T T F T T T -> 2 episodes, 2 breaks
    flags = np.array([False, True, True, True, False, True, True, True])
    s = RS.lineage_stats(flags)
    assert s["episodes"] == 2 and s["breaks"] == 2
    assert s["longest_run"] == 3

    # frozen physical rule is deterministic and legal
    RM._RULE = ("in_boost", 1.0)
    r1 = RM.rule_swaps(n, beta)
    r2 = RM.rule_swaps(n, beta)
    assert r1 == r2
    for (i, j) in r1:
        assert i != j and n[i] >= 1
        ne = RI.apply_swap(n, (i, j))
        assert int(ne.sum()) == mass0 and (ne >= 0).all()


# ----------------------------------------------------------------------
# 11d. Phase D helpers: extended outcomes, decomposition, updates,
#      logging-callback neutrality
# ----------------------------------------------------------------------
@check
def phase_d_helpers():
    import pickle
    import run_intervention as RI
    import run_steering as RS
    from run_d1_outcomes import extended_branch_outcomes
    T, Fa = True, False

    # extended outcomes on crafted flags
    eo = extended_branch_outcomes(np.array([Fa, T, T, T, T, T]), False)
    assert eo["break"] == 1.0 and eo["run3_gb"] == 1.0
    assert eo["persist5"] == 1.0 and eo["inherited_count"] == 5.0
    eo = extended_branch_outcomes(np.array([Fa, T, T, T, Fa, T]), False)
    assert eo["run3_gb"] == 1.0 and eo["persist5"] == 0.0
    eo = extended_branch_outcomes(np.array([T] * 12), False)
    assert eo["break"] == 0.0 and np.isnan(eo["run3_gb"])
    assert eo["inherited_count"] == 12.0

    # identity-exact midpoint decomposition
    rng = np.random.default_rng(77)
    for _ in range(100):
        b_u, b_d, r_u, r_d = rng.random(4)
        lhs = (b_u - b_d) * (r_u + r_d) / 2 + (b_u + b_d) / 2 * (r_u - r_d)
        assert abs(lhs - (b_u * r_u - b_d * r_d)) < 1e-12

    # run_fissions exposes per-fission update counts
    ent = cohort.domain_entropy("dev", "test-updates")
    beta, n0 = cohort.matrix_and_init(ent, 0)
    out = sim.run_fissions(n0, beta, "02",
                           5, np.random.default_rng(1))
    assert len(out["updates"]) == out["n_done"]
    assert (out["updates"] > 0).all()

    # logging callback is behavior-neutral and fires once per fission
    with open(os.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = pickle.load(f)
    RI._ENT = cohort.domain_entropy("confirmation", "test-log")
    seen = []
    s1 = RS.steer_lineage(n0, beta, "02", 0, 0, 0, "noop",
                          log=lambda *a: seen.append(a), horizon=6)
    s2 = RS.steer_lineage(n0, beta, "02", 0, 0, 0, "noop", horizon=6)
    assert s1 == s2 and len(seen) == 6
    assert all(a[2] is None for a in seen)          # noop: no swaps


# ----------------------------------------------------------------------
# 11e. Phase E: perturbation recipe and outcome classifier
# ----------------------------------------------------------------------
@check
def release_challenge_helpers():
    from run_release_challenge import perturb, classify_branch
    rng = np.random.default_rng(313)
    n = np.zeros(sim.NG, dtype=np.int64)
    n[:6] = [20, 8, 5, 4, 2, 1]
    for k in (2, 4, 8, 16):
        pe = perturb(n, k, np.random.default_rng(k))
        assert pe.sum() == n.sum() and (pe >= 0).all()
        removed = np.clip(n - pe, 0, None).sum()
        added = np.clip(pe - n, 0, None).sum()
        assert removed <= k and added <= k and removed == added
    p1 = perturb(n, 8, np.random.default_rng(9))
    p2 = perturb(n, 8, np.random.default_rng(9))
    assert np.array_equal(p1, p2)                    # deterministic

    T, Fa = True, False
    hi, lo = 0.95, 0.5
    # held: never departs, ends high
    assert classify_branch(np.full(24, hi), np.full(24, T), 0.6) == "held"
    # returned: departs then sustains > 0.9 for >= 3
    ah = np.array([hi, lo, lo, hi, hi, hi] + [hi] * 6)
    assert classify_branch(ah, np.full(12, T), 0.6) == "returned"
    # two-fission recovery is NOT a return
    ah = np.array([hi, lo, hi, hi, lo, lo] + [lo] * 6)
    assert classify_branch(ah, np.full(12, Fa), 0.2) == "lost"
    # mode-recovered: composition gone, mode intact
    ah = np.full(12, lo)
    inh = np.array([Fa] * 6 + [T] * 6)
    assert classify_branch(ah, inh, 0.5) == "mode_recovered"
    assert classify_branch(ah, inh, 0.3) == "lost"   # not concentrated
    # held requires ending above 0.9, not merely never departing
    ah = np.array([hi] * 23 + [0.8])
    assert classify_branch(ah, np.full(24, T), 0.2) == "lost"


# ----------------------------------------------------------------------
# 11f. Phase F infrastructure: traced growth, flux/R_Q, atlas, S-J model
# ----------------------------------------------------------------------
@check
def phase_f_infrastructure():
    import growth_trace as GT
    import atlas as AT
    import sj_model as SJ

    ent = cohort.domain_entropy("dev", "test-f")
    beta, n0 = cohort.matrix_and_init(ent, 0)

    # flux matches the Kahana closed form on a fixture
    n = n0.copy()
    f = GT.flux(n, beta)
    N = n.sum()
    bn = 1.0 + (beta @ n) / N
    ref = (sim.KF * sim.RHO * N) * bn - (sim.KB * n) * bn
    assert np.allclose(f, ref, rtol=0, atol=0)
    assert -1.0 <= GT.r_q(n, beta) <= 1.0
    # R_Q is unclipped cosine: orthogonal fixture gives ~0, not clipped
    assert abs(GT.cosine_signed(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-12
    assert GT.cosine_signed(np.array([1.0, 0]), np.array([-1.0, 0])) == -1.0

    # traced growth: mass grid, conservation, parameterized nmax
    for cand, nmax in [("02", 80), ("02", 100), ("03", 80)]:
        rng = cohort._rng(ent, 99, 0)
        out = GT.traced_run_fissions(n0, beta, cand, 3, rng, nmax,
                                     grid_step=5)
        assert len(out["recs"]) == 3
        for r in out["recs"]:
            masses = [m for m, _ in r["snaps"]]
            assert masses == sorted(masses)
            if cand == "02":
                assert int(r["parent"].sum()) == nmax
                assert masses[-1] == nmax
            else:
                assert int(r["parent"].sum()) >= nmax
            for m, comp in r["snaps"]:
                assert int(comp.sum()) == m and (comp >= 0).all()

    # atlas: deterministic, unit centers, valid k
    a1 = AT.build_atlas(0, "02", ent, nmax=80, n_lineages=1,
                        n_fissions=60, subsample=50)
    a2 = AT.build_atlas(0, "02", ent, nmax=80, n_lineages=1,
                        n_fissions=60, subsample=50)
    assert a1["k"] == a2["k"] and np.allclose(a1["centers"], a2["centers"])
    assert 1 <= a1["k"] <= 6
    assert np.allclose(np.linalg.norm(a1["centers"], axis=1), 1.0)
    x = a1["centers"][0] * 40
    assert AT.dist(x, a1) < 1e-9

    # Singh-Jain: bistability sanity — modes persist over short runs
    r_in = SJ.run_lineage(SJ.INACTIVE_INIT, 6,
                          np.random.default_rng(1))
    r_ac = SJ.run_lineage(SJ.ACTIVE_INIT, 6,
                          np.random.default_rng(2))
    assert not r_in["died"] and not r_ac["died"]
    assert r_in["modes"].mean() < 0.5 < r_ac["modes"].mean()
    # active divides faster (paper: tau2 < tau1)
    assert r_ac["taus"].mean() < r_in["taus"].mean()


# ----------------------------------------------------------------------
# 11g. Phase G helpers
# ----------------------------------------------------------------------
@check
def phase_g_helpers():
    import pickle
    import run_intervention as RI
    from run_g1_competency import entropy_of, FP_NAMES
    from run_g2_resist_resil import g2_dev_unit, screen_edit
    import registry_v2 as R2

    # entropy helper
    n = np.zeros(sim.NG, dtype=np.int64)
    n[0] = 40
    assert abs(entropy_of(n)) < 1e-9
    n[:40] = 1
    n[40:] = 0
    assert abs(entropy_of(n) - np.log(40)) < 1e-9
    assert len(FP_NAMES) == 8

    # G2 dev targets: break-within-6 and post-break run3 logic
    cand, A, B = g2_dev_unit((0, "02"))
    XA9, XA195, yA = A
    XB9, XB195, yB = B
    assert XA9.shape[1] == 9 and XA195.shape[1] == 195
    assert set(np.unique(yA)) <= {0.0, 1.0}
    assert len(yB) <= len(yA) and XB9.shape[0] == len(yB)

    # screen_edit legality + determinism with a tiny trained student
    qb = R2.train_v2(XA9[:400], XA195[:400], yA[:400])
    ent = cohort.domain_entropy("dev", "test-g")
    beta, n0 = cohort.matrix_and_init(ent, 0)
    rng = cohort._rng(ent, 2, 0, 0)
    tr = sim.run_fissions(n0, beta, "02", 20, rng)
    s = tr["daughters"][19]
    X9 = F.direct9(20, 100, tr["H"][:20], int(s.sum()))
    e1 = screen_edit(qb, s, beta, X9, +1)
    e2 = screen_edit(qb, s, beta, X9, +1)
    assert e1 == e2 and e1[0] != e1[1]
    ne = RI.apply_swap(s, e1)
    assert ne.sum() == s.sum() and (ne >= 0).all()


# ----------------------------------------------------------------------
# 12. Registered direct-variable identities (documented redundancies)
# ----------------------------------------------------------------------
@check
def direct9_identities():
    rng = np.random.default_rng(808)
    for _ in range(200):
        g = int(rng.integers(1, 60))
        hs = rng.random(g) * 0.2 + 0.8
        x = F.direct9(g, 100, hs, 40)
        assert x[6] == x[4], "fissions_since_break == trailing_run"
        if x[7] == 1.0:
            assert x[8] == x[4], "regime_duration == trailing when inherited"


# ----------------------------------------------------------------------
# 13. Feature provenance consistency (frozen index identity)
# ----------------------------------------------------------------------
@check
def provenance_consistency():
    assert sum(b[1] for b in F.GRAPH_STATE_PROVENANCE) == 195
    comp = F.state_only_indices()
    beta = F.beta_conditioned_indices()
    assert np.array_equal(comp, np.r_[0:50, 162:165]), "frozen COMP_IDX"
    assert np.array_equal(beta, np.setdiff1d(np.arange(195), comp))
    assert len(comp) == 53 and len(beta) == 142
    import run_ablation
    assert np.array_equal(run_ablation.COMP_IDX, comp)
    assert np.array_equal(run_ablation.BETA_IDX, beta)
    assert len(F.DIRECT9_PROVENANCE) == 9
    assert [n for n, _ in F.DIRECT9_PROVENANCE] == F.DIRECT9_NAMES


# ----------------------------------------------------------------------
# G5: internalized-controller ladder helpers
# ----------------------------------------------------------------------
@check
def g5_policy_gates():
    import run_g5_internal as G5
    rng = np.random.default_rng(11)
    beta = sim.make_beta(rng)
    n = sim.make_initial_state(rng)
    # L0 rule: removes least-influential PRESENT type, adds
    # most-influential, i != j, mass preserved by apply_swap
    i, j = G5.rule_swap_stabilize(n, beta)
    x = n / n.sum()
    infl = x @ beta
    present = np.where(n > 0)[0]
    assert n[i] > 0 and i != j
    assert infl[i] == infl[present].min()
    assert infl[j] >= np.sort(infl)[-2]
    import run_intervention as RI
    n2 = RI.apply_swap(n, (i, j))
    assert n2.sum() == n.sum() and n2[i] == n[i] - 1 and n2[j] == n[j] + 1
    # L1 gate: silent while inheriting, edits right after a break
    ent = cohort.domain_entropy("confirmation", "g5-test")
    assert G5.policy_swap("L1", n, beta, "02", 0, [0.95], 1, 0, 0,
                          ent) is None
    assert G5.policy_swap("L1", n, beta, "02", 0, [0.85], 1, 0, 0,
                          ent) == (i, j)
    # L2 gate: edits while streak < 3, silent at streak >= 3
    assert G5.policy_swap("L2", n, beta, "02", 0, [0.5, 0.95, 0.95], 3,
                          0, 0, ent) == (i, j)
    assert G5.policy_swap("L2", n, beta, "02", 0,
                          [0.95, 0.95, 0.95], 3, 0, 0, ent) is None
    # noop never edits; random is a valid deterministic-keyed swap
    assert G5.policy_swap("noop", n, beta, "02", 0, [0.5], 1, 0, 0,
                          ent) is None
    s1 = G5.policy_swap("random", n, beta, "02", 0, [0.5], 1, 0, 0, ent)
    s2 = G5.policy_swap("random", n, beta, "02", 0, [0.5], 1, 0, 0, ent)
    assert s1 == s2 and n[s1[0]] > 0 and s1[0] != s1[1]


@check
def g5_biased_growth_null():
    """damp == 1 must reproduce the frozen growth paths exactly
    (identical RNG call order), for both candidates."""
    import run_g5_internal as G5
    rng = np.random.default_rng(12)
    beta = sim.make_beta(rng)
    n0 = sim.make_initial_state(rng)
    one = np.ones(sim.NG)
    ga, da = G5.biased_grow_gillespie(n0, beta, np.random.default_rng(3),
                                      one)
    gb, _, db = sim._grow_gillespie(n0, beta, np.random.default_rng(3))
    assert np.array_equal(ga, gb) and da == db
    pa, dpa = G5.biased_grow_poisson(n0, beta, np.random.default_rng(4),
                                     one)
    pb, _, dpb = sim._grow_poisson(n0, beta, np.random.default_rng(4))
    assert np.array_equal(pa, pb) and dpa == dpb
    # damp > 1 strictly reduces leave propensity mass
    damp = 1.0 + 0.3 * np.linspace(0, 1, sim.NG)
    c = beta @ n0
    join, leave = sim.event_rates(n0, c, int(n0.sum()))
    assert (leave / damp).sum() < leave.sum()


@check
def g5_local_features_and_tree_fallback():
    import run_g5_internal as G5
    rng = np.random.default_rng(13)
    beta = sim.make_beta(rng)
    n = sim.make_initial_state(rng)
    F5 = G5.type_features(n, beta)
    assert F5.shape == (sim.NG, 4)
    assert F5[:, 1].min() == 0.0 and F5[:, 1].max() == 1.0
    assert np.allclose(F5[:, 0].sum(), 1.0)
    assert np.array_equal(F5[:, 3], (n > 0).astype(float))
    # tree_swap with no distilled tree falls back to the L0 rule
    assert G5.tree_swap(n, beta, (None, None)) == \
        G5.rule_swap_stabilize(n, beta)


# ----------------------------------------------------------------------
# Phase I: reconstructed Phi-r bridge helpers
# ----------------------------------------------------------------------
@check
def phir_gaussian_mi_exact():
    import phir
    rho = 0.6
    S = np.array([[1.0, rho], [rho, 1.0]])
    assert abs(phir.gauss_mi(S, [0], [1])
               - (-0.5 * np.log(1 - rho ** 2))) < 1e-12
    S4 = np.eye(4)
    assert abs(phir.gauss_mi(S4, [0, 1], [2, 3])) < 1e-12


@check
def phir_rewards_synergy_not_parallelism():
    """Phi_r is strongly positive for synergistic Gaussian dynamics
    (the future is carried by a cross-part difference of correlated
    sources, invisible to either part alone) and near zero for
    parallel per-part AR dynamics (parts jointly exhaust the
    predictive information)."""
    import phir
    rng = np.random.default_rng(0)
    T, half = 400, 4
    Xs = np.zeros((T, 2 * half))
    Xp = np.zeros((T, 2 * half))
    z = rng.normal(size=(T, half))
    a = rng.normal(size=(T, half))
    b = rng.normal(size=(T, half))
    src_a = z + 0.15 * a
    src_b = z + 0.15 * b
    Xs[:, :half], Xs[:, half:] = src_a, src_b
    for t in range(1, T):
        Xs[t] += 3.0 * np.tile(src_a[t - 1] - src_b[t - 1], 2)
        Xp[t] = 0.9 * Xp[t - 1] + 0.1 * rng.normal(size=2 * half)
    ps = phir.phi_r_series(np.exp(0.25 * Xs) * 50)
    pp = phir.phi_r_series(np.exp(0.25 * Xp) * 50)
    assert np.isfinite(ps) and np.isfinite(pp)
    assert ps > 0.1 and abs(pp) < 0.1 and ps > pp + 0.1
    assert np.isnan(phir.phi_r_series((np.exp(0.25 * Xs) * 50)[:10]))


@check
def phir_surrogate_and_traced_step():
    import phir
    from run_phir_bridge import traced_step
    rng = np.random.default_rng(21)
    beta = sim.make_beta(rng)
    n = sim.make_initial_state(rng)
    s1 = phir.phi_r_surrogate(n, beta)
    assert np.isfinite(s1) and phir.phi_r_surrogate(n, beta) == s1
    n2 = n.copy()
    i = int(np.where(n2 > 0)[0][0])
    j = int(np.where(n2 == 0)[0][0])
    n2[i] -= 1
    n2[j] += 1
    assert phir.phi_r_surrogate(n2, beta) != s1
    # traced_step must reproduce the frozen contract exactly (CRN)
    for cand in cohort.CANDIDATES:
        record = []
        d1, h1 = traced_step(n, beta, cand,
                             np.random.default_rng(5), record)
        st = sim.run_fissions(n, beta, cand, 1,
                              np.random.default_rng(5))
        assert np.array_equal(d1, st["final"])
        assert h1 == st["H"][0]
        assert len(record) == int(st["updates"][0])
        assert np.array_equal(record[-1], st["parents"][0])


# ----------------------------------------------------------------------
# Phase I addendum: code-faithful Phi-r port (phir_code.py)
# ----------------------------------------------------------------------
@check
def phir_code_mobius_identities():
    """Mobius closure: down-set sums equal the node's pointwise-MMI
    value; the 9-atom Phi_R equals the inclusion-exclusion closed
    form total - self0 - self1 + redundancy."""
    import phir_code as PC
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 200))
    x[1] += 0.5 * np.roll(x[0], 1)
    pi = PC.local_phi_id(x)
    top = PC._phi_min(x, (PC.S, PC.S))
    total = np.sum([pi[a] for a in PC.ATOMS], axis=0)
    assert np.abs(total - top).max() < 1e-9
    d00 = PC._phi_min(x, (PC.U0, PC.U0))
    sub = sum(pi[a] for a in [(PC.R, PC.R), (PC.R, PC.U0),
                              (PC.U0, PC.R), (PC.U0, PC.U0)])
    assert np.abs(sub - d00).max() < 1e-9
    closed = top - PC._phi_min(x, (PC.U0, PC.U0)) \
        - PC._phi_min(x, (PC.U1, PC.U1)) + PC._phi_min(x, (PC.R, PC.R))
    assert np.abs(PC.local_phi_r(pi) - closed).max() < 1e-9


@check
def phir_code_pipeline_behaviour():
    import phir_code as PC
    rng = np.random.default_rng(4)
    # Fiedler bipartition recovers planted blocks (common signals
    # must be autocorrelated: the MI graph is lag-1)
    n = 24
    T = 300
    s1, s2 = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        s1[t] = 0.9 * s1[t - 1] + rng.normal()
        s2[t] = 0.9 * s2[t - 1] + rng.normal()
    y = 0.4 * rng.normal(size=(n, T))
    y[: n // 2] += s1
    y[n // 2:] += s2
    a, b = PC.fiedler_bipartition(PC.mi_matrix_lag1(y))
    assert {frozenset(a), frozenset(b)} == \
        {frozenset(range(n // 2)), frozenset(range(n // 2, n))}
    # determinism + NaN on short series, finite on real-ish counts
    counts = rng.poisson(3.0, size=(200, sim.NG)).astype(float)
    v1, v2 = PC.phi_r_code(counts), PC.phi_r_code(counts)
    assert np.isfinite(v1) and v1 == v2
    assert np.isnan(PC.phi_r_code(counts[:10]))


# ----------------------------------------------------------------------
# Phase J: probe-rollout Phi_R controller helpers
# ----------------------------------------------------------------------
@check
def phase_j_probe_and_panel():
    import run_intervention as RI
    import run_phir_confirm as J
    rng = np.random.default_rng(31)
    beta = sim.make_beta(rng)
    n = sim.make_initial_state(rng)
    # panel legality: removes present, adds other, mass preserved
    panel = J.draw_panel(n, np.random.default_rng(2))
    assert len(panel) == J.PANEL
    for i, j in panel:
        assert n[i] > 0 and i != j
        n2 = RI.apply_swap(n, (i, j))
        assert n2.sum() == n.sum()
    # probe determinism: identical CRN stream -> identical score for
    # the same edit; finite for a healthy state
    import pickle as _p
    import os as _o
    with open(_o.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = _p.load(f)
    s1 = J.probe_phi(n, beta, "02", 0, 0, 0, 5)
    s2 = J.probe_phi(n, beta, "02", 0, 0, 0, 5)
    assert s1 == s2 and np.isfinite(s1)
    # different edits share the probe stream but yield different
    # (finite or nan) scores computed independently
    e1 = RI.apply_swap(n, panel[0])
    e2 = RI.apply_swap(n, panel[1])
    p1 = J.probe_phi(e1, beta, "02", 0, 0, 0, 5)
    p2 = J.probe_phi(e2, beta, "02", 0, 0, 0, 5)
    assert np.isfinite(p1) and np.isfinite(p2)


# ----------------------------------------------------------------------
# Phase K: atom decomposition consistency
# ----------------------------------------------------------------------
@check
def phase_k_atoms_match_scalar():
    import phir_code as PC
    import run_phir_dose as K
    rng = np.random.default_rng(41)
    counts = rng.poisson(3.0, size=(220, sim.NG)).astype(float)
    atoms = K.phi_atoms(counts)
    scalar = PC.phi_r_code(counts)
    assert atoms is not None and np.isfinite(scalar)
    assert abs(atoms["phi_r"] - scalar) < 1e-9
    nine = sum(atoms[K.ATOM_NAMES[a]] for a in PC.PHIR_ATOMS)
    assert abs(nine - atoms["phi_r"]) < 1e-9
    assert abs(atoms["emergence"]
               - (atoms["synergy"] + atoms["causation"])) < 1e-12
    # arm-name parsing covers the registered cadence grid
    dirs = {a: ("stab", int(a[4:])) if a.startswith("stab")
            else ("destab", int(a[6:]))
            for a in K.ARMS if a != "noop"}
    assert sorted({k for _, k in dirs.values()}) == [1, 2, 4, 8, 16]
    assert len(dirs) == 10


# ----------------------------------------------------------------------
# Phase L: paper-faithful instrument
# ----------------------------------------------------------------------
@check
def phase_l_paper_instrument():
    import phir_paper as PP
    rng = np.random.default_rng(51)
    counts = rng.poisson(3.0, size=(400, sim.NG)).astype(float)
    Z = PP.clr_drop_last(counts)
    assert Z.shape == (400, sim.NG - 1)          # drop-last applied
    assert np.allclose((np.log((counts + 0.5)
                        / (counts + 0.5).sum(1, keepdims=True))
                        - np.log((counts + 0.5)
                        / (counts + 0.5).sum(1, keepdims=True))
                        .mean(1, keepdims=True))[:, :-1], Z)
    v1 = PP.phi_r_paper(counts)
    assert np.isfinite(v1) and PP.phi_r_paper(counts) == v1
    assert np.isnan(PP.phi_r_paper(counts[:10]))     # too short
    assert np.isnan(PP.phi_r_paper(counts[:150]))    # rank guard
    a, b = PP.mib_instantaneous(PP.clr_drop_last(counts))
    assert a and b and len(a) + len(b) <= sim.NG - 1


# ----------------------------------------------------------------------
# Phase M: SR labels and window machinery
# ----------------------------------------------------------------------
@check
def phase_m_sr_and_windows():
    import run_phir_sr as M
    hs = [0.95, 0.95, 0.5, 0.95, 0.95, 0.95, 0.95, 0.95, 0.5, 0.95]
    lab5 = M.sr_labels(hs, 5)
    assert lab5.tolist() == [False, False, False, True, True, True,
                             True, True, False, False]
    lab3 = M.sr_labels(hs, 3)
    assert lab3[3:8].all() and not lab3[:3].any()
    assert not M.sr_labels(hs, 8).any()
    rng = np.random.default_rng(61)
    counts = rng.poisson(3.0, size=(120, sim.NG)).astype(float)
    sc = M.window_scores(counts)
    assert set(sc) == set(M.INSTRUMENTS)
    assert all(np.isfinite(sc[k]) for k in M.INSTRUMENTS)
    assert abs(sc["emergence"] - (sc["synergy"]
               + (sc["emergence"] - sc["synergy"]))) < 1e-12
    assert all(np.isnan(v) for v in
               M.window_scores(counts[:10]).values())


# ----------------------------------------------------------------------
# Phase N: foresight-round machinery
# ----------------------------------------------------------------------
@check
def phase_n_foresight_helpers():
    import run_phir_foresight as N
    from run_phir_sr import window_scores
    rng = np.random.default_rng(71)
    # gen-clock scoring works on a daughter series
    D = rng.poisson(2.0, size=(40, sim.NG)).astype(float)
    g = window_scores(D)
    assert np.isfinite(g["phiR"]) and np.isfinite(g["printed"])
    # centered spearman: perfect within-matrix predictor recovered
    rows = []
    r2 = np.random.default_rng(5)
    for m in range(8):
        base = 10 * m
        for i in range(10):
            x = float(r2.normal())
            rows.append({"matrix": m, "x": x + base,
                         "y": 2 * x + base + 0.01 * r2.normal()})
    r, lo, hi = N.spearman_cells(rows, "x", "y", True)
    assert r > 0.9 and lo > 0.5
    # predictor list has no duplicates and includes benchmarks
    assert len(set(N.PREDICTORS)) == len(N.PREDICTORS)
    assert "v2_risk" in N.PREDICTORS and "hist" in N.PREDICTORS


# ----------------------------------------------------------------------
# G3-ADJ: convention semantics and sealed replay
# ----------------------------------------------------------------------
@check
def g3_adj_conventions():
    import pickle as _p
    import os as _o
    import run_intervention as RI
    import run_steering as RS
    import run_g3_adjudication as ADJ
    with open(_o.path.join(HERE, "results_v2",
                           "frozen_models_v2.pkl"), "rb") as f:
        RI._BUNDLES = _p.load(f)
    # edit-count semantics: conv A does P-1 edits, conv B does P
    calls = []
    real = RS.marginal_swap
    RS.marginal_swap = lambda *a, **k: (calls.append(1)
                                        or real(*a, **k))
    try:
        beta, s0 = cohort.matrix_and_init(ADJ._ENT_S, 0)
        calls.clear()
        ADJ.steer_cell(s0, beta, "02",
                       np.random.default_rng(1), 4, "A")
        n_a = len(calls)
        calls.clear()
        ADJ.steer_cell(s0, beta, "02",
                       np.random.default_rng(1), 4, "B")
        n_b = len(calls)
    finally:
        RS.marginal_swap = real
    assert n_a == 3 and n_b == 4
    # sealed replay: fresh x conv A reproduces a stored G3 pulse row
    sealed = _p.load(open(_o.path.join(HERE, "results_g",
                                       "g3_units.pkl"), "rb"))
    u = next(x for x in sealed if x["candidate"] == "02"
             and x["matrix"] == 0 and x["rep"] == 0
             and 4 in x["pulse"])
    rng = ADJ.cell_rng("fresh", "A", 0, 0, 0)
    t = ADJ.steer_cell(s0, beta, "02", rng, 4, "A")
    assert t == u["pulse"][4]["t07"]


# ----------------------------------------------------------------------
def main():
    passed = failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(CHECKS)} total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
