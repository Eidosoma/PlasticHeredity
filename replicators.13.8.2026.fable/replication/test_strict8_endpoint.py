"""Phase H registered endpoint fixtures (STRICT8 preregistration §
"Registered endpoint fixtures"). All must pass before the module is
sealed. Deterministic; no scientific matrices are touched."""

import sys
import traceback

import numpy as np

import sim
import cohort
import run_strict8_occurrence_replication as S8

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


NG = sim.NG
V = np.zeros(NG); V[:40] = 1.0                 # episode composition
Q = np.zeros(NG); Q[40:80] = 1.0               # orthogonal helper
P = np.zeros(NG); P[90:100] = 4.0              # distinct old anchor


def synth(hs, p_old=P, daughter=V, overrides=None):
    k = len(hs)
    parents = np.tile(p_old, (k, 1))
    daughters = np.tile(daughter, (k, 1))
    for idx, vec in (overrides or {}).items():
        daughters[idx - 1] = vec               # 1-indexed boundary
    return np.array(hs, float), parents, daughters


# 1 ---------------------------------------------------------------
@check
def no_break_negative():
    r = S8.classify_future(*synth([0.95] * 20))
    assert not r["break_within"] and not r["positive"]


# 2 ---------------------------------------------------------------
@check
def seven_inherited_negative():
    r = S8.classify_future(*synth([0.5] + [0.95] * 7))
    assert r["break_within"] and not r["run8_after"] \
        and not r["positive"]


# 3 ---------------------------------------------------------------
@check
def coherence_failure_negative():
    w = Q.copy()                               # cos(V, w) = 0 <= 0.9
    r = S8.classify_future(*synth([0.5] + [0.95] * 8,
                                  overrides={5: w}))
    assert r["run8_after"] and r["coh_first"] is False \
        and r["dis_first"] is True and not r["positive"]


# 4 ---------------------------------------------------------------
@check
def distinctness_failure_negative():
    vn = V / np.linalg.norm(V)
    qn = Q / np.linalg.norm(Q)
    u = 0.88 * vn + np.sqrt(1 - 0.88 ** 2) * qn   # cos(V,u)=0.88
    c = float(np.dot(vn, u) / np.linalg.norm(u))
    assert 0.85 < c <= 0.9
    r = S8.classify_future(*synth([0.5] + [0.95] * 8, p_old=u))
    assert r["coh_first"] is True and r["dis_first"] is False \
        and not r["positive"]


# 5 ---------------------------------------------------------------
@check
def valid_event_positive():
    r = S8.classify_future(*synth([0.5] + [0.95] * 8))
    assert r["positive"] and r["primary_r"] == 2 and r["cert"] == 9
    assert r["eligible"] and r["eligible"][0][0] == 2


# 6 ---------------------------------------------------------------
@check
def prebreak_run_does_not_count():
    r = S8.classify_future(*synth([0.95] * 8 + [0.5] + [0.95] * 3))
    assert r["break_within"] and not r["positive"]
    r2 = S8.classify_future(*synth([0.95] * 8 + [0.5] + [0.95] * 8))
    assert r2["positive"] and r2["primary_r"] == 10 and r2["cert"] == 17


# 7 ---------------------------------------------------------------
@check
def interrupted_then_valid_positive():
    r = S8.classify_future(*synth([0.5] + [0.95] * 5 + [0.5]
                                  + [0.95] * 8))
    assert r["positive"] and r["primary_r"] == 8 and r["cert"] == 15
    # overlapping eligible windows are all persisted, first = primary
    r3 = S8.classify_future(*synth([0.5] + [0.95] * 10))
    assert [w[0] for w in r3["eligible"]] == [2, 3, 4]
    assert r3["positive"] and r3["primary_r"] == 2 \
        and r3["run_len"] == 10


# 8 ---------------------------------------------------------------
@check
def certification_at_32_positive():
    hs = [0.95] * 23 + [0.5] + [0.95] * 8      # break 24, run 25..32
    r = S8.classify_future(*synth(hs))
    assert len(hs) == 32 and r["positive"] and r["cert"] == 32


# 9 ---------------------------------------------------------------
@check
def certification_after_32_negative():
    hs = [0.95] * 24 + [0.5] + [0.95] * 8      # cert would be 33
    r = S8.classify_future(*synth(hs))
    assert r["break_within"] and not r["positive"]


# 10 --------------------------------------------------------------
@check
def exact_threshold_semantics():
    assert S8.is_inherited(0.9) is False        # H = 0.9 is a break
    assert S8.is_inherited(0.9000000001) is True
    assert S8.pair_coherent(0.9) is False       # pairwise 0.9 fails
    assert S8.anchor_distinct(0.85) is True     # anchor 0.85 passes
    assert S8.anchor_distinct(0.8500000001) is False
    r = S8.classify_future(*synth([0.9] + [0.95] * 8))
    assert r["break_within"] and r["first_break"] == 1 \
        and r["positive"]


# 11 --------------------------------------------------------------
@check
def permutation_invariance():
    hs, parents, daughters = synth([0.5] + [0.95] * 8)
    perm = np.random.default_rng(5).permutation(NG)
    a = S8.classify_future(hs, parents, daughters)
    b = S8.classify_future(hs, parents[:, perm], daughters[:, perm])
    for k in ("positive", "primary_r", "cert", "first_break",
              "run_len"):
        assert a[k] == b[k]
    assert abs(a["minpair_first"] - b["minpair_first"]) < 1e-12
    assert abs(a["maxanchor_first"] - b["maxanchor_first"]) < 1e-12


# 12 --------------------------------------------------------------
@check
def candidate_daughter_semantics():
    rng = np.random.default_rng(7)
    beta = sim.make_beta(rng)
    n0 = sim.make_initial_state(rng)
    for cand in cohort.CANDIDATES:
        br = sim.run_fissions(n0, beta, cand, 5,
                              np.random.default_rng(9))
        for i in range(br["n_done"]):
            h = sim.cosine_h(br["parents"][i].astype(float),
                             br["daughters"][i].astype(float))
            assert h == br["H"][i]
            if cand == "02":               # equal split, first daughter
                assert br["daughters"][i].sum() \
                    == br["parents"][i].sum() // 2


# 13 --------------------------------------------------------------
@check
def exact_replay_from_branch_seed():
    ent = cohort.domain_entropy("strict8-smoke", S8.TAG)
    beta = sim.make_beta(cohort._rng(ent, S8.DOMAIN, 0, 0))
    n0 = sim.make_initial_state(cohort._rng(ent, S8.DOMAIN, 1, 0))
    outs = []
    for _ in range(2):
        rb = cohort._rng(ent, S8.DOMAIN, 3, 0, 0, 20, 5)
        br = sim.run_fissions(n0, beta, "02", S8.HORIZON, rb)
        outs.append(S8.classify_future(br["H"], br["parents"],
                                       br["daughters"]))
        outs.append(br["H"].copy())
    assert np.array_equal(outs[1], outs[3])
    assert outs[0] == outs[2]


# 14 --------------------------------------------------------------
@check
def extinction_semantics():
    # certified at 9, trace then ends (extinction after) -> positive
    r = S8.classify_future(*synth([0.5] + [0.95] * 8))
    assert r["positive"] and r["k"] == 9
    # extinction before certification -> negative
    r2 = S8.classify_future(*synth([0.5] + [0.95] * 6))
    assert not r2["positive"] and r2["k"] == 7


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
