"""Feature blocks, the JOINT_BREAK_RUN3 target, and the seven process
outcomes.

Blocks:
  direct9      : nine direct history/phase variables (paper-enumerated)
  graph_state  : 195 molecule-label-permutation-invariant coordinates of
                 current composition and catalytic-network-conditioned
                 state (registered reconstruction inventory; the source
                 paper does not enumerate its 195 coordinates)
  beta_only    : matrix-level permutation-invariant features of beta alone

Target (fixed prospectively, F12 horizon):
  JOINT_BREAK_RUN3: within the next 12 fissions, an inheritance break
  (H <= 0.9) occurs at some fission t, and a run of three consecutive
  inherited fissions (H > 0.9) starts strictly after t and completes
  within the horizon.
"""

from __future__ import annotations

import numpy as np

from sim import NG, KF, KB, RHO, H_THRESH, cosine_h

EPS = 1e-12

# ----------------------------------------------------------------------
# Feature provenance (typed metadata, per reviewer #3's recommendation)
#
# Ordered blocks of graph_state_195's concatenation. Flags:
#   state - depends on the current composition n
#   beta  - depends on the catalytic matrix beta
# No graph/state coordinate depends on history or growth clocks; the
# direct block (DIRECT9_PROVENANCE) carries history/phase explicitly.
# ----------------------------------------------------------------------

GRAPH_STATE_PROVENANCE = [
    # (block_name, length, depends_on_state, depends_on_beta)
    ("sorted_rel_composition_top40", 40, True, False),
    ("composition_scalars", 10, True, False),
    ("boost_quantiles", 7, True, True),
    ("boost_mean_std", 2, True, True),
    ("boost_xweighted", 4, True, True),
    ("join_distribution", 8, True, True),
    ("leave_distribution", 2, True, True),
    ("self_coupling", 6, True, True),
    ("two_step_propagation", 5, True, True),
    ("subnetwork_eigenvalues", 8, True, True),
    ("subnetwork_traces", 2, True, True),
    ("sorted_xb_profile_top20", 20, True, True),
    ("sorted_boost_present_top20", 20, True, True),
    ("pairwise_logbeta_present", 6, True, True),
    ("eigvec_alignment", 2, True, True),
    ("sorted_join_distribution_top20", 20, True, True),
    ("split_stability_proxies", 3, True, False),
    ("total_rate_scalars", 3, True, True),
    ("sorted_logboost_top27", 27, True, True),
]
assert sum(b[1] for b in GRAPH_STATE_PROVENANCE) == 195

DIRECT9_PROVENANCE = [
    # (name, kind) - kind in {"phase", "state", "history"}
    ("norm_generation", "phase"),
    ("current_mass", "state"),
    ("prefix_inherit_frac", "history"),
    ("recent5_inherit_frac", "history"),
    ("trailing_inherit_run", "history"),
    ("latest_H", "history"),
    ("fissions_since_break", "history"),   # == trailing_inherit_run (exact)
    ("current_inherit_state", "history"),
    ("regime_duration", "history"),        # == trailing run when inherited
]

# beta_only(): every coordinate depends on beta alone (no state/history).


def _graph_state_indices(depends_on_beta: bool):
    idx, start = [], 0
    for _, length, _, uses_beta in GRAPH_STATE_PROVENANCE:
        if uses_beta == depends_on_beta:
            idx.extend(range(start, start + length))
        start += length
    return np.array(idx, dtype=int)


def state_only_indices() -> np.ndarray:
    """Graph/state coordinates depending on composition alone."""
    return _graph_state_indices(False)


def beta_conditioned_indices() -> np.ndarray:
    """Graph/state coordinates depending on composition AND beta."""
    return _graph_state_indices(True)


# ----------------------------------------------------------------------
# Target and process outcomes
# ----------------------------------------------------------------------

def joint_break_run3(inh: np.ndarray) -> bool:
    """inh: boolean inheritance flags for the 12 branch fissions."""
    n = len(inh)
    # Only the FIRST break needs checking: any later break followed by a
    # 3-run implies the first break is followed by that same 3-run.
    for t in range(n):
        if not inh[t]:
            for u in range(t + 1, n - 2):
                if inh[u] and inh[u + 1] and inh[u + 2]:
                    return True
            return False  # first break found, no subsequent run of 3
    return False


def process_outcomes(inh: np.ndarray, daughters: np.ndarray,
                     restored: np.ndarray) -> dict:
    """Seven process outcomes for one branch (12 fissions).

    Registered operationalizations:
      break      : any fission with H <= 0.9 (unconditional)
      resume2    : given a break, >=2 consecutive inherited fissions occur
                   after the first break
      episode3   : given a break, >=3 consecutive inherited fissions occur
                   after the first break
      persist5   : given a break, >=5 consecutive inherited fissions occur
                   after the first break
      old_return : composition departs the old anchor neighbourhood
                   (cos < 0.7 vs anchor) after the break and later
                   returns (cos > 0.9) within the horizon (unconditional
                   prevalence)
      pos_gain   : given a break with a subsequent 3-episode, the anchor
                   similarity at the start of the new episode exceeds the
                   anchor similarity immediately after the break
                   (gain > 0); `gain` itself is also returned
      repeat     : two disjoint break->3-episode cycles within the horizon
                   (unconditional)

    anchor = parent composition heritage proxy: the post-fission
    composition at the last inherited fission before the first break
    (the restored state itself if the first fission already breaks).
    """
    n = len(inh)
    out = {k: np.nan for k in
           ["break", "resume2", "episode3", "persist5",
            "old_return", "pos_gain", "gain", "repeat"]}
    brk = bool((~inh).any()) if n > 0 else False
    out["break"] = float(brk)
    out["old_return"] = 0.0
    out["repeat"] = 0.0
    if not brk:
        return out
    t = int(np.argmin(inh))          # first break index
    anchor = restored if t == 0 else daughters[t - 1]
    anchor = anchor.astype(float)

    def max_run_after(a, flags):
        best = cur = 0
        for v in flags[a:]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    mr = max_run_after(t + 1, inh)
    out["resume2"] = float(mr >= 2)
    out["episode3"] = float(mr >= 3)
    out["persist5"] = float(mr >= 5)

    # anchor similarity trace after the break. Registered semantics:
    # sims[0] is the breaking fission's own daughter and is eligible as
    # the gain reference below but NOT as a departure point; departures
    # (cos < 0.7) and returns (cos > 0.9) are counted from fission t+1
    # onward (sims[1:]).
    sims = np.array([cosine_h(daughters[k].astype(float), anchor)
                     for k in range(t, n)])
    departed = False
    for s in sims[1:]:
        if s < 0.7:
            departed = True
        elif departed and s > H_THRESH:
            out["old_return"] = 1.0
            break

    # gain toward the old anchor at the start of the new episode
    if mr >= 3:
        u = None
        run = 0
        for k in range(t + 1, n):
            run = run + 1 if inh[k] else 0
            if run == 3:
                u = k - 2
                break
        if u is not None:
            # sims[0] = anchor similarity of the immediate post-break
            # composition (the daughter produced by the breaking fission)
            out["gain"] = float(
                cosine_h(daughters[u].astype(float), anchor) - sims[0])
            out["pos_gain"] = float(out["gain"] > 0)
        # repeated cycle: another break then episode3 after the first episode
        if u is not None and u + 3 < n:
            tail = inh[u + 3:]
            if (~tail).any():
                t2 = int(np.argmin(tail))
                out["repeat"] = float(max_run_after(t2 + 1, tail) >= 3)
    return out


def coherence_outcomes(inh: np.ndarray, daughters: np.ndarray,
                       restored: np.ndarray) -> dict:
    """Registered coherence/distinctness indicators (reviewer #7).

    For the FIRST certified 3-run (start u) after the first break:
      span_sim   = H(d_u, d_{u+2})   (episode first vs last daughter)
      anchor_sim = H(d_u, anchor)    (anchor as in process_outcomes)
      coherent   = joint event AND span_sim > 0.9
      distinct   = coherent AND anchor_sim < 0.9
    `joint` here equals joint_break_run3 by construction.
    """
    n = len(inh)
    out = {"joint": False, "span_sim": np.nan, "anchor_sim": np.nan,
           "coherent": False, "distinct": False}
    if n == 0 or bool(inh.all()):
        return out
    t = int(np.argmin(inh))
    run, u = 0, None
    for k in range(t + 1, n):
        run = run + 1 if inh[k] else 0
        if run == 3:
            u = k - 2
            break
    if u is None:
        return out
    out["joint"] = True
    anchor = (restored if t == 0 else daughters[t - 1]).astype(float)
    du = daughters[u].astype(float)
    out["span_sim"] = cosine_h(du, daughters[u + 2].astype(float))
    out["anchor_sim"] = cosine_h(du, anchor)
    out["coherent"] = bool(out["span_sim"] > H_THRESH)
    out["distinct"] = bool(out["coherent"]
                           and out["anchor_sim"] < H_THRESH)
    return out


# ----------------------------------------------------------------------
# Direct history/phase variables (9)
# ----------------------------------------------------------------------

DIRECT9_NAMES = [
    "norm_generation", "current_mass", "prefix_inherit_frac",
    "recent5_inherit_frac", "trailing_inherit_run", "latest_H",
    "fissions_since_break", "current_inherit_state", "regime_duration",
]


def direct9(g: int, n_total_gen: int, hs: np.ndarray, mass: int) -> np.ndarray:
    """History/phase variables at the post-fission state after fission g
    (1-based); hs = parent->daughter H for fissions 1..g."""
    inh = hs > H_THRESH
    trailing = 0
    for v in inh[::-1]:
        if v:
            trailing += 1
        else:
            break
    breaks = np.where(~inh)[0]
    since_break = g - (breaks[-1] + 1) if len(breaks) else g
    cur = bool(inh[-1])
    dur = 0
    for v in inh[::-1]:
        if v == cur:
            dur += 1
        else:
            break
    return np.array([
        g / n_total_gen,
        mass,
        float(inh.mean()),
        float(inh[-5:].mean()),
        float(trailing),
        float(hs[-1]),
        float(since_break),
        float(cur),
        float(dur),
    ])


# ----------------------------------------------------------------------
# 195-dim graph/state block
# ----------------------------------------------------------------------

def _quantiles(v, qs):
    return np.quantile(v, qs) if len(v) else np.zeros(len(qs))


def graph_state_195(n: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Permutation-invariant encoding of current composition and
    catalytic-network-conditioned state. Exactly 195 coordinates."""
    n = n.astype(float)
    N = n.sum()
    x = n / max(N, 1.0)
    present = n > 0
    xp = x[present]
    feats = []

    # 1. sorted relative composition, top 40                      (40)
    sx = np.sort(x)[::-1]
    feats.append(sx[:40])

    # 2. composition scalars                                      (10)
    ent = -np.sum(xp * np.log(xp + EPS))
    simpson = np.sum(x * x)
    feats.append(np.array([
        N, present.sum() / NG, (n >= 2).sum() / NG, (n >= 3).sum() / NG,
        ent, simpson, sx[0], sx[:3].sum(), sx[:5].sum(), sx[:10].sum(),
    ]))

    # boost vector and rate structure
    b = (beta @ n) / max(N, 1.0)
    bn = 1.0 + b
    lb = np.log1p(b)

    # 3. boost statistics                                         (13)
    feats.append(_quantiles(lb, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    feats.append(np.array([lb.mean(), lb.std()]))
    lbp = lb[present]
    feats.append(np.array([
        float(np.sum(x * lb)),                       # x-weighted mean
        float(np.sqrt(max(np.sum(x * lb ** 2) - np.sum(x * lb) ** 2, 0.0))),
        float(lbp.min()) if len(lbp) else 0.0,
        float(lbp.max()) if len(lbp) else 0.0,
    ]))

    # 4. expected join distribution                               (8)
    p = RHO * N * bn
    p = p / p.sum()
    ent_p = -np.sum(p * np.log(p + EPS))
    feats.append(np.array([
        ent_p, np.max(p), np.sort(p)[::-1][:5].sum(),
        np.sort(p)[::-1][:10].sum(), np.sort(p)[::-1][:20].sum(),
        cosine_h(x, p),
        float(np.sum(xp * np.log((xp + EPS) / (p[present] + EPS)))),
        float(np.sum(p * np.log((p + EPS) / (x + EPS)))),
    ]))

    # 5. expected leave distribution                              (2)
    ql = n * bn
    ql = ql / max(ql.sum(), EPS)
    qlp = ql[ql > 0]
    feats.append(np.array([
        cosine_h(x, ql), -np.sum(qlp * np.log(qlp + EPS)),
    ]))

    # 6. self-coupling                                            (6)
    xb = float(np.sum(x * b))
    corr = 0.0
    if present.sum() > 2:
        xv, bv = x[present], b[present]
        sxv, sbv = xv.std(), bv.std()
        if sxv > 0 and sbv > 0:
            corr = float(np.corrcoef(xv, bv)[0, 1])
    feats.append(np.array([
        np.log1p(xb), float(np.sum(x * np.log1p(b))), corr,
        cosine_h(x, b), float(p[present].sum()), float(p[n >= 2].sum()),
    ]))

    # 7. two-step propagation                                     (5)
    b2 = beta @ p
    p2 = RHO * (1.0 + b2)
    p2 = p2 / p2.sum()
    feats.append(np.array([
        float(np.sum(x * np.log1p(b2))), cosine_h(p, p2),
        -np.sum(p2 * np.log(p2 + EPS)),
        np.sort(p2)[::-1][:10].sum(), np.log1p(float(np.sum(x * b2))),
    ]))

    # 8. weighted sub-network spectrum                            (10)
    top = np.argsort(x)[::-1][:25]
    W = beta[np.ix_(top, top)] * x[top][None, :]
    ev = np.linalg.eigvals(W)
    mods = np.sort(np.abs(ev))[::-1]
    mods = np.concatenate([mods, np.zeros(8)])[:8]
    feats.append(np.log1p(mods))
    feats.append(np.array([
        np.log1p(abs(np.trace(W))), np.log1p(abs(np.trace(W @ W))),
    ]))

    # 9. sorted x*b profile, top 20                               (20)
    feats.append(np.log1p(np.sort(x * b)[::-1][:20]))

    # 10. sorted boost over present species, top 20               (20)
    bp = np.sort(b[present])[::-1]
    bp = np.concatenate([bp, np.zeros(20)])[:20]
    feats.append(np.log1p(bp))

    # 11. pairwise log-beta among present, x-x weighted           (6)
    idx = np.where(present)[0]
    lbeta = np.log(beta[np.ix_(idx, idx)] + EPS)
    w = np.outer(x[idx], x[idx])
    w = w / w.sum()
    mu_w = float(np.sum(w * lbeta))
    sd_w = float(np.sqrt(max(np.sum(w * lbeta ** 2) - mu_w ** 2, 0.0)))
    feats.append(np.array([
        mu_w, sd_w, *np.quantile(lbeta, [0.1, 0.5, 0.9]), lbeta.max(),
    ]))

    # 12. alignment with dominant eigenvector of beta*x           (2)
    v = x.copy() + 1e-3
    for _ in range(20):
        v = beta @ (x * v)
        nv = np.linalg.norm(v)
        if nv < EPS:
            break
        v = v / nv
    lam = float(v @ (beta @ (x * v)))
    feats.append(np.array([cosine_h(np.abs(v), x), np.log1p(abs(lam))]))

    # 13. sorted join distribution, top 20                        (20)
    feats.append(np.sort(p)[::-1][:20])

    # 14. split-stability proxies                                 (3)
    het = float(np.sum(x * (1 - x)))
    feats.append(np.array([
        simpson * N, het / max(N * simpson, EPS),
        1.0 / (1.0 + het / max(2 * N * simpson, EPS)),
    ]))

    # 15. total rate scalars                                      (3)
    jt = float(np.sum(KF * RHO * N * bn))
    lt = float(np.sum(KB * n * bn))
    feats.append(np.array([np.log1p(jt), np.log1p(lt),
                           np.log1p(jt / max(lt, EPS))]))

    # 16. sorted log-boost over all species, top 27               (27)
    feats.append(np.sort(lb)[::-1][:27])

    out = np.concatenate(feats)
    assert out.shape == (195,), out.shape
    return out


# ----------------------------------------------------------------------
# Beta-only matrix features
# ----------------------------------------------------------------------

def beta_only(beta: np.ndarray) -> np.ndarray:
    lb = np.log(beta)
    rows = lb.mean(axis=1)
    cols = lb.mean(axis=0)
    ev = np.linalg.eigvals(beta / NG)
    mods = np.sort(np.abs(ev))[::-1][:5]
    bq = np.quantile(beta, [0.5, 0.9, 0.99, 0.999])
    from scipy import stats as st
    return np.array([
        lb.mean(), lb.std(), float(st.skew(lb.ravel())),
        float(st.kurtosis(lb.ravel())),
        *np.log(bq + EPS),
        rows.mean(), rows.std(), rows.max(),
        cols.mean(), cols.std(), cols.max(),
        np.diag(lb).mean(), np.diag(lb).std(),
        float(np.corrcoef(lb.ravel(), lb.T.ravel())[0, 1]),
        np.log1p(mods[0]), np.log1p(mods[1]), np.log1p(mods[2]),
        np.log1p(mods[3]), np.log1p(mods[4]),
        np.log1p(float(beta.max())),
        np.log1p(float(beta.sum() / NG ** 2)),
    ])
