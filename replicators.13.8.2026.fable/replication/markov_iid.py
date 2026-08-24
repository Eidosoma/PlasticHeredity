"""Markov-versus-IID inheritance analysis, in both the published
(support-mismatched) form and the reviewer-corrected form.

Sequences are post-break F12 suffixes: for each branch's 12 inheritance
flags, find the first break t; the suffix is flags[t+1:]. Empty and
singleton suffixes are retained — they are exactly what the reviewer's
bug feeds on.

All probabilities use identical Jeffreys (+0.5) smoothing. Both models
are fit and scored under identical two-way cross-fitting by catalytic
matrix (fit on fold A, score fold B, and vice versa; folds are the
even/odd ranks of the sorted matrix ids). The ONLY difference between
the two IID variants is the fitting support:

  biased    : fit on ALL suffix symbols (first symbols and singletons
              included), score on a[1:]        (as published)
  corrected : fit on destinations a[1:] of suffixes with len >= 2,
              score on a[1:]                   (reviewer's fix)

The Markov model fits p(next | prev) on transitions and scores the same
transitions, in both variants.

Gains are reported in bits per transition:
  pooled : transition-weighted, all scored transitions pooled
  macro  : equal-weight mean over sequences (each sequence's mean
           per-transition gain counts once)
"""

from __future__ import annotations

import numpy as np

LOG2 = np.log(2.0)
ALPHA = 0.5          # Jeffreys smoothing


def post_break_suffixes(seqs: np.ndarray, lens: np.ndarray):
    """seqs: (B, 12) bool inheritance flags; returns list of 1-D bool
    arrays (may be empty) — the post-first-break suffixes."""
    out = []
    for b in range(len(seqs)):
        flags = seqs[b, :lens[b]]
        breaks = np.where(~flags)[0]
        if len(breaks) == 0:
            continue                      # no break, no suffix
        out.append(flags[breaks[0] + 1:])
    return out


def _smoothed(pos: float, n: float) -> float:
    return (pos + ALPHA) / (n + 2 * ALPHA)


def fit_iid_biased(suffixes):
    """As published: every symbol counts, including first symbols of
    each sequence and symbols of length-one sequences."""
    pos = sum(int(s.sum()) for s in suffixes)
    n = sum(len(s) for s in suffixes)
    return _smoothed(pos, n)


def fit_iid_corrected(suffixes):
    """Reviewer's fix: fit on precisely the transition destinations
    that will be scored."""
    dests = [s[1:] for s in suffixes if len(s) >= 2]
    pos = sum(int(d.sum()) for d in dests)
    n = sum(len(d) for d in dests)
    return _smoothed(pos, n)


def fit_markov(suffixes):
    """p(next=1 | prev) from transitions."""
    counts = np.zeros((2, 2))
    for s in suffixes:
        if len(s) < 2:
            continue
        prev, nxt = s[:-1].astype(int), s[1:].astype(int)
        for a in (0, 1):
            m = prev == a
            counts[a, 1] += int(nxt[m].sum())
            counts[a, 0] += int((1 - nxt[m]).sum())
    return np.array([_smoothed(counts[a, 1], counts[a].sum())
                     for a in (0, 1)])


def _seq_losses(s, p_iid, p_mk):
    """Per-transition log losses (nats) for one suffix; None if no
    transitions."""
    if len(s) < 2:
        return None
    prev, nxt = s[:-1].astype(int), s[1:].astype(int)
    q_iid = np.where(nxt == 1, p_iid, 1 - p_iid)
    q_mk = np.where(nxt == 1, p_mk[prev], 1 - p_mk[prev])
    return -np.log(q_iid), -np.log(q_mk)


def crossfit_gain(suffix_groups: dict, iid_fitter):
    """suffix_groups: {matrix_id: [suffix, ...]}. Two-way cross-fit by
    matrix rank parity. Returns per-matrix pooled sums so callers can
    bootstrap by matrix:
      {matrix: (sum_gain_nats, n_transitions, macro_gains_list)}"""
    mats = sorted(suffix_groups)
    folds = {m: (i % 2) for i, m in enumerate(mats)}
    fitted = {}
    for f in (0, 1):
        train = [s for m in mats if folds[m] != f for s in suffix_groups[m]]
        fitted[f] = (iid_fitter(train), fit_markov(train))
    out = {}
    for m in mats:
        p_iid, p_mk = fitted[folds[m]]
        tot, n, macro = 0.0, 0, []
        for s in suffix_groups[m]:
            r = _seq_losses(s, p_iid, p_mk)
            if r is None:
                continue
            l_iid, l_mk = r
            tot += float((l_iid - l_mk).sum())
            n += len(l_iid)
            macro.append(float((l_iid - l_mk).mean()))
        out[m] = (tot, n, macro)
    return out


def summarize(per_matrix, rng, n_boot=2048):
    """Pooled and macro gains in bits/transition + whole-matrix
    bootstrap CIs."""
    mats = sorted(per_matrix)
    tot = sum(per_matrix[m][0] for m in mats)
    n = sum(per_matrix[m][1] for m in mats)
    pooled = tot / max(n, 1) / LOG2
    macro_all = [g for m in mats for g in per_matrix[m][2]]
    macro = float(np.mean(macro_all)) / LOG2 if macro_all else np.nan

    boots_p, boots_m = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(mats, size=len(mats), replace=True)
        t = sum(per_matrix[m][0] for m in pick)
        c = sum(per_matrix[m][1] for m in pick)
        boots_p[i] = t / max(c, 1) / LOG2
        mg = [g for m in pick for g in per_matrix[m][2]]
        boots_m[i] = np.mean(mg) / LOG2 if mg else np.nan
    ci = lambda v: [float(np.nanquantile(v, 0.025)),
                    float(np.nanquantile(v, 0.975))]
    return {"pooled_bits": float(pooled), "pooled_ci": ci(boots_p),
            "macro_bits": macro, "macro_ci": ci(boots_m),
            "n_transitions": int(n)}


def support_stats(suffix_groups):
    """First-symbol vs destination inheritance rates and suffix-length
    profile — the direction and size of the contamination."""
    firsts, dests, lens, singles, empties = [], [], [], 0, 0
    for sfx in suffix_groups.values():
        for s in sfx:
            lens.append(len(s))
            if len(s) == 0:
                empties += 1
                continue
            firsts.append(bool(s[0]))
            if len(s) == 1:
                singles += 1
            else:
                dests.extend(s[1:].tolist())
    return {
        "n_suffixes": len(lens),
        "n_empty": empties, "n_singleton": singles,
        "mean_len": float(np.mean(lens)) if lens else 0.0,
        "first_symbol_rate": float(np.mean(firsts)) if firsts else np.nan,
        "destination_rate": float(np.mean(dests)) if dests else np.nan,
    }


def simulate_null(suffix_groups, rng, mode):
    """Nulls with matched per-matrix suffix-length profiles.
    mode='iid': stationary Bernoulli(p_m) per matrix (p_m = that
      matrix's overall suffix symbol rate).
    mode='nonstat': first symbol Bernoulli(first-rate), later symbols
      Bernoulli(destination-rate), independent — no Markov dependence."""
    out = {}
    for m, sfx in suffix_groups.items():
        symbols = [x for s in sfx for x in s.tolist()]
        p_all = np.mean(symbols) if symbols else 0.5
        firsts = [bool(s[0]) for s in sfx if len(s) > 0]
        dests = [x for s in sfx if len(s) >= 2 for x in s[1:].tolist()]
        p1 = np.mean(firsts) if firsts else p_all
        p2 = np.mean(dests) if dests else p_all
        sim = []
        for s in sfx:
            L = len(s)
            if L == 0:
                sim.append(np.zeros(0, dtype=bool))
                continue
            if mode == "iid":
                sim.append(rng.random(L) < p_all)
            else:
                seq = rng.random(L)
                thr = np.full(L, p2)
                thr[0] = p1
                sim.append(seq < thr)
        out[m] = sim
    return out
