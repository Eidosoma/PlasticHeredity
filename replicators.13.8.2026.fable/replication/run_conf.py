"""Untouched confirmation stage (L54 analog).

- New 256-bit seed domain, 40 new matrices, 80 trajectories.
- 5 restored post-fission landmark states per trajectory (400 states).
- 64 independent F12 branches per state, halves of 32 fixed before
  outcomes: 25,600 branches; the whole campaign is regenerated a second
  time with identical seeds and compared by hash (replay gate).
- Frozen L53-analog models are applied without refitting.

Writes results/conf_data.pkl with everything analyze.py needs.
"""

import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

import cohort
import models

N_CONF_MATRICES = 40
N_WORKERS = 12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=int, default=40)
    ap.add_argument("--tag", type=str, default="2026-08-13")
    ap.add_argument("--out", type=str, default="results")
    return ap.parse_args()


def run_campaign():
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_CONF_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(cohort.conf_unit, jobs)
    return units


def main():
    global N_CONF_MATRICES, OUT
    args = parse_args()
    N_CONF_MATRICES = args.matrices
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", args.tag)
    bundles, frozen_hash = models.thaw(os.path.join(OUT, "frozen_models.pkl"))
    print("loaded frozen models:", frozen_hash[:16])

    t0 = time.time()
    units1 = run_campaign()
    h1 = cohort.campaign_hash(units1)
    print(f"campaign 1 done in {time.time()-t0:.0f}s, hash {h1[:16]}")

    t1 = time.time()
    units2 = run_campaign()
    h2 = cohort.campaign_hash(units2)
    print(f"campaign 2 done in {time.time()-t1:.0f}s, hash {h2[:16]}")
    replay_ok = h1 == h2
    print("replay gate:", "PASS" if replay_ok else "FAIL")

    # collate per-state table with frozen predictions
    table = []
    for u in units1:
        cand = u["candidate"]
        for s in u["states"]:
            X9 = s["X9"][None, :]
            X195 = s["X195"][None, :]
            Xb = u["Xbeta"][None, :]
            p = models.predict(bundles[cand], X9, X195, Xb)
            table.append({
                "candidate": cand, "matrix": s["matrix"],
                "landmark": s["landmark"],
                "X9": s["X9"], "X195": s["X195"], "Xbeta": u["Xbeta"],
                "qA": float(s["qA"]), "qB": float(s["qB"]),
                "q": float(s["y64"].mean()),
                "y64": s["y64"].astype(np.int8),
                "p_full": float(p["full"][0]),
                "p_direct": float(p["direct"][0]),
                "p_beta": float(p["beta"][0]),
                "p_prior": float(p["prior"][0]),
                "proc": s["proc"],
            })

    n_dead = sum(u["died"] for u in units1)
    with open(os.path.join(OUT, "conf_data.pkl"), "wb") as f:
        pickle.dump({
            "table": table,
            "replay": {"hash1": h1, "hash2": h2, "ok": replay_ok},
            "frozen_models_sha256": frozen_hash,
            "conf_entropy_hex": hex(cohort.CONF_ENTROPY),
            "n_dead_trajectories": n_dead,
        }, f, protocol=4)
    print(f"states: {len(table)} | dead trajectories: {n_dead} | "
          f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
