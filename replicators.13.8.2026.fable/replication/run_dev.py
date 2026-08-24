"""Development stage (L53 analog): generate 40 dev matrices x 2
candidates x 100 fissions, extract per-fission training rows with
realized F12 JOINT_BREAK_RUN3 outcomes, train the four students per
candidate, and freeze them."""

import json
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

N_DEV_MATRICES = 40
N_WORKERS = 12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=int, default=40)
    ap.add_argument("--tag", type=str, default="2026-08-13")
    ap.add_argument("--out", type=str, default="results")
    return ap.parse_args()


def main():
    global N_DEV_MATRICES, OUT
    args = parse_args()
    N_DEV_MATRICES = args.matrices
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    cohort.DEV_ENTROPY = cohort.domain_entropy("dev", args.tag)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_DEV_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(cohort.dev_unit, jobs)
    print(f"dev cohort simulated in {time.time()-t0:.0f}s "
          f"({len(units)} trajectories)")

    bundles, summary = {}, {}
    for cand in cohort.CANDIDATES:
        cu = [u for u in units if u["candidate"] == cand]
        n_empty = sum(1 for u in cu if len(u["y"]) == 0)
        if n_empty:
            print(f"cand {cand}: {n_empty} trajectories yielded no "
                  f"training rows (early death), skipped")
        cu = [u for u in cu if len(u["y"]) > 0]
        X9 = np.vstack([u["X9"] for u in cu])
        X195 = np.vstack([u["X195"] for u in cu])
        Xb = np.vstack([np.tile(u["Xbeta"], (len(u["y"]), 1)) for u in cu])
        y = np.concatenate([u["y"] for u in cu])
        t1 = time.time()
        bundles[cand] = models.train_students(X9, X195, Xb, y)
        p = models.predict(bundles[cand], X9, X195, Xb)
        insample = {k: float(np.mean(-(y * np.log(v) + (1 - y) * np.log(1 - v))))
                    for k, v in p.items()}
        summary[cand] = {
            "n_examples": int(len(y)),
            "prevalence": float(y.mean()),
            "died": int(sum(u["died"] for u in cu)),
            "insample_logloss": insample,
            "pca_explained_var": [float(v) for v in
                                  bundles[cand]["pca"].explained_variance_ratio_],
            "train_seconds": time.time() - t1,
        }
        print(cand, summary[cand]["n_examples"], "examples, prevalence",
              round(summary[cand]["prevalence"], 3),
              "in-sample logloss", {k: round(v, 4) for k, v in insample.items()})

    frozen_hash = models.freeze(bundles, os.path.join(OUT, "frozen_models.pkl"))
    summary["frozen_models_sha256"] = frozen_hash
    summary["dev_entropy_hex"] = hex(cohort.DEV_ENTROPY)
    with open(os.path.join(OUT, "dev_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("frozen:", frozen_hash[:16], "| total", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
