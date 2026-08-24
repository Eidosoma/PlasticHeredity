"""Part A orchestration: regenerate the 5x confirmation cohort's branch
inheritance sequences from seeds, then run the Markov-vs-IID analysis
under both the published (biased) and reviewer-corrected IID fits, plus
the two null calibrations."""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np

import cohort
import markov_iid as MI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_markov")
CONF_TAG = "5x-2026-08-13"
N_MATRICES = 200
N_WORKERS = 12
N_BOOT = 2048


def main():
    os.makedirs(OUT, exist_ok=True)
    cohort.CONF_ENTROPY = cohort.domain_entropy("confirmation", CONF_TAG)
    t0 = time.time()
    jobs = [(m, c) for c in cohort.CANDIDATES for m in range(N_MATRICES)]
    with Pool(N_WORKERS) as pool:
        units = pool.map(cohort.conf_sequences_unit, jobs)
    print(f"sequences regenerated in {time.time()-t0:.0f}s")

    results = {}
    for cand in cohort.CANDIDATES:
        groups = {}
        for u in units:
            if u["candidate"] != cand:
                continue
            sfx = []
            for s in u["states"]:
                sfx.extend(MI.post_break_suffixes(s["seqs"], s["lens"]))
            groups[u["matrix"]] = sfx
        stats = MI.support_stats(groups)
        res = {"support": stats}

        for name, fitter in [("biased", MI.fit_iid_biased),
                             ("corrected", MI.fit_iid_corrected)]:
            per_mat = MI.crossfit_gain(groups, fitter)
            rng = np.random.default_rng(555)
            res[name] = MI.summarize(per_mat, rng, N_BOOT)

        # null calibrations (16 replicates each, matched length profiles)
        for null in ("iid", "nonstat"):
            pooled = {"biased": [], "corrected": []}
            for rep in range(16):
                nrng = np.random.default_rng(10_000 + rep)
                sim_groups = MI.simulate_null(groups, nrng, null)
                for name, fitter in [("biased", MI.fit_iid_biased),
                                     ("corrected", MI.fit_iid_corrected)]:
                    pm = MI.crossfit_gain(sim_groups, fitter)
                    tot = sum(v[0] for v in pm.values())
                    n = sum(v[1] for v in pm.values())
                    pooled[name].append(tot / max(n, 1) / MI.LOG2)
            res[f"null_{null}"] = {
                k: {"mean_bits": float(np.mean(v)),
                    "sd_bits": float(np.std(v))}
                for k, v in pooled.items()}
        results[cand] = res

        print(f"\n=== Candidate {cand} ===")
        s = stats
        print(f"suffixes {s['n_suffixes']} (empty {s['n_empty']}, "
              f"singleton {s['n_singleton']}, mean len {s['mean_len']:.2f}) | "
              f"first-symbol rate {s['first_symbol_rate']:.3f} vs "
              f"destination rate {s['destination_rate']:.3f}")
        for name in ("biased", "corrected"):
            r = res[name]
            print(f"{name:9s} gain: pooled {r['pooled_bits']:+.4f} bits "
                  f"[{r['pooled_ci'][0]:+.4f},{r['pooled_ci'][1]:+.4f}] | "
                  f"macro {r['macro_bits']:+.4f} bits "
                  f"[{r['macro_ci'][0]:+.4f},{r['macro_ci'][1]:+.4f}] | "
                  f"n_trans {r['n_transitions']}")
        for null in ("iid", "nonstat"):
            n = res[f"null_{null}"]
            print(f"null {null:7s}: biased {n['biased']['mean_bits']:+.4f}"
                  f"±{n['biased']['sd_bits']:.4f} | corrected "
                  f"{n['corrected']['mean_bits']:+.4f}"
                  f"±{n['corrected']['sd_bits']:.4f}")

    with open(os.path.join(OUT, "markov_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten:", os.path.join(OUT, "markov_results.json"))


if __name__ == "__main__":
    main()
