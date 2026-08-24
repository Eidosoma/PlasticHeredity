"""Run the 100-simulation GARD batch, compute Phi_r + self-replication labels.

Saves one npz per run to results/runs/run_<seed>.npz with:
  counts (n_steps, n_types) int16, generation, fission_steps,
  phi (n_steps - lag,), sr (n_steps,) bool self-replication labels,
  m1/m2 (MIB parts), phi_static, sim_to_center.
Alignment convention: phi[t] describes transition t -> t+lag; downstream
analyses align phi[t] with sr[t + lag].
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from gard import GardParams, simulate
from composomes import label_self_replication
from phi import phi_r_local

N_RUNS = 100
LAG = 1


def process_run(seed: int, p: GardParams, outdir: Path) -> dict:
    traj = simulate(seed=seed, p=p)
    counts = traj.counts
    phi, info = phi_r_local(counts, lag=LAG, mib_seed=seed)
    sr = label_self_replication(counts, traj.fission_steps, seed=seed)
    np.savez_compressed(
        outdir / f"run_{seed:03d}.npz",
        counts=counts.astype(np.int16),
        generation=traj.generation.astype(np.int32),
        fission_steps=traj.fission_steps.astype(np.int64),
        phi=phi.astype(np.float32),
        sr=sr,
        m1=info["m1"], m2=info["m2"],
        phi_static=info["phi_static"],
    )
    return dict(seed=seed, n_steps=len(counts), sr_frac=float(sr.mean()),
                phi_static=info["phi_static"],
                n_spikes=int((phi > phi.mean() + 3 * phi.std()).sum()))


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "fine")
    sub = {"fine": "runs", "coarse": "runs_coarse", "pred": "runs_pred"}[mode]
    outdir = Path(__file__).parent.parent / "results" / sub
    outdir.mkdir(parents=True, exist_ok=True)
    # coarse: dt raised so a generation spans ~10 molecular steps (~1000/run),
    # matching the step scale implied by the paper's Table 1
    # pred: the high-fate-predictability regime found by regime_sweep
    # (k_b=1e-4 as in GARD10, k_f*rho=1e-3; dt tuned back to ~1000 steps/run)
    p = {"fine": GardParams(),
         "coarse": GardParams(dt=0.4, max_events_per_step=24.0),
         "pred": GardParams(dt=0.1, max_events_per_step=24.0,
                            k_b=1e-4, k_f=1e-1)}[mode]
    t0 = time.time()
    rows = []
    for seed in range(N_RUNS):
        rows.append(process_run(seed, p, outdir))
        if (seed + 1) % 10 == 0:
            print(f"{seed + 1}/{N_RUNS} done ({time.time() - t0:.0f}s)", flush=True)
    sr = np.array([r["sr_frac"] for r in rows])
    ns = np.array([r["n_steps"] for r in rows])
    sp = np.array([r["n_spikes"] for r in rows])
    print(f"\nsteps/run: median {int(np.median(ns))} [{ns.min()}-{ns.max()}]")
    print(f"self-replication fraction: median {np.median(sr):.2f} "
          f"[{sr.min():.2f}-{sr.max():.2f}]; runs with any SR: {(sr > 0).sum()}")
    print(f"runs with >=1 Phi spike (>3SD): {(sp > 0).sum()}")


if __name__ == "__main__":
    main()
