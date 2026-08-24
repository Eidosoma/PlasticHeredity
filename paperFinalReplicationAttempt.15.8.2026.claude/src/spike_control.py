"""Representation control for the route-2 spike result: the same 9
episode statistics computed on the CAUSAL BASELINE signals (molecular
flux, delta-composition) over the same first-25% window. If these also
beat all four baselines, the spike win is a property of the episode
representation, not of Phi. Classic09 labels (the informative rule).
Writes results/spike_control.json."""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from recovery_c5 import SPLIT, CONFIGS
from recovery_lattice_spikes import spike_features, experiment

ROOT = Path(__file__).parent.parent


def main():
    with open(ROOT / "results" / "lattice_features.pkl", "rb") as fh:
        runs = pickle.load(fh)
    for r in runs:
        counts = r["counts"]
        cut = int((counts.shape[0] - 1) * SPLIT)
        rel = counts / counts.sum(axis=1, keepdims=True)
        flux = np.abs(np.diff(counts, axis=0)).sum(axis=1)[:cut]
        dcomp = np.linalg.norm(np.diff(rel, axis=0), axis=1)[:cut]
        r["spikes_flux"] = spike_features(flux)
        r["spikes_dcomp"] = spike_features(dcomp)
        r["sr_classic09"] = r["sr"]
    results = {}
    for variant in ("spikes_flux", "spikes_dcomp"):
        for cfg in CONFIGS:
            name = f"classic09/{variant}/in{cfg['n_in']}"
            row = experiment(runs, variant, "sr_classic09", cfg)
            results[name] = row
            flag = "  *** BEATS ALL ***" if row["beats_all"] else ""
            print(name, json.dumps(row), flag, flush=True)
    (ROOT / "results" / "spike_control.json").write_text(
        json.dumps(results, indent=2))
    print("written results/spike_control.json")


if __name__ == "__main__":
    main()
