"""Third figure-candidate from the authors' compute_phi: their code's
info["emergence"] = synergy + downward causation = sts + stx + sty (3
atoms) — the quantity their code literally NAMES "emergence", distinct
from both the printed Psi and info["integrated"] (= local_phi_r, 9
atoms). Also info["synergy"] (sts alone).

Runs the sign regime + C2/C3/C4/consistency headline for both, on the
same macro pipeline as phi_r_code. C5 needs no separate cells: the
atoms16 closure test upper-bounds every projection of the lattice,
including these. Writes results/recovery_emergence3.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
np.seterr(all="ignore")

from phi_r_code import macro_halves, local_phi_id, S, U0, U1
from reanalyze_authors import headline

ROOT = Path(__file__).parent.parent
EMERGENCE3 = [(S, S), (S, U0), (S, U1)]   # their synergy + causation


def main():
    files = sorted((ROOT / "results" / "runs_coarse").glob("run_*.npz"))
    rows_e3, rows_sts, m_e3, m_sts = [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        counts = d["counts"].astype(float)
        edge, _ = macro_halves(counts, "full")
        if edge is None:
            continue
        pi = local_phi_id(edge)
        e3 = np.sum([pi[a] for a in EMERGENCE3], axis=0)
        sts = np.asarray(pi[(S, S)], float)
        sr = d["sr"][1:]
        rows_e3.append({"phi": e3, "sr": sr, "counts": counts})
        rows_sts.append({"phi": sts, "sr": sr, "counts": counts})
        m_e3.append(float(np.nanmean(e3)))
        m_sts.append(float(np.nanmean(sts)))
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)

    m_e3, m_sts = np.array(m_e3), np.array(m_sts)
    out = {
        "sign_regime": {
            "emergence3_run_mean":
                f"{m_e3.mean():+.4f}±{m_e3.std():.4f}",
            "emergence3_runs_positive": int((m_e3 > 0).sum()),
            "sts_run_mean": f"{m_sts.mean():+.4f}±{m_sts.std():.4f}",
            "sts_runs_positive": int((m_sts > 0).sum()),
            "n_runs": len(m_e3),
        },
        "headline_emergence3": headline(rows_e3),
        "headline_sts": headline(rows_sts),
    }
    print(json.dumps(out, indent=2, default=float))
    (ROOT / "results" / "recovery_emergence3.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("written results/recovery_emergence3.json")


if __name__ == "__main__":
    main()
