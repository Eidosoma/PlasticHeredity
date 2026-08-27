# Golden-trace reconciliation report

Generated: 2026-08-20 16:53:13 UTC

## Headline

Golden replay: **PASS** (907/907 sweeps, 15/15 spectra).
Raw 88-rule atlas exact reproduction: **YES**.
Five-point, 440-cell phase exact reproduction: **NO**.

## Core contract

- Contract digest: `0a37385a1177ac49f534a160b730a744542bacc3248a42b56d4237412b599f41`
- NumPy: `2.5.2` (pinned `2.5.2`)
- Launch: 16 shared heterogeneous rows; no preparation.
- Generation: post-rule noise, realized activity, timeout-or-monochrome boundary death.
- Observation: first completed generation, terminal pre-copy final4; copy draws unconditional.
- RNG: one shrinking PCG64 batch stream per rule and seed.

## Exact endpoint counts

- Atlas: `{"break_by_8": 88, "mean_survival": 88, "median_gen_sweeps": 88, "strict": 88}`
- Atlas libraries: `88/88`
- Phase: `{"break_by_8": 100, "median_gen_sweeps": 405, "strict": 190}`
- Phase Spearman: `{"break_by_8": 0.9987731599093972, "median_gen_sweeps": 0.9990471541911295, "strict": 0.963616425877581}`
- Phase MAE: `{"break_by_8": 0.004777388139204545, "median_gen_sweeps": 0.07727272727272727, "strict": 0.002068536931818182}`

## Downstream stages

- Particle rule 110 strict: `0.01806640625`; redemption gate: `True`. Exact particle numerics are not claimed because four domain launch rows remain undisclosed.
- Life supports: glider `159`, blinker `667`, toad `2715`; all three Life gates pass.
- Evolution: selection-boundary gate `True`; sticky-walk gate `False` (both match the retained decisions).

The full per-rule, per-cell, gate, and reference comparisons are retained in `RESULTS.json`.
