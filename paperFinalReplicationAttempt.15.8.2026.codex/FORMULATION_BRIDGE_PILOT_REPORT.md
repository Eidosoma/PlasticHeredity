# Arrivals formulation bridge: pilot report

**Completed:** 2026-08-21  
**Registration:** `95f8359f17e5c14790dc4fe0cc6c4014e0b78deb54c76d328436dc96f385695c`  
**Protocol:** [FORMULATION_BRIDGE_PROTOCOL.md](FORMULATION_BRIDGE_PROTOCOL.md)

## Result

None of the four frozen instruments passed the combined retrospective-
association and leakage-free early-prediction screen. The registered next
action is therefore `stop_pending_author_code_or_new_instrument`.

This result does not alter the original negative replication, establish that
the paper is wrong, or show that the provisional full-dimensional instrument
is generally uninformative. It says only that the formulation lead did not
produce a viable instrument in this fixed 12-seed pilot under the existing
reconstruction's trajectories and labels.

No Phi-guided intervention or paper-scale rerun was launched.

## Frozen screen results

| Instrument | Evaluable associations | Positive rho | Mean rho | Median rho | Early accuracy | Majority accuracy | Held-out wins vs majority | Pilot viable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original macro WMS | 11/12 | 4/11 | -0.0244 | -0.0164 | 0.5052 | 0.9462 | 0/12 | No |
| Original macro MMI | 11/12 | 2/11 | -0.0993 | -0.1295 | 0.4991 | 0.9462 | 0/12 | No |
| Public nine-atom revised Phi-r | 11/12 | 0/11 | -0.0583 | -0.0471 | 0.5234 | 0.9462 | 0/12 | No |
| Provisional full-dimensional revised | 11/12 | 3/11 | -0.0128 | -0.0108 | 0.5269 | 0.9462 | 0/12 | No |

The association target was nine positive correlations, calculated before the
pilot as `ceil(0.73 * evaluable_runs)`. Every instrument missed it. The
full-dimensional score had the least-negative mean rho and the highest raw
early-prediction accuracy, but its median rho remained negative, it was
positive in only 3 of 11 evaluable runs, and it never beat the majority dummy
on a held-out seed. Those relative rankings are descriptive and do not pass a
gate.

The MMI addition did not improve the original macro estimator in this cohort:
its mean rho was 0.0748 lower than macro WMS. Relative to the public nine-atom
score, the full-dimensional score had higher rho in 8 of 11 paired evaluable
runs, but the mean early-accuracy advantage was only 0.00347 and occurred in 4
of 12 held-out runs. This is not a prediction rescue.

## Replicator-label audit

The existing 0.95 cosine detector labeled only 5.58% of molecular observations
as replicating across the new trajectories (range 0% to 20.21%). One seed had
constant-zero labels and was non-evaluable for within-run association. The
paper reported 88% for its control definition; the initial 100-seed
reconstruction reported 16.7%.

The more severe 5.58% pilot imbalance made the training-only majority dummy
94.62% accurate. The four MLPs remained near chance at 49.91% to 52.69%.
Therefore the early-prediction comparison is primarily evidence that none of
these score histories overcame the reconstruction-label imbalance in a tiny
leave-one-seed-out pilot. It is not a well-powered comparison of forecasting
architectures.

The detector discrepancy remains unresolved and is still a plausible reason
the paper's results do not map to this reconstruction. The formulation bridge
cannot repair a non-equivalent outcome definition.

## Estimator and execution audit

- Original macro WMS and MMI exactly replay the existing implementation.
- The PX port matches the source-hashed plastic-heredity implementation on a
  frozen synthetic fixture for both the public nine-atom and full-block global
  values.
- All 32 repository tests passed; nine are bridge-specific synthetic gates.
- All 12 scientific traces regenerated bit-for-bit from their registered
  seeds and hashes.
- All 96 archived full/prefix estimator arrays replayed exactly.
- The provisional full-dimensional local score averaged to its registered
  global formula with maximum absolute error
  `5.95e-14` across the 12 full trajectories.
- All PX fits retained 100 active molecular coordinates. The original
  drop-last fits retained 99.
- The full-dimensional minimum channel was cross-part in every run: A-to-B in
  eight and B-to-A in four.
- The traces contained zero interventions.
- Runtime provenance records Python 3.13.5, NumPy 2.5.2, SciPy 1.18.0, pandas
  3.0.5, and scikit-learn 1.9.0. These are newer compatible wheels than the
  repository's declared Python-era pins; exact versions are retained with the
  outputs.

## Interpretation and stop rule

The new formulation was scientifically worth testing because it changes the
information instrument rather than a parameter. In this bounded cohort it did
not recover the paper-like sign pattern or useful early prediction. The most
defensible action is now to preserve both negative results and wait for the
authors' detector and online-estimator code.

If that code arrives, the next audit should proceed in this order:

1. reproduce the authors' control replication probability and label sequence;
2. establish simulator/trajectory parity or characterize its failure;
3. establish exact estimator parity on shared synthetic and trajectory
   fixtures; and only then
4. rerun the observational four-instrument bridge on author-equivalent labels.

Interventions remain out of scope until an observational instrument is viable
under a resolved label contract and its online, past-only scoring rule is
specified.

## Generated evidence

The complete gitignored result directory is
`results/formulation-bridge-pilot12/`. Its principal files are:

- `SUMMARY.md` and `pilot_screen.json`;
- `trace_manifest.csv` with all 12 trace hashes and label rates;
- `association_runs.csv` and `association_summary.csv`;
- `early_prediction_runs.csv` and `early_prediction_summary.csv`;
- `paired_estimator_contrasts.csv`;
- `estimator_components.csv` and `instrument_contract.csv`;
- the exact registration and runtime provenance; and
- 12 compressed trace checkpoints plus 12 score checkpoints.

The separate registration is in
`results/formulation-bridge-registration/registration.json`.
