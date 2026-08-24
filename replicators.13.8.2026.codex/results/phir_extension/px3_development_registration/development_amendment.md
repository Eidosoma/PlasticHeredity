# PX3 resource-bounded development amendment

Date: 2026-08-21

This amendment supersedes only the development-cohort size and execution
budget in the original PX3 registration
`86a91cf7c75b6b268e25fbd06f5b1efd94eccd37fac16c49adea65ed3977707d`.
The original 24-matrix development gate remains incomplete and can never be
reported as passed.

## Reason and timing

The first six matrix checkpoints consumed 23.7793 aggregate process CPU-hours,
already exceeding the registered 20-hour allocation. After the phase paused,
the six-matrix development diagnostics were inspected at the user's request.
They were provisionally favorable. The reduction below is therefore explicitly
post-interim and may not be represented as an untouched confirmatory stopping
decision.

## Amended development design

- Use matrix identifiers 0 through 11: 12 development matrices total.
- Retain both candidates, two replicates, landmarks 20/30/40/50, 24 edits per
  state, 16 futures per edit, horizon F8, the fixed ridge grid, five-fold
  whole-matrix cross-validation, endpoints, features, seeds, and inference.
- Carry forward matrix checkpoints 0 through 5 only after their prior
  registration identity and scientific digests are verified.
- Generate matrices 6 through 11 under the identical scientific worker.
- Exactly replay all 12 matrices from their frozen seed keys.
- Run detached with at most eight workers.
- Use an amended cumulative execution ceiling of 104 process CPU-hours. This
  ceiling accommodates generation plus exact replay; expected wall time after
  amendment is approximately six to eight hours on eight workers.
- No 48-matrix run is authorized.

The original program-wide 80-CPU-hour estimate was inaccurate for this phase.
The overrun, discarded in-flight work, and all carried-forward checkpoints
remain in the audit trail.

## Claim boundary

This phase may develop and freeze a resource-bounded pilot selector. Even if
its internal development criterion is positive, it does not pass the original
24-matrix gate. A later completely fresh result may be called only a
prospective confirmation of a pilot-developed selector. It may not be called
the originally registered PX3 confirmation.

The public nine-atom Phi-r result remains negative/inconsistent and unchanged.
The material full-block reading remains a separate, representation-specific
information statistic. No result implies that Phi causes heredity,
consciousness, agency, life, or a universal origin-of-life mechanism.
