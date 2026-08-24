# Cluster Report: Foundation

## Cluster scope

`FOUNDATION` comprises S01–S11 and S11R. Its purpose was not to test whether the manuscript's biological conclusions were true. It was to make later tests interpretable by fixing four prerequisites: a complete claim/specification registry, independent GARD implementations, explicit self-replicator and compositional-data semantics, and validated information-dynamic estimators.

## Methods and tools

The work combined manuscript forensics, source pinning, clean-room Python implementations, deterministic fixtures, Monte Carlo distribution tests, fault injection, typed trajectory and seed schemas, centered-log-ratio transformation audits, and synthetic PhiID benchmarks. Scientific branches were treated as explicit alternatives rather than silently filled defaults. Where a method could not satisfy its preregistered numerical or sample-size contract, it was made ineligible rather than rescued using a favorable downstream result.

The simulation foundation used two separately structured engines:

- a public-historical compatibility implementation;
- an independent NumPy implementation exposing categorical-event, direct-Gillespie, and vector-Poisson branches, alternative fission laws, and daughter-continuation choices.

The information foundation used pinned `phyid` and guarded OmegaID branches, independent analytic or enumerated oracles, all 16 PhiID atoms, explicit minimum-information-partition candidates, and positive/negative synthetic systems including independent noise, duplicated Markov bits, XOR synergy, coupled Gaussian autoregression, and planted block structure.

## Key findings

### Specification uncertainty was material

S01 registered 59 manuscript claims and 12 discrepancies. S02 expanded this into 120 parameters, including 64 unresolved, conflicting, or evidence-deferred items and 21 branch sets. The registry deliberately remained non-executable until a complete branch was chosen. Important ambiguities included the update clock, Poisson exposure, fission and daughter semantics, molecular versus generational observation clocks, self-replicator labels, CLR zero handling, PhiID redundancy and atom identities, temporal fitting, and intervention scoring.

### The reconstructed simulator family was internally reproducible

S05 showed that two independent engines reproduced the same model-level distributions when configured to the same historical branch. All 512 deterministic propensity cases agreed exactly; event, fission, and one-generation distribution distances passed prospectively set tolerances. S07 then passed 26/26 stochastic distribution tests, 54/54 deterministic invariants, and 7/7 injected-failure detectors. The minimum primary Monte Carlo p-value, 0.030695, remained well above the multiplicity-adjusted rejection threshold of 0.0001923.

![Observed-versus-expected stochastic event frequencies and standardized residuals for the validated simulator profiles.](figures/foundation_s07_event_validation.png)

*Figure FOUNDATION-1. Stochastic event-law validation from S07. Bars compare observed and expected event probabilities across explicit historical, direct-Gillespie, and vector-Poisson fixtures; the residual panels remain within the preregistered diagnostic envelope. This validates the named reconstruction branches, not unavailable author code.*

### Label definitions were not interchangeable

S08 established that adjacent-state similarity, boundary inheritance, compotype membership, and retrospective recurring-attractor labels can classify the same trajectory differently. Past-only and completed-run labels can also move onset in opposite directions. This result became central later: an apparent association can be induced or suppressed by the label clock and its use of future observations.

### Compositional transforms were numerically controlled

S09 enumerated 13 zero-treatment choices and validated closure, CLR transformation, dimension removal, inversion, and provenance on 4,901 rows. Maximum inverse error was approximately `2e-15`. This removed a potential low-level explanation for later discrepancies while preserving zero treatment as a declared branch.

### PhiID was validated branch by branch, not globally

S10 processed 448 synthetic cases and 8,512 atom results. Lattice closure was no worse than `2.22e-16` nats. Pinned-reference and guarded Gaussian CPU/GPU branches agreed within binary64 tolerances; the plotted backend discrepancies were approximately `1e-18` to `1e-13`, below the `1e-10` contract.

![Absolute agreement errors among eligible PhiID backends.](figures/foundation_s10_backend_agreement.png)

*Figure FOUNDATION-2. Eligible PhiID backend agreement in S10. Agreement for Gaussian and reference fixtures was several orders of magnitude inside the frozen tolerance. The plot does not include the discrete OmegaID branch that failed relabel invariance and was excluded.*

One important negative result was preserved: OmegaID's discrete 2×2 implementation failed all 16/16 independent relabel controls on both CPU and GPU, whereas the pinned reference passed its discrete controls. That branch was declared ineligible. A second constraint followed from sample-size validation: the strict estimator was supported only at 512 or more effective samples, so the proposed 32–256-observation fixed windows could not be used without registering a different estimator. S11/S11R therefore constrained the initial fixed-window plan rather than weakening the gate.

## Null results and contradictions

- Distributional agreement between reconstructed simulators did not identify the paper authors' exact implementation.
- Passing a paper-like vector-Poisson fixture did not resolve the paper's missing exposure, clipping, overshoot, or daughter details.
- The paper's printed Phi-r equation mapped to a signed aggregate of atoms, not unambiguously to one source-named atom.
- A numerically accelerated discrete information branch was invalid under harmless state relabeling.
- Fixed windows shorter than 512 effective samples lacked validated estimator support.

## Evidence assessment

| Question | Evidence | Assessment |
|---|---|---|
| Can the reconstructed GARD transition laws be simulated reproducibly? | Two independent engines, distributional tests, replay, fault injection | Supported for explicit branches |
| Is one branch the authors' implementation? | No author code; unresolved specification registry | Underdetermined |
| Are compositional transforms a likely source of large downstream differences? | Inversion and closure errors near machine precision | Unlikely within tested transforms |
| Is every tested PhiID backend eligible? | Gaussian/reference branches pass; discrete OmegaID relabeling fails | No; branch-specific constraint |
| Are short fixed-window Phi estimates validated? | Effective-sample gate fails below 512 | No |

## Cluster conclusion

The foundation is strong enough to support a forensic replication: simulator branches, seeds, stochastic laws, labels, transforms, and information calculations are explicit and testable. It is not strong enough to infer an unavailable implementation from manuscript prose. The most consequential lesson is methodological: apparent paper resemblance must survive branch, time, label, and future-information audits before it can support a prospective or causal claim.

