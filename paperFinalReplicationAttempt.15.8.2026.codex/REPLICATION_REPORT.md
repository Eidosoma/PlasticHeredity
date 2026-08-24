# White-room replication report

> **Scope note (2026-08-20):** This report covers the initial white-room Phi-r
> reconstruction only. It predates the later plastic-heredity campaigns and
> reviewer-prompted controls. For revision of the current integrated preprint,
> use [PREPRINT_AGENT_HANDOFF.md](PREPRINT_AGENT_HANDOFF.md) as the entry point.

## Executive result

The central causal, predictive, and interventional claims of arXiv:2607.28250v1
did **not** reproduce under this registered clean-room reconstruction. The
reconstruction did reproduce punctuated causal-emergence trajectories and much
of their temporal autocorrelation, and it reproduced the secondary conclusion
that established graph/nonlinear-dynamics summaries do not correlate after
multiplicity correction.

This is a robustness result, not a claim that the manuscript's unavailable code
is wrong. The paper omits several result-determining definitions—notably the
GARD kinetic constants and time scale, zero treatment before CLR, exact ΦID
estimator, compotype algorithm/cutoff, online intervention estimator, and MLP
architecture. Exact numerical reproduction is therefore not identifiable from
the manuscript alone.

All primary results below use 100 independent controls and 100 matched runs in
each intervention arm, with seed 1729 through 1828. Mean control trajectory
length was 976.8 molecular steps (range 286–1550), close to the visual scale of
the paper's trajectories.

## Claim-by-claim comparison

| Claim | Preprint | Clean-room result | Assessment |
|---|---:|---:|---|
| No aggregate Φ trend | linear-regression p=0.1995 | positive slope 2.87×10⁻⁵, p=0.0120 | Not reproduced |
| Runs with >3-SD positive spikes | “most” | 97/100 | Reproduced qualitatively |
| Positive Φ/self-replication correlation | 73/100 | 40/94 evaluable; 6 constant-label runs | Not reproduced |
| Positive and significant correlation | 54/100 | 19/100; negative and significant in 27/100 | Not reproduced |
| Population mean Spearman ρ positive | significant | mean −0.0165, one-sample p=0.0986 | Not reproduced |
| Mean Φ higher in self-replication | 57/100 | 44/100 | Not reproduced by run count |
| Within-run Mann–Whitney significant | 57/100 | 25/100 | Not reproduced |
| Fisher-combined evidence | p<0.001 | p=3.08×10⁻²¹ | Reproduced, despite mixed run directions |
| Raw Φ rejects white noise | 86/100; median p=2.07×10⁻⁵¹ | 80/100; median p=1.05×10⁻⁶ | Broadly reproduced, weaker |
| Differenced Φ rejects white noise | 100/100 | 100/100 | Reproduced |
| Spike time predicts replication | ρ=+0.66, p<0.001 | ρ=−0.234, p=0.0210 | Opposite sign |
| Inter-spike distance predicts replication | ρ=+0.71, p<0.001 | ρ=−0.0440, p=0.681 | Not reproduced |
| Spike height not predictive | non-significant | ρ=0.0155, p=0.880 | Reproduced |
| Early Φ forecasts future replication better than all baselines | all p<0.01 | accuracy 0.506; majority dummy 0.820; one-sided p=0.9999 | Not reproduced |
| Max-Φ increases persistence | 874±233 vs control 716±198, p<0.001 | 141±154 vs 132±126, p=0.904 | Not reproduced |
| Max-Φ increases consistency | 0.52±0.04 vs 0.38±0.06, p<0.001 | 0.839±0.095 vs 0.853±0.076, p=0.412 | Not reproduced; point estimate opposite |
| Min-Φ reduces probability | 80±3% vs control 88±3%, p<0.001 | 17.8±21.8% vs 16.7±21.0%, p=0.800 | Not reproduced; point estimate opposite |
| Only max-Φ probability rises over generations | max p<0.001; control n.s. | all three pooled trends positive and significant; min slope largest | Not reproduced |
| No established-metric correlations | no significant comparisons | all 13 Pearson/Spearman BH-adjusted q>0.21 | Reproduced under the documented scalar reconstruction |

The original run wrote the machine-readable values to
`results/main/claim_comparison.csv`; that generated result directory is not
retained in this checkout.

## Robustness and sensitivity findings

The preprint describes self-replicators as a tight cluster relative to the most
recurring composition but supplies neither the similarity function nor its
cutoff. The registered primary detector uses standard GARD cosine/H similarity
at 0.95. It yields a control self-replication probability of 16.7%, far below
the paper's 88%.

This discrepancy is consequential but does not rescue the main claims:

- Sweeping the cutoff from 0.50 through 0.98 changes control probability from
  79.9% to 9.0%. A 0.50 cutoff brings the observational sign count closer to
  the paper (67 positive correlations), demonstrating strong label dependence.
- None of five causal-estimator variants yields a positive-correlation fraction
  near 73%; the range is 15/94 to 42/94 evaluable runs.
- Changing the unreported Poisson leap from 0.25 to 1.0 leaves mean replication
  probability near 14–16% at the primary cutoff.
- No detector cutoff produces the reported max-up/min-down intervention
  ordering. Almost every Mann–Whitney comparison is non-significant; the only
  paired cutoff result below 0.05 is max-versus-control persistence at 0.50,
  and its direction is negative.
- Repeating forecasting at all seven cutoffs never makes Φ competitive with the
  dummy. Φ accuracy remains 0.499–0.513; its smallest deficit to the dummy is
  0.044 at cutoff 0.70.
- A separate 20-seed intervention sensitivity fitted the information model to
  each completed matched control—the future-leaking alternative most favorable
  to reproducing the effect. It also showed no max-up/min-down ordering and no
  primary p-value below 0.50.

The original run wrote the full run-level sensitivity tables to
`results/main/sensitivity` and the completed-control experiment to
`results/matched-control-sensitivity`. Those generated result directories are
not retained in this checkout.

## Interpretation

The strongest result is not a clean negative verdict on the paper; it is that
the paper's headline effects are not robustly recoverable from its published
methods. Some time-series facts are stable: local Gaussian whole-minus-sum
information produces spikes and strong serial structure. But whether those
values track, forecast, or control a compotype changes materially with omitted
definitions, and none of the preregistered or sensitivity variants tested here
recovers all three.

The control probability and consistency differences also show that this
reconstruction's compotype state is not numerically equivalent to the authors'
unpublished detector. Accordingly, the intervention null could reflect a
different GARD/detector implementation as well as a non-robust causal claim.
The appropriate resolution is release of the manuscript's promised code or a
complete algorithmic specification, followed by a blinded rerun of this claim
map.

## Reproducibility audit

- Raw traces, exact catalytic matrices, independent RNG seed, joins/leaves,
  fission phases, and intervention actions/scores are checkpointed for all 300
  primary trajectories.
- All plots and aggregate tables were regenerated from those checkpoints.
- The primary action audit found a mean 99.63 interventions per 100 generations
  in both arms. Extremal scores separate strongly, so the treatment null is not
  caused by an inactive intervention routine.
- The complete test suite passes (18 tests), and the full result directory
  contains 300 trace and 300 analysis checkpoints.
- No code by the preprint authors, including older projects, was inspected or
  reused. The implementation was derived from the supplied manuscript and
  primary standard-method descriptions.

See [`docs/REPLICATION_SPEC.md`](docs/REPLICATION_SPEC.md) for the complete
reported/inferred/unresolved parameter ledger. The original run also generated
`results/main/SUMMARY.md`, which is not retained in this checkout.
