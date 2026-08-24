# Arrivals Phi-family covariance-support report

**Completed:** 2026-08-21  
**Amended registration:**
`52efe88727c87343b845e1cc831937b8f96769a97bb48927dc4d5acf96f710f5`  
**Protocol:** [COVARIANCE_SUPPORT_PROTOCOL.md](COVARIANCE_SUPPORT_PROTOCOL.md)  
**Label status:** [LABEL_CONTRACT_STATUS.md](LABEL_CONTRACT_STATUS.md)

## Result

The finite-support warning is confirmed, but the prospectively frozen PCA8
stabilization did not solve it. The registered numerical-stability gate failed,
so the required action is `stop_and_retain_numerical_instability_null`.

This was a label-blind numerical audit, not a second outcome search. It read or
computed no replicator labels, ran no association or prediction analysis, and
performed no interventions. Consequently, no new 12-seed outcome pilot or
paper-scale rerun is authorized.

The exact raw-statistic sign reversal observed in the separate Plastic
Heredity experiment was not reproduced on these GARD trajectories: the raw
full-block scores remained positive. The more general numerical pathology was
reproduced strongly. Their magnitude and trajectory ordering depended on the
number of transition pairs, and the ordinary score collapsed as the joint
covariance moved out of its severely rank-deficient regime.

## Fixed-trajectory support result

Six fresh untreated trajectories supplied a common 512-pair pool each. Twelve
deterministic nested subsamples at each support used the same trajectory,
terminal pair, transform, beta partition, and representation. Only covariance
support changed in the primary comparison.

| Pairs | Raw 100D joint rank / dimension | Raw median score | PCA8 joint rank / dimension | PCA8 median score |
|---:|---:|---:|---:|---:|
| 64 | 63 / 200 | 51.113 | 32 / 32 | 4.953 |
| 96 | 95 / 200 | 188.836 | 32 / 32 | 3.693 |
| 128 | 127 / 200 | 194.674 | 32 / 32 | 3.156 |
| 192 | 190 / 200 | 76.459 | 32 / 32 | 2.467 |
| 256 | 196 / 200 | 34.969 | 32 / 32 | 2.117 |
| 384 | 196 / 200 | 19.344 | 32 / 32 | 1.717 |
| 512 | 196 / 200 | 13.581 | 32 / 32 | 1.528 |

The raw 100-coordinate score rose almost fourfold between 64 and 128 pairs,
then fell by 93.0% between 128 and 512 pairs. At 64, 96, and 128 pairs, its
200-dimensional joint covariance could have rank at most 63, 95, and 127,
respectively. The fixed numerical ridge was therefore helping define the
quantity rather than merely regularizing a well-supported covariance.

The component means show that the full-dimensional whole-system Gaussian-MI
term was the largest contributor to the 128-to-512 collapse:

| Raw 100D component | 128 pairs | 512 pairs | Contribution to score change |
|---|---:|---:|---:|
| Whole-system MI | 354.889 | 63.410 | -291.480 |
| A-to-own-future MI, subtracted | 36.236 | 21.115 | +15.121 |
| B-to-own-future MI, subtracted | 141.659 | 44.941 | +96.719 |
| Minimum four-channel term, added | 32.215 | 16.155 | -16.060 |
| **Ordinary revised score** | **209.210** | **13.509** | **-195.701** |

The PCA8 representation removed covariance rank deficiency: its
32-dimensional joint covariance was full rank even at 64 pairs. That was not
enough to make its finite-sample score stable. Its ordinary median was 3.24
times the 512-pair value at 64 pairs and 2.06 times it at 128 pairs.

## Frozen PCA8 gate

| Pairs | Ordering agreement | Spearman vs 512 | Normalized drift | End-anchored ordering | All conditions pass |
|---:|---:|---:|---:|---:|:---:|
| 64 | 0.622 | 0.429 | 1.174 | 0.400 | No |
| 96 | 0.694 | 0.771 | 0.735 | 0.667 | No |
| 128 | 0.778 | 0.771 | 0.538 | 0.733 | No |
| 192 | 0.839 | 1.000 | 0.301 | 0.800 | No |
| 256 | 0.894 | 1.000 | 0.180 | 1.000 | Yes |
| 384 | 0.944 | 1.000 | 0.057 | 0.867 | Yes |

The frozen thresholds were at least 0.80 ordering agreement, at least 0.70
Spearman correlation, at most 0.25 normalized drift, and at least 0.80
end-anchored ordering agreement at every support below 512. PCA8 failed all
four conditions at 64 pairs, three at 96 and 128, and the drift condition at
192. Passing only at 256 and 384 cannot rescue an estimator whose intended
window may operate below that support.

## Other instruments

- The public nine-atom score had comparatively stable covariance-only levels:
  paired-subsample medians ranged from 0.977 to 1.047. Its end-anchored
  trajectory ordering agreement nevertheless fell to 0.467 at 96 pairs and
  0.333 at 128 pairs. It was a comparator, not a selectable replacement, and
  it was already approximately null in the completed outcome bridge.
- The typeset full-coordinate WMS reconstruction changed sign as support grew:
  its median was -140.945 at 64, +47.493 at 128, and -13.756 at 512 pairs. Its
  pairwise trajectory-contrast flip rate reached 64.4% at 128 pairs.
- The original macro WMS was much less affected in score level, but its
  trajectory-contrast flip rate reached 25.6% at 64 pairs.
- The diagnostic raw 100-coordinate revised measure had a 53.3% contrast flip
  rate at 64 pairs and normalized drift near seven at 96 and 128 pairs. It
  remains ineligible for outcome use.

These comparisons are diagnostic. The prospective protocol permits only the
PCA8 candidate to pass; selecting whichever alternative looked best after
seeing the support results would be post hoc tuning.

## Integrity checks

- All six registered trajectories were eligible and replayed exactly from
  their seed and archived SHA-256 digest.
- The audit retained 2,730 unique finite score readings, 45,864 covariance and
  component diagnostics, and 30 molecule-permutation readings.
- The revised-score identity replayed for every PCA8 and raw 100D reading; the
  maximum absolute residuals were `7.11e-15` and `1.14e-13`.
- Simultaneous molecule/beta relabeling changed all-coordinate scores by at
  most `8.67e-10`, below the registered `2e-7` tolerance.
- PCA was fitted on past states only. Every recorded support used exactly the
  declared number of explicit transition pairs.
- No detector module was imported by the audit runner or estimator module.
  Result manifests, provenance, and the final gate all record no label access,
  no outcome pilot, and no intervention.
- The first registration
  (`ccbcebdf3fd13c9420c94655cb79365d72d891f8dd22ddf532270d6e578de9c2`)
  remains preserved as a visible execution failure. A pandas `mode`-column
  access error occurred after computation but before any score table or gate
  was emitted. [Amendment 001](COVARIANCE_SUPPORT_AMENDMENT_001.md) changed
  only that access syntax, added a regression test, and created the amended
  source seal used here.

## Interpretation and next step

The earlier full-block positive hint cannot be treated as reliable evidence:
the same estimator family has a large finite-sample Gaussian-MI bias and can
change scientific comparisons merely as support grows. Reducing the state to
16 PCA coordinates fixes the algebraic rank problem but does not establish
sample-size stability at 64--192 pairs.

Trying PCA4, PCA16, covariance shrinkage, canonical-correlation MI, or
predictive log loss now and choosing the best-looking result would violate the
frozen stop rule. Any such attempt needs a new prospective, label-blind
stabilization protocol with independent development trajectories.

The higher-priority blocker remains the self-replicator definition. The
existing reconstruction produced 16.7% replication in its original controls
and 5.58% in the previous bridge, versus 88% reported by the paper. The next
paper-replication step is therefore to obtain or reconstruct the authors'
detector on shared fixtures, then establish exact online-estimator parity.
Until that happens, the negative replication and both diagnostic nulls remain
unchanged, and Phi-guided intervention stays locked.

## Generated evidence

The complete gitignored result directory is
`results/covariance-support-audit/`. Its principal files are:

- `SUMMARY.md` and `stability_gate.json`;
- `support_scores.csv` and `score_level_summary.csv`;
- `covariance_diagnostics.csv` and `stability_summary.csv`;
- `molecule_permutation_audit.csv`;
- `trace_manifest.csv` and six compressed trace checkpoints; and
- the exact amended registration, configuration, and runtime provenance.

The amended registration is retained separately in
`results/covariance-support-registration-amendment-001/registration.json`.
