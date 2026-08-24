# E01/S19-L07 — Exploratory 88%-versus-98% occupancy-setting search

## Concise top summary

- **Research step ID:** S19-L07 (`E01-S19-L07-OCCUPANCY-SETTING-SEARCH-v1.0.0`)
- **Completion status:** COMPLETE; mandatory post-L07 human-review boundary active
- **Artifacts written:** six chronologically locked round packages; 228-setting specification/attempt ledger; 29,200 trajectory fingerprints; occupancy, clock/boundary, threshold, exposure, fresh-seed, seed, runtime, validation, figure, failure, classification, status, provenance, and hash manifests; this canonical report and one-page handoff
- **Validation result:** PASS — 1,584/1,584 immutable prior files; 200 trajectories and 180,635 frozen clock/label rows; 200/200 exact S13Y simulator replays; zero fresh-seed hash overlap; all 228 settings accounted for; 20/20 independently regenerated scientific table components exact; repository pushed and clean
- **Outcome classification:** `EXPLORATORY_PAPER_MATCH` — occupancy only; the sole human-waived target passed in both branches through two nonunique mechanisms
- **Caveats or blockers:** occupancy alone cannot identify author code; the search is adaptive; exact paper denominator, exposure, clock, and clustering semantics remain unavailable; high exposure shortens trajectories markedly; threshold matching is outcome-directed; other temporal/predictive/causal gates were waived, not passed
- **Lay summary:** We can now reproduce approximately 88% in both branches. The most paper-coherent route is to count inheritance at fission boundaries (`87.72%/87.77%`) rather than every smooth molecular update. A second route uses the unchanged molecular `H>0.9` label but a previously undocumented Poisson exposure near `h=2.875` (`88.38%/88.03%` on fresh matrices). These are reproducible explanations, not proof of the authors' exact settings.
- **Recommended next action:** Human review. If another loop is authorized, freeze one untouched discriminating test between the fission-boundary denominator and high-exposure mechanisms using trajectory-length/persistence and source fingerprints. Do not start L08, S20, E02, author contact, or report-bundle generation automatically.

## Frozen question and waiver

L07 asked only which paper- or source-plausible settings can move self-replicator occupancy from the frozen adjacent-molecular value near 98% toward the paper's approximately 88%. The human explicitly made occupancy proximity the sole scientific success target, did not require exact 88%, and waived every other paper-fingerprint and promotion gate. The waiver did **not** relax complete attempt logging, candidate separation, preservation of prior evidence, numerical validity, provenance, deterministic regeneration, or artifact integrity.

No L07 result is confirmatory. No result is labelled author-code identity. No emergence, prediction, intervention, or causal-control result was used to select settings.

## Lay interpretation

The 98% value is not an unavoidable property of these simulations. It is produced by asking, at every molecular observation, whether composition is very similar to its immediate predecessor. Three changes can lower it toward 88%:

1. Count the parent-to-selected-daughter inheritance event once per fission. This retains strict `H>0.9`, the frozen trajectories, and the original exposures, and gives `87.72%/87.77%`.
2. Retain every molecular observation and strict `H>0.9`, but use larger Poisson batches. A stable region near `h=2.875` gives `88.11%/88.10%` in the bracketing run and `88.38%/88.03%` on fresh matrices.
3. Raise or change the similarity transcription. Thresholds near `0.97` can also give about 88%, but this was outcome-directed and is not the historical source's `0.9` setting.

The first explanation better preserves the paper's generational inheritance language and paper-scale persistence. The second proves that an omitted simulator parameter can reproduce occupancy, but it also shortens the trajectories, so it is not by itself a coherent reconstruction of Table 1.

![Occupancy reconstruction summary](occupancy_search_summary.png)

**Figure 1.** Exploratory occupancy paths. Left: all-molecular strict incoming `H>0.9` occupancy falls as Poisson exposure grows; stars are the fresh-seed `h=2.875` validation. Right: multiple settings reach the paper's approximate occupancy band, demonstrating nonidentifiability from occupancy alone.

## Inputs and immutable context

- Original paper attachment: `pdf-markdown.md`, arXiv version `2607.28250v1`.
- Frozen S13Y data: 100 shared catalytic matrices, 200 candidate-2/candidate-3 trajectories, selected molecular clocks, adjacent-H arrays, and exact `H>0.9` labels.
- Candidate 2 baseline: `h=0.6031526490073492`, first-daughter continuation.
- Candidate 3 baseline: `h=0.5613315384859516`, random-nonempty daughter continuation.
- Historical GARD source commit: `86dff6320d5ae91b4e831471079ff46749b14df9`; source retained in cache only because no license file was found.
- Immutable baseline: all S01–S18, V1/V2, S19-L01–L06R, classifications, failures, and the S17 waiver.

The paper directly reports `88±3%` for control probability, describes recurring composition-space attractors inherited across generations, and specifies Poisson updates. It does not uniquely state the probability denominator/object or Poisson exposure duration. Historical GARD v10 uses generation-level compositions and a drift/non-drift `H=0.9` parameter; another clustering helper uses `0.95`. Those sources are lineage evidence, not target-paper code.

## Detailed methods and chronological search

### Numerical gate and amendment

The first replay attempt opened no L07 occupancy. All 180,635 frozen boolean labels agreed, but two mathematically equivalent float64 normalization orders were not bit-identical (maximum absolute `8.881784197001252e-16`). A separately committed value-preserving amendment required identical finite masks and labels plus absolute and relative error `<=1e-12` and ULP distance `<=8`. The amended replay passed all 200 trajectories: maximum absolute `8.881784197001252e-16`, relative `8.946934114690078e-16`, and `8` ULP.

### Round inventory

| Round | Frozen question | Registered settings | Outcome |
| --- | --- | ---: | --- |
| R01 | Clock, boundary object, strict/`>=`, and fixed projections | 16 | Parent→selected-daughter strict `H>0.9` boundary occupancy `0.8772/0.8777` |
| R02 | Fixed threshold/transcription family | 48 | Several `~0.97` settings enter the band; nonunique and outcome-directed |
| R03 | Coarse Poisson exposure and exact frozen replay | 100 | All-molecular occupancy declines from about 0.986 at `h=0.45` to about 0.956 at `h=1.25`; boundary occupancy remains near 0.88 |
| R04 | Missing-exposure diagnostic through `h=4` | 36 | All-molecular match appears near `h=3`; boundary match remains broad |
| R05 | Local bracket `h=2.75–3.25` | 20 | `h=2.875` gives `0.881090/0.880995` all-molecular occupancy |
| R06 | Fresh 100-matrix seed set | 8 | `h=2.875` validates at `0.883845/0.880294`; zero matrix/initial overlap |

Every setting was serialized before its result in `setting_registry.parquet` and retained after outcome access in `chronological_attempt_ledger.parquet`. Unsuccessful settings were not deleted or reordered. Candidate/branch results are separate; pooled selection was not used.

### Simulation contract

The exposure rounds retained the S13Y simulation kernel, 100 molecule types, 100 fissions, fixed shared matrices/initial states per round, overshoot handling that trims only excess newly joined molecules, and the two continuation rules. CPU float64 was authoritative; no GPU was used. R06 used a new domain-separated 256-bit root and required no beta or initial-state hash overlap with S13Y.

### Occupancy and uncertainty

For each trajectory, occupancy is the fraction of eligible analysis units labelled positive. Each catalytic matrix is one inferential unit. Reported intervals are deterministic 4,096-replicate matrix-bootstrap 95% intervals. Pair ranking minimizes the maximum candidate/branch absolute error from `0.88`, then mean error. The approximate target band is `0.88±0.03`, solely for L07.

## Results

### 1. Fission-boundary denominator matches 88% without changing trajectories

| Definition on frozen original-exposure trajectories | Candidate 2 | Candidate 3 | Key scope |
| --- | ---: | ---: | --- |
| Adjacent molecular incoming `H>0.9` | 0.980891 | 0.982657 | Every selected molecular observation |
| Parent→selected-daughter boundary, strict `H>0.9` | 0.877200 | 0.877700 | 100 fission events/run |
| Boundary label projected to following eligible interval | 0.864980 | 0.865323 | Molecular time, first prefix excluded |

The boundary-only result is the closest paper/source-plausible match because it changes neither threshold nor simulator and directly measures inheritance through fission. It also remains near 88% across every tested exposure and continuation rule; therefore its match is not an `h`-selected accident.

The denominator matters. Boundary-only persistence is only the number of positive fissions (`87.72/87.77`), so it is not directly comparable to the paper's molecular-step persistence. Projecting those boundary decisions onto molecular intervals gives persistence `733.86/779.03`, close in scale to the paper control value 716, but occupancy is about 86.5%, consistency is `0.914/0.915`, and onset is `15.36/18.46` steps. Those descriptive mismatches do not fail the human-waived L07 target, but they prevent an exact label claim.

### 2. A missing Poisson exposure also matches 88%

The coarse sweep showed a smooth occupancy decrease under the all-molecular strict label. Values up to the earlier S12F ceiling `h=1.25` remained at 95.5–95.6%. Extending the paper-undocumented exposure reached the target near `h=3`; the fixed bracket selected `h=2.875` for cross-branch occupancy proximity.

| Dataset | First-daughter branch | Random-nonempty branch | Maximum error from 0.88 |
| --- | ---: | ---: | ---: |
| R05 shared S13Y matrices, `h=2.875` | 0.881090 | 0.880995 | 0.001090 |
| R06 fresh matrices, `h=2.875` | 0.883845 | 0.880294 | 0.003845 |

The fresh validation passed the seed firewall: 100 new catalytic-matrix hashes and 100 new initial-state hashes, with zero overlap. This makes `h=2.875` a reproducible occupancy-only lead.

It is not a complete Table-1 reconstruction. On fresh matrices, mean selected-clock length was `321.43/322.86` and persistence only `283.51/283.64`; onset remained `3.41/3.44` and consistency `0.084/0.086`. Large Poisson batches also produced substantially larger overshoot before the frozen trim rule. Thus high exposure reaches 88% partly by reducing the number and smoothness of recorded updates, and the paper gives no evidence that `h=2.875` was used.

### 3. Threshold/transcription routes are nonunique

The best cross-candidate incoming-clock threshold pair was C0 strict `H>0.9725` (`0.876139/0.886973`). A historical-technique-like two-neighbor average on C0 at `H>0.97` gave occupancy `0.871142/0.880144`, persistence `673.08/728.58`, and consistency `0.464/0.466`. That consistency is directionally closer to the paper's 0.38 than the adjacent incoming label, but onset remained only `4.76/4.42` steps.

These results confirm the user's prior observation: a threshold can force occupancy toward 88% without recovering the temporal state. Moreover, `0.97` was searched after the target was known, conflicts with the historical `0.9` parameter, and is not promoted as an author setting.

### 4. What most likely explains 88% versus 98%

The evidence now favors a **measurement-object/denominator mismatch** as the most parsimonious source-grounded explanation: the paper discusses recurrence and inheritance across generations, while the frozen 98% label evaluates smoothness at every molecular observation. Measuring strict similarity at parent→daughter fission events immediately produces 87.7% under both original candidate pipelines. This inference is stronger than a bare numerical match but remains an inference; the paper also describes recurrence relative to a recurring attractor, which a single parent→daughter comparison does not fully implement.

An omitted exposure near `h=2.875` is a genuine alternative explanation for occupancy alone. It is less coherent with molecular-step persistence and has no recovered source identity. Occupancy therefore does not identify one unique author pipeline.

## Validation

| Check | Result |
| --- | --- |
| Immutable S01–S18, V1/V2, L01–L06R baseline | PASS, 1,584/1,584 files |
| Frozen trajectory/clock/H/label replay | PASS, 200 trajectories and 180,635 rows; exact labels |
| Numerical amendment | PASS, masks/labels exact; abs/rel `<=1e-12`, ULP `<=8` |
| Exact frozen S13Y simulator replay in R03 | PASS, 200/200 trajectory hashes |
| Setting accounting | PASS, 228/228 registered settings complete |
| Fresh R06 seed firewall | PASS, 0 beta and 0 initial-state overlaps |
| Independent regeneration | PASS, 20/20 scientific components exact |
| Repository | PASS, scientific regeneration at `699bdfa08696a2a5d5e4f83e441f60884303e2b9`; reporting finalizer at `c76d7c313ac993468191875d10dcb3be4ed30b82`; branch clean and pushed at handoff |
| Scientific failures | None after replay amendment; initial failed attempt retained as `L07-F001` |

The NumPy warnings recorded during fingerprints occur when consecutive-label Pearson correlation is undefined for constant sequences; the code serializes those values as status-bearing nulls. They do not affect occupancy or any success decision.

## Runtime, commands, and dependencies

Core commands:

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l07.py
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r01 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r02 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r03 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r04 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r04 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r05 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r05 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py prepare-r06 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py run-r06 --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l07.py regenerate --workers 8
```

The six outcome rounds used `0.1735` worker CPU-hours plus `0.0617` coordinator CPU-hours; regeneration took `5.31` wall-minutes. Eight workers and one numerical-library thread per worker were used. GPU hours were zero. Dependencies were the preinstalled Python 3.13, NumPy, pandas, SciPy, PyArrow, PyYAML, and Matplotlib stack; no package was installed.

Pushed commits, in order: `a3ef41d` preregistration; `edf9851` numerical amendment; `13efb7e` exposure extension; `d58becf` local bracket; `d6f22bb` fresh validation; `699bdfa` exact regeneration audit; `c76d7c3` deterministic reporting finalizer.

## Provenance and artifact map

- Governing locks: `preregistration.yaml`, `method_lock.json`, `round_registry.yaml`, and `round_R04/R05/R06_lock.yaml`.
- Complete setting history: `setting_registry.parquet`, `chronological_attempt_ledger.parquet`, and `specification_ledger.parquet`.
- Primary machine result: `occupancy_results.parquet`; full per-trajectory evidence: `trajectory_fingerprint_results.parquet`.
- Mechanism-specific evidence: `clock_boundary_results.parquet`, `threshold_sensitivity_results.parquet`, and `simulator_setting_results.parquet`.
- Fresh validation: `R06_FRESH_SEED_VALIDATION_*`, `R06_seed_firewall.json`, and `strongest_setting_validation.parquet`.
- Exact regeneration: `regeneration_results.parquet` and `regeneration_validation.json`.
- Integrity: `immutable_prior_postcheck.json`, `storage_validation.json`, and `artifact_manifest.json`.
- Reproducible repository code: `src/e01_s19_occupancy_search/`, `scripts/e01/run_s19_l07.py`, this finalizer, configs, and tests on branch `eidosoma/groups/42`.

## Caveats and limitations

1. The search was explicitly adaptive and used a known 88% target. An occupancy match is exploratory.
2. The paper's 88% appears in Table 1's control intervention context; equivalence to the frozen S13Y baseline dataset is plausible but not established by author code.
3. Boundary-only occupancy changes the unit from molecular steps to fissions. Projection restores molecular time but requires an undocumented rule.
4. High exposure is paper-plausible only because exposure duration is omitted; no source identifies `h=2.875`, and high batches alter trajectory length and overshoot.
5. Threshold values near 0.97 are outcome-selected and not a substitute for the historical/source `0.9` contract.
6. None of these routes reconstructs the full recurring-attractor definition uniquely. Full-run compotype and recurrence alternatives remain historical exploratory evidence from L02–L06R.
7. Other temporal fingerprints were descriptive by human waiver. Prediction and causal-control non-support from S16–S18 is unchanged.
8. No authors were contacted, and public code without an identified license was not redistributed.

## Outcome and mandatory handoff

L07 succeeds on its sole authorized target: there are reproducible, candidate-consistent settings close to 88%. The result is an `EXPLORATORY_PAPER_MATCH` limited to occupancy. The strongest paper-coherent lead is the strict parent→selected-daughter fission-boundary denominator; the strongest all-molecular alternative is an omitted fixed exposure near `h=2.875`. Neither is confirmed as the authors' implementation.

All L07 artifacts are frozen for human review. No L08, S20, E02, author contact, or report-bundle generation has been activated.
