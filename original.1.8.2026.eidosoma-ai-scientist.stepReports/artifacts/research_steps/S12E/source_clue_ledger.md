# S12E source clue ledger

## Top summary

- **Research step ID:** `E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0`.
- **Completion status:** Phase-0 archaeology complete; scientific development outcomes remained unopened when this ledger was frozen.
- **Artifacts written:** Phase-0 preregistration, source snapshot, paper fingerprints and figure measurements, ambiguity ledger, four implementation registries, immutable-prior baseline, and validation records under `/artifacts/research_steps/S12E/`.
- **Validation result:** Source identities and paper payload passed their pinned hash/commit checks; the arXiv source response is PDF-only and byte-identical to the S03 copy.
- **Outcome classification:** `UNDERDETERMINED` at Phase 0; no engine, label, emergence, or intervention outcome has been observed.
- **Caveats or blockers:** No public GARD-paper code or TeX source was found. Public GRN and PhiRL code are lineage clues, not the unavailable paper implementation.
- **Recommended next action:** Commit and push the locked S12E method, then run Phase 1 without computing replication labels or information metrics.

## Paper and archive

- arXiv `2607.28250v1` source endpoint returned a PDF with SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`; it has 18 pages, zero embedded files, empty Comments metadata, and Word/Acrobat producer metadata. No TeX comments or original figure filenames are available.
- The supplied raster figures have no embedded PNG metadata. Their hashes, pixel sizes, and manually read axis ranges are frozen in `source_snapshot_manifest.json` and `paper_figure_measurements.csv`.
- Methods explicitly state distinct-type initialization, Poisson vector updates, binomial fission, and the 100/40/80/100/1000/-4/4 tuple. This conflicts materially with public historical v10 eventwise growth, with-replacement initialization, unbounded generations, and fixed-size split behavior.

## Historical GARD

- `tgs_parameters_v10.m` fixes `Kf=1e-2`, `Kb=1e-4`, lognormal `mu=-4`, `sigma=4`, `hthresh=0.9`, `ks=1:10`, and ten replicas.
- `tgs_grow_v10.m` draws one normalized join/loss event per loop, not a simultaneous Poisson vector.
- `tgs_agard_v10.m` supplies uniform `rho`, initializes counts with replacement, and follows the first split output.
- `tgs_split_v10.m` selects a fixed half without replacement and can discard one odd molecule; it is not binomial componentwise fission.
- `tgs_nondrift.m` technique 1 labels adjacent-composition cosine scores averaged over incoming/outgoing transitions; `tgs_acluster.m` applies k-means only to non-drift generations.

## Public information-theory lineage

- IIGR commit `7c1c22f` corrects the `local_phi_r` overwrite bug. Its `main.py` separately assigns `integrated = local_phi_r(...)` and `emergence = synergy + causation`, uses lag-one Gaussian MI, a noise-connected unnormalized Fiedler partition, and completed supplied arrays for local Gaussian fits.
- PhiRL commit `a6d1d0d` preserves that scalar split, filters dimensions at standard deviation `1e-8`, and descends from the `9030b598` trace-scaled covariance-regularization commit.
- The 2026 public `BreakingGRNMemories` lineage uses `MEASURES=['emergence']`, the same synergy-plus-causation atoms, `noise=True`, and public GRN preprocessing. One file also replaces nonfinite values with zero. Because that replacement is not authorized among M1-M4, it remains a clue rather than a branch.
- Public GitHub inspection found 34 repositories for `pigozzif`, no exact-title repository, no public fork of IIGR or PhiRL, and only the pinned `master` branches. This is an absence result, not proof that code does not exist elsewhere.

## Frozen forensic consequence

S12E must decide upstream GARD and label identity without any emergence value. Metric branches remain exactly M1-M4; no GRN nonfinite-replacement branch or post-outcome method is permitted. Full supplied-array values are retrospective, while prefix refits are the only prospective source reconstruction.
