# Arrival of Replicators: independent replication

This repository is a clean-room reconstruction of the simulations and analyses
in *Causal Architecture Dynamics Prior to Arrival of Self-replicators in a
Model of Catalytic Networks Relevant to Origin-of-Life* (arXiv:2607.28250v1).
It implements the stated GARD growth–fission experiment, the CLR/spectral
bipartition/local Gaussian causal-emergence estimator, compotype detection,
forecasting, and molecule-addition/deletion interventions.

This is a replication, not the authors' unreleased implementation. The code was
written from the public preprint and primary descriptions of standard GARD
and spectral minimum-information methods. No code by the preprint authors,
including their older projects, was inspected or reused.

## Preprint revision handoff

Agents editing the current integrated plastic-heredity manuscript should start
with [PREPRINT_AGENT_HANDOFF.md](PREPRINT_AGENT_HANDOFF.md). It records which
reviewer concerns are already addressed in the current manuscript, which later
analyses still need integration, the exact evidence status of each result, and
the claims that must not be made.

`REPLICATION_REPORT.md` below is the report for the initial white-room Phi-r
reconstruction. It predates the later F12, strict-eight, reviewer-control, and
mechanistic analyses and must not be used as a complete preprint evidence
summary.

## Arrivals formulation bridge

The additive [formulation-bridge protocol](FORMULATION_BRIDGE_PROTOCOL.md)
restarts one narrowly bounded observational question without changing the
original negative replication. It compares the original macro WMS and macro
MMI readings with the public nine-atom revised Phi-r and a provisional
full-dimensional revised measure on the same 12 fresh untreated trajectories.
It retains the existing replicator labels—and therefore the unresolved 16.7%
versus 88% detector discrepancy—and authorizes no Phi-guided intervention.

The completed pilot is reported in
[FORMULATION_BRIDGE_PILOT_REPORT.md](FORMULATION_BRIDGE_PILOT_REPORT.md). None
of the four frozen instruments passed both pilot screens, so this branch stops
without interventions pending author code or a separately justified new
instrument.

Validation and scientific generation are deliberately separate:

```bash
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/pytest
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate bridge-register \
  --output results/formulation-bridge-registration
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate bridge-pilot \
  --registration results/formulation-bridge-registration \
  --output results/formulation-bridge-pilot12
```

The pilot command refuses source drift from the generated registration. For a
long run, use `scripts/run-formulation-bridge-detached.sh`; status is available
from `scripts/status-formulation-bridge.sh`.

## Covariance-support diagnostic

New evidence that the provisional 100-coordinate statistic is highly
sample-size sensitive motivated a prospective, label-blind support audit. The
[frozen protocol](COVARIANCE_SUPPORT_PROTOCOL.md) compares identical
trajectories and endpoints at 64, 96, 128, 192, 256, 384, and 512 transition
pairs, reports covariance rank and ordinary score level, and tests one
prospectively selected PCA8 stabilization. It does not read replicator labels.

The completed [covariance-support report](COVARIANCE_SUPPORT_REPORT.md)
confirms severe support dependence in the raw full-block statistic. PCA8 made
the 32-dimensional joint covariance full rank at every support, but still
failed the frozen ordering and score-drift gate below 256 pairs. The registered
action is therefore to retain the numerical-instability null. No new outcome
pilot or intervention was launched, and the self-replicator-label discrepancy
remains unresolved.

```bash
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/python \
  -m aor_replication.covariance_support register \
  --output results/covariance-support-registration-amendment-001
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/python \
  -m aor_replication.covariance_support run \
  --registration results/covariance-support-registration-amendment-001 \
  --output results/covariance-support-audit
```

Long executions can use `scripts/run-covariance-support-detached.sh`; status is
available from `scripts/status-covariance-support.sh`.

## Current scientific status

The complete workflow is executable and checkpointed. The preprint leaves
several result-determining details unspecified, so exact numerical reproduction
is not identifiable from the paper alone. Most importantly, the standard GARD
cosine-composition cutoff of 0.95 produces substantially less self-replication
in this reconstruction than the 88% control probability reported in Table 1.
The code therefore registers 0.95 as its primary choice and emits threshold,
causal-estimator, and Poisson-time-scale sensitivity analyses. It does not tune
the primary analysis to the published outcome.

See [the reconstruction ledger](docs/REPLICATION_SPEC.md) for every reported,
inferred, and unresolved choice.

## Probe omitted settings

The separate genetic probe searches bounded, under-specified method choices
against manuscript aggregates, then evaluates its winner on untouched seeds.
It also checks the winner against the approximately 84.5% Φ prediction median
digitized from Figure 5. This is post hoc calibration and is never presented as
independent confirmation.

```bash
MPLCONFIGDIR=/tmp/aor-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/aor-replicate probe \
  --output results/probe --population 24 --ga-generations 10 \
  --calibration-runs 8 --holdout-runs 24 --workers 8 --objective full
```

The search is cached and resumes at generation checkpoints. See [the genetic
probe specification](docs/GENETIC_PROBE.md) for the figure inventory, genome,
fitness, guardrails, and checked-in overnight profile.

## Install and verify

Python 3.9 or newer is supported.

The third-party source PDF is not redistributed here. Download
[`arXiv:2607.28250v1`](https://arxiv.org/abs/2607.28250v1) from arXiv and place
it at `2607.28250v1.pdf`, or pass its local path with `--source-pdf`. The runner
records its SHA-256 in each result directory; the expected v1 hash is
`77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/pytest
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate smoke --output results/smoke
```

The smoke command retains the paper's 100 molecular types but reduces the run
and generation counts. It exercises simulation, interventions, statistics,
forecasting, sensitivity analysis, checkpoint reloads, and figure generation.

## Run the paper-scale experiment

```bash
MPLCONFIGDIR=/tmp/aor-mpl \
  .venv/bin/aor-replicate run --output results/main
```

Runs are deterministic from the stored configuration and base seed. Existing
matching checkpoints are resumed. A configuration mismatch is rejected rather
than silently mixing runs; use a new output directory for a changed model.

Useful staged commands are:

```bash
# Controls and sensitivity first
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate run \
  --output results/main --skip-interventions --skip-forecast

# Resume the same output and add interventions and forecasting
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate run \
  --output results/main
```

The preprint does not define the distribution used to score hypothetical
interventions. The primary `online_initial` estimator is leakage-free. The
explicit sensitivity alternative can be run into a separate directory:

```bash
MPLCONFIGDIR=/tmp/aor-mpl .venv/bin/aor-replicate run \
  --output results/matched-control-sensitivity --runs 20 \
  --intervention-estimator matched_control --skip-forecast --skip-sensitivity
```

## Outputs

Each result directory contains:

- `config.json` and `provenance.json`, including the source-PDF SHA-256;
- compressed raw traces and per-run analysis checkpoints;
- `run_metrics.csv`, aggregate tests, spike tests, and intervention trends;
- forecasting scores and matched treatment comparisons;
- primary Figures 2–6 and a sensitivity figure;
- `sensitivity/` with run-level and aggregate robustness tables; and
- `SUMMARY.md`, a human-readable result snapshot.

## Methodological boundaries

The primary causal quantity follows the equation displayed in the preprint,
the local whole-minus-sum (WMS) information value. An MMI-synergy variant is a
registered sensitivity analysis. The minimum-information bipartition is
approximated with a normalized spectral cut, following the paper's cited
spectral method. Interventions are calibrated at the first fission with enough
past observations, then use a fixed pre-intervention information model; this
prevents completed-trajectory leakage. These choices are all explicit because
the preprint does not specify the corresponding algorithms.

The implementation is licensed under the MIT License. The source preprint is
the authors' work, is not redistributed here, and is not relicensed by this
repository.
