# Reviewer cross-substrate plastic-heredity response

This isolated package tests whether the operational break-and-renewal process
transfers from compositional GARD assemblies to either of two cellular-
automaton substrates: a stochastic spatial hypercycle protocell and a
published Evoloop rule in a stochastic ecology.

The package never reads the unpublished hypothesis-generating materials.  All
writes are confined to `reviewer_cross_substrate_response/artifacts/`.

## Staged workflow

Run from the repository root:

```bash
reviewer_cross_substrate_response/run_developmental_pilot.sh 14
```

That resumable launcher is exactly the following staged workflow:

```bash
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment prepare
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment validate
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment calibrate --model all --workers 14
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment pilot --model all --workers 14
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment pilot-report
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment status
```

The registered workflow stops after `pilot-report`.  The following commands
are implemented but must not be invoked without a later explicit research
instruction:

```bash
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment register-confirmation
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment confirm --model eligible --workers 14
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment report
.venv/bin/python -m reviewer_cross_substrate_response.run_experiment verify --full-replay
```

Use `--profile smoke` only for non-scientific mechanics and I/O validation.
Smoke outputs can never satisfy a pilot or confirmation gate.

Each parent is compared with a size-matched child from a different
independently seeded world block.  Parent and child rasters are retained in
compressed sidecars; tables, manifests, source commitments, reports, and all
scientifically used raw files carry SHA-256 integrity records.  Every
mechanics seed, calibration block, pilot block, and confirmation block is an
atomic resumable checkpoint.  Futures run
for 16 boundaries so F16 can be reported, but the primary event and pilot
eligibility use only the first 12.

Run the isolated test suite with:

```bash
.venv/bin/python -m pytest -q reviewer_cross_substrate_response/tests
```

## Evidence boundary

A passing model supports the preregistered operational process in that tested
CA contract.  It does not establish universality, life, biological memory,
Phi/PhiID, real chemistry, or a shared GARD mechanism.  The two CA models are
never pooled; one cannot conceal the other's result.
