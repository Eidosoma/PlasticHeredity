#!/bin/sh
set -eu

probe_output="${1:-results/probe-overnight-full}"

# A process worker owns each genome. Giving BLAS another thread pool inside
# every worker caused severe oversubscription in the paper-dimensional pilot.
export MPLCONFIGDIR=/tmp/aor-mpl
export PYTHONPYCACHEPREFIX=/tmp/aor-probe-pycache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec .venv/bin/aor-replicate probe \
  --output "$probe_output" \
  --population 64 \
  --ga-generations 40 \
  --calibration-runs 64 \
  --holdout-runs 256 \
  --workers 8 \
  --objective full \
  --gard-generations 100
