#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# This script intentionally does not register Stage 2. Registration is a
# separate, reviewed command after a robust Stage-1 decision.
.venv/bin/python -m reviewer_motif_channel_replication.run stage2 --workers "${MOTIF_WORKERS:-8}"
.venv/bin/python -m reviewer_motif_channel_replication.run stage2-report
