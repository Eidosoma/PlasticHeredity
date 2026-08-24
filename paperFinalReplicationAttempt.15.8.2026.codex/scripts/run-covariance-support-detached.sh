#!/bin/sh
set -eu

audit_output="${1:-results/covariance-support-audit}"
registration="${2:-results/covariance-support-registration}"
log_path="$audit_output/audit.log"
pid_path="$audit_output/audit.pid"

mkdir -p "$audit_output"
if [ -f "$pid_path" ]; then
  prior_pid="$(sed -n '1p' "$pid_path")"
  if [ -n "$prior_pid" ] && kill -0 "$prior_pid" 2>/dev/null; then
    printf 'covariance-support audit is already running as PID %s\n' "$prior_pid"
    exit 1
  fi
fi

export MPLCONFIGDIR=/tmp/aor-mpl
export PYTHONPYCACHEPREFIX=/tmp/aor-support-pycache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

nohup .venv/bin/python -m aor_replication.covariance_support run \
  --registration "$registration" \
  --output "$audit_output" \
  >"$log_path" 2>&1 </dev/null &
audit_pid=$!
printf '%s\n' "$audit_pid" >"$pid_path"
printf 'started covariance-support audit as PID %s; log: %s\n' "$audit_pid" "$log_path"
