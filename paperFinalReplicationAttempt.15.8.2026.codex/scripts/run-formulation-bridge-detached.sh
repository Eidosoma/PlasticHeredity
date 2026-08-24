#!/bin/sh
set -eu

pilot_output="${1:-results/formulation-bridge-pilot12}"
registration="${2:-results/formulation-bridge-registration}"
log_path="$pilot_output/pilot.log"
pid_path="$pilot_output/pilot.pid"

mkdir -p "$pilot_output"
if [ -f "$pid_path" ]; then
  prior_pid="$(sed -n '1p' "$pid_path")"
  if [ -n "$prior_pid" ] && kill -0 "$prior_pid" 2>/dev/null; then
    printf 'formulation bridge is already running as PID %s\n' "$prior_pid"
    exit 1
  fi
fi

export MPLCONFIGDIR=/tmp/aor-mpl
export PYTHONPYCACHEPREFIX=/tmp/aor-bridge-pycache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

nohup .venv/bin/aor-replicate bridge-pilot \
  --registration "$registration" \
  --output "$pilot_output" \
  >"$log_path" 2>&1 </dev/null &
pilot_pid=$!
printf '%s\n' "$pilot_pid" >"$pid_path"
printf 'started formulation bridge as PID %s; log: %s\n' "$pilot_pid" "$log_path"
