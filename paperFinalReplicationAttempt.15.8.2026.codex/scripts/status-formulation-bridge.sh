#!/bin/sh
set -eu

pilot_output="${1:-results/formulation-bridge-pilot12}"
pid_path="$pilot_output/pilot.pid"
log_path="$pilot_output/pilot.log"

if [ ! -f "$pid_path" ]; then
  printf 'no formulation-bridge PID file at %s\n' "$pid_path"
elif pilot_pid="$(sed -n '1p' "$pid_path")" && kill -0 "$pilot_pid" 2>/dev/null; then
  printf 'running as PID %s\n' "$pilot_pid"
else
  printf 'not running; last recorded PID: %s\n' "${pilot_pid:-unknown}"
fi

if [ -f "$log_path" ]; then
  tail -n 20 "$log_path"
fi

if [ -f "$pilot_output/SUMMARY.md" ]; then
  printf '\nsummary: %s/SUMMARY.md\n' "$pilot_output"
fi
