#!/bin/sh
set -eu

audit_output="${1:-results/covariance-support-audit}"
pid_path="$audit_output/audit.pid"
log_path="$audit_output/audit.log"

if [ ! -f "$pid_path" ]; then
  printf 'no covariance-support PID file at %s\n' "$pid_path"
elif audit_pid="$(sed -n '1p' "$pid_path")" && kill -0 "$audit_pid" 2>/dev/null; then
  printf 'running as PID %s\n' "$audit_pid"
else
  printf 'not running; last recorded PID: %s\n' "${audit_pid:-unknown}"
fi

if [ -f "$log_path" ]; then
  tail -n 20 "$log_path"
fi

if [ -f "$audit_output/SUMMARY.md" ]; then
  printf '\nsummary: %s/SUMMARY.md\n' "$audit_output"
fi
