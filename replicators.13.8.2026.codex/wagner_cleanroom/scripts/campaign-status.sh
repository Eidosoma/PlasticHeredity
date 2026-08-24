#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT" >&2
  exit 2
fi

output="$1"
launcher_dir="${output}.launcher"
pid_file="${launcher_dir}/PID"
session_file="${launcher_dir}/TMUX_SESSION"

if [[ -f "${pid_file}" ]]; then
  campaign_pid="$(sed -n '1p' "${pid_file}")"
  if [[ "${campaign_pid}" =~ ^[0-9]+$ ]] && kill -0 "${campaign_pid}" 2>/dev/null; then
    echo "launcher: running (PID ${campaign_pid})"
  else
    echo "launcher: not running (last PID ${campaign_pid})"
  fi
else
  echo "launcher: no PID file"
fi
if [[ -f "${session_file}" ]]; then
  session_name="$(sed -n '1p' "${session_file}")"
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux: attached to session ${session_name}"
  else
    echo "tmux: session ${session_name} has exited"
  fi
fi

if [[ -f "${output}/STATUS.json" ]]; then
  echo "scientific status:"
  sed -n '1,200p' "${output}/STATUS.json"
else
  echo "scientific status: not created"
fi
