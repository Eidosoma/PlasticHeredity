#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIRECTORY" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
run_dir="$1"
launcher_dir="${run_dir}.launcher"

if [[ -f "${launcher_dir}/PID" ]]; then
  campaign_pid="$(sed -n '1p' "${launcher_dir}/PID")"
  if [[ "${campaign_pid}" =~ ^[0-9]+$ ]] && kill -0 "${campaign_pid}" 2>/dev/null; then
    echo "launcher: running (PID ${campaign_pid})"
  else
    echo "launcher: not running (last PID ${campaign_pid})"
  fi
else
  echo "launcher: no PID file"
fi

env PYTHONPATH="${project_dir}/src" "${project_dir}/.venv/bin/python" -m wagner_memory_cleanroom status --run "${run_dir}"

