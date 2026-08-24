#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT [WORKERS=12] [PROFILE=full]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
output="$1"
workers="${2:-12}"
profile="${3:-full}"
launcher_dir="${output}.launcher"
pid_file="${launcher_dir}/PID"
log_file="${launcher_dir}/campaign.log"
session_file="${launcher_dir}/TMUX_SESSION"

if [[ "${profile}" != "full" && "${profile}" != "smoke" ]]; then
  echo "profile must be full or smoke" >&2
  exit 2
fi
if [[ ! "${workers}" =~ ^[0-9]+$ ]] || (( workers < 1 || workers > 12 )); then
  echo "workers must be an integer from 1 through 12" >&2
  exit 2
fi
if [[ -f "${pid_file}" ]]; then
  previous_pid="$(sed -n '1p' "${pid_file}")"
  if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "campaign already running as PID ${previous_pid}" >&2
    exit 1
  fi
fi

mkdir -p -- "${launcher_dir}"
python_bin="${project_dir}/../.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "missing Python environment: ${python_bin}" >&2
  exit 1
fi

session_name="wagner_${$}_${RANDOM}"
command_text="exec env PYTHONPATH='${project_dir}/src' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '${python_bin}' -m wagner_cleanroom campaign --output '${output}' --workers '${workers}' --profile '${profile}' >'${log_file}' 2>&1"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${session_name}" "${command_text}"
  campaign_pid="$(tmux display-message -p -t "${session_name}" '#{pane_pid}')"
  printf '%s\n' "${session_name}" >"${session_file}"
else
  setsid -f sh -c "${command_text}"
  campaign_pid=""
  for _ in 1 2 3 4 5; do
    campaign_pid="$(pgrep -n -f "wagner_cleanroom campaign --output ${output}" || true)"
    [[ -n "${campaign_pid}" ]] && break
  done
  if [[ -z "${campaign_pid}" ]]; then
    echo "detached process did not start" >&2
    exit 1
  fi
fi
printf '%s\n' "${campaign_pid}" >"${pid_file}"
printf '{"pid":%s,"session":"%s","output":"%s","workers":%s,"profile":"%s","log":"%s"}\n' \
  "${campaign_pid}" "${session_name}" "${output}" "${workers}" "${profile}" "${log_file}" \
  >"${launcher_dir}/launcher.json"

echo "launched PID ${campaign_pid}"
echo "tmux session: ${session_name}"
echo "log: ${log_file}"
