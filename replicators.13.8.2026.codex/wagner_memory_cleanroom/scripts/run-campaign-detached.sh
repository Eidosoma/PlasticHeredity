#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIRECTORY [full|quick|smoke]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
run_dir="$1"
profile="${2:-full}"
python_bin="${project_dir}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing project environment: ${python_bin}" >&2
  exit 1
fi
if [[ "${profile}" != "full" && "${profile}" != "quick" && "${profile}" != "smoke" ]]; then
  echo "profile must be full, quick, or smoke" >&2
  exit 2
fi

launcher_dir="${run_dir}.launcher"
mkdir -p -- "${launcher_dir}"
pid_file="${launcher_dir}/PID"
if [[ -f "${pid_file}" ]]; then
  previous_pid="$(sed -n '1p' "${pid_file}")"
  if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "campaign already running as PID ${previous_pid}" >&2
    exit 1
  fi
fi

session_name="wagnermemory_${$}_${RANDOM}"
log_file="${launcher_dir}/campaign.log"
command_text="exec env PYTHONPATH='${project_dir}/src' CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '${python_bin}' -m wagner_memory_cleanroom campaign --run '${run_dir}' --profile '${profile}' >'${log_file}' 2>&1"
tmux new-session -d -s "${session_name}" "${command_text}"
campaign_pid="$(tmux display-message -p -t "${session_name}" '#{pane_pid}')"
printf '%s\n' "${campaign_pid}" >"${pid_file}"
printf '%s\n' "${session_name}" >"${launcher_dir}/TMUX_SESSION"
printf '{"pid":%s,"session":"%s","run":"%s","profile":"%s","log":"%s"}\n' \
  "${campaign_pid}" "${session_name}" "${run_dir}" "${profile}" "${log_file}" \
  >"${launcher_dir}/launcher.json"

echo "launched PID ${campaign_pid} on GPUs 0 and 1"
echo "tmux session: ${session_name}"
echo "log: ${log_file}"

