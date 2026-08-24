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
python_bin="${project_dir}/../wagner_memory_cleanroom/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing frozen Python environment: ${python_bin}" >&2
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

log_file="${launcher_dir}/campaign.log"
nohup setsid env \
  PYTHONPATH="${project_dir}/src" \
  CUDA_VISIBLE_DEVICES=0,1 \
  JAX_PLATFORMS=cuda \
  WAGNER_FORCE_GPU=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "${python_bin}" -m wagner_memory_cleanroom_v2 campaign \
  --run "${run_dir}" --profile "${profile}" \
  >"${log_file}" 2>&1 </dev/null &
campaign_pid="$!"
printf '%s\n' "${campaign_pid}" >"${pid_file}"
printf '{"pid":%s,"run":"%s","profile":"%s","log":"%s"}\n' \
  "${campaign_pid}" "${run_dir}" "${profile}" "${log_file}" \
  >"${launcher_dir}/launcher.json"

echo "launched detached PID ${campaign_pid} with CUDA devices 0 and 1"
echo "log: ${log_file}"
