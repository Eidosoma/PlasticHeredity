#!/usr/bin/env bash
set -euo pipefail

workers="${1:-16}"
mode="${2:-parent}"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codex_root="$(cd "${task_dir}/../.." && pwd)"
artifact_dir="${task_dir}/artifacts"
log_file="${artifact_dir}/detached_pipeline.log"
pid_file="${artifact_dir}/detached_pipeline.pid"
watchdog_file="${artifact_dir}/watchdog.pid"
status_file="${artifact_dir}/STATUS.json"
python_bin="${PYTHON_BIN:-${codex_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not executable: ${python_bin}" >&2
  exit 1
fi

mkdir -p "${artifact_dir}"

if [[ "${mode}" != "--child" ]]; then
  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(tr -cd '0-9' < "${pid_file}")"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Campaign already running with PID ${existing_pid}"
      exit 0
    fi
  fi
  touch "${log_file}"
  : >"${pid_file}"
  setsid --fork "${script_path}" "${workers}" --child >>"${log_file}" 2>&1 </dev/null
  child_pid=""
  for _ in $(seq 1 50); do
    if [[ -s "${pid_file}" ]]; then
      child_pid="$(tr -cd '0-9' < "${pid_file}")"
      [[ -n "${child_pid}" ]] && break
    fi
    sleep 0.1
  done
  if [[ -z "${child_pid}" ]]; then
    echo "Detached child did not publish a PID" >&2
    exit 1
  fi
  echo "Detached clean-room carrier campaign launched with PID ${child_pid}"
  echo "Status: ${status_file}"
  echo "Log: ${log_file}"
  exit 0
fi

printf '%s\n' "$$" >"${pid_file}"
cd "${codex_root}"
remaining="$(${python_bin} -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign remaining-hard)"
if [[ "${remaining}" -le 0 ]]; then
  printf '[%s] cumulative eight-hour limit already exhausted\n' "$(date -u +%FT%TZ)"
  exit 0
fi

campaign_group="$$"
(
  sleep "${remaining}"
  printf '[%s] hard wall-time watchdog terminating process group %s\n' "$(date -u +%FT%TZ)" "${campaign_group}" >>"${log_file}"
  kill -TERM -- "-${campaign_group}" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-${campaign_group}" 2>/dev/null || true
) &
watchdog_pid="$!"
printf '%s\n' "${watchdog_pid}" >"${watchdog_file}"
trap 'kill "${watchdog_pid}" 2>/dev/null || true' EXIT

printf '[%s] starting clean-room carrier campaign with %s workers; cumulative hard remainder %ss\n' "$(date -u +%FT%TZ)" "${workers}" "${remaining}"
"${python_bin}" -m reviewer_lineage_identity_response.followup_carrier_cleanroom.run_campaign all --workers "${workers}"
printf '[%s] clean-room carrier campaign process exited\n' "$(date -u +%FT%TZ)"
