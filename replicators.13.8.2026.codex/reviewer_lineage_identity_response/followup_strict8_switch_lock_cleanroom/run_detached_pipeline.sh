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
  echo "Detached strict-eight switch-lock campaign launched with PID ${child_pid}"
  echo "Status: ${artifact_dir}/STATUS.json"
  echo "Log: ${log_file}"
  exit 0
fi

printf '%s\n' "$$" >"${pid_file}"
cd "${codex_root}"
remaining="$(${python_bin} -m reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom.run_campaign remaining-hard)"
if [[ "${remaining}" -le 0 ]]; then
  printf '[%s] cumulative eight-hour limit already exhausted\n' "$(date -u +%FT%TZ)"
  exit 0
fi

printf '[%s] starting strict-eight switch-lock campaign with %s workers; hard remainder %ss\n' "$(date -u +%FT%TZ)" "${workers}" "${remaining}"
set +e
timeout --signal=TERM --kill-after=15s "${remaining}s" \
  "${python_bin}" -m reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom.run_campaign all --workers "${workers}"
exit_code="$?"
set -e
if [[ "${exit_code}" -eq 124 || "${exit_code}" -eq 137 ]]; then
  "${python_bin}" -m reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom.run_campaign mark-hard-stop
  printf '[%s] hard watchdog stopped campaign\n' "$(date -u +%FT%TZ)"
  exit 0
fi
printf '[%s] campaign process exited with status %s\n' "$(date -u +%FT%TZ)" "${exit_code}"
exit "${exit_code}"

