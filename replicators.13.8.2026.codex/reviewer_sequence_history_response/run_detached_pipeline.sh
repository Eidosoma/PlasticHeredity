#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${TASK_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
LOG_FILE="${TASK_DIR}/artifacts/detached_pipeline.log"
STATUS_FILE="${TASK_DIR}/artifacts/detached_pipeline.status"

mkdir -p "${TASK_DIR}/artifacts"
cd "${REPO_DIR}"
printf 'running\n' >"${STATUS_FILE}"
trap 'code=$?; if [[ ${code} -ne 0 ]]; then printf "failed:%s\n" "${code}" >"${STATUS_FILE}"; fi' EXIT

{
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py prepare
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py replay --dataset all --workers "${SEQUENCE_WORKERS:-12}"
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py fit
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py analyze
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py report
  "${PYTHON_BIN}" reviewer_sequence_history_response/run_analysis.py verify
  "${PYTHON_BIN}" -m pytest -q reviewer_sequence_history_response/test_sequence_history.py
} >"${LOG_FILE}" 2>&1

printf 'complete\n' >"${STATUS_FILE}"
trap - EXIT
