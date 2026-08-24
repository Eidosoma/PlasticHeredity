#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd "${TASK_DIR}/.." && pwd)"
ARTIFACT_DIR="${TASK_DIR}/artifacts"
STATUS_FILE="${ARTIFACT_DIR}/detached_pipeline.status"
PYTHON="${REPOSITORY_DIR}/.venv/bin/python"
RUNNER="${TASK_DIR}/run_sensitivity.py"
WORKERS="${THRESHOLD_SENSITIVITY_WORKERS:-14}"

mkdir -p "${ARTIFACT_DIR}"
cd "${REPOSITORY_DIR}"

finish() {
    exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        printf 'COMPLETE\t%s\n' "$(date -Iseconds)" > "${STATUS_FILE}"
    else
        printf 'FAILED\t%s\texit=%s\n' "$(date -Iseconds)" "${exit_code}" > "${STATUS_FILE}"
    fi
}
trap finish EXIT

stage() {
    name=$1
    shift
    printf 'RUNNING\t%s\t%s\n' "${name}" "$(date -Iseconds)" > "${STATUS_FILE}"
    printf '[pipeline] %s started at %s\n' "${name}" "$(date -Iseconds)"
    "$@"
    printf '[pipeline] %s completed at %s\n' "${name}" "$(date -Iseconds)"
}

export PYTHONUNBUFFERED=1
stage prepare "${PYTHON}" "${RUNNER}" prepare
stage replay_f12 "${PYTHON}" "${RUNNER}" replay --dataset f12 --workers "${WORKERS}"
stage replay_f32 "${PYTHON}" "${RUNNER}" replay --dataset f32 --workers "${WORKERS}"
stage replay_cr1 "${PYTHON}" "${RUNNER}" replay --dataset cr1 --workers "${WORKERS}"
stage replay_reference "${PYTHON}" "${RUNNER}" replay --dataset reference --workers "${WORKERS}"
stage analyze "${PYTHON}" "${RUNNER}" analyze
stage report "${PYTHON}" "${RUNNER}" report
stage verify "${PYTHON}" "${RUNNER}" verify
stage unit_tests "${PYTHON}" -m pytest -q "${TASK_DIR}/test_sensitivity.py"
