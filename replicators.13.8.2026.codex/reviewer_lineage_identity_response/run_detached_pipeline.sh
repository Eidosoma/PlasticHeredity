#!/usr/bin/env bash
set -euo pipefail

workers="${1:-1}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
log_file="reviewer_lineage_identity_response/artifacts/detached_pipeline.log"
mkdir -p "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python"
fi
status_file="reviewer_lineage_identity_response/artifacts/detached_pipeline.status"
mkdir -p "$(dirname "$status_file")"
printf 'running\n' >"$status_file"
finish() {
  code=$?
  if [[ $code -eq 0 ]]; then
    printf 'complete\n' >"$status_file"
  else
    printf 'failed exit=%s\n' "$code" >"$status_file"
  fi
}
trap finish EXIT

if [[ ! -f reviewer_lineage_identity_response/artifacts/protocol/protocol.json ]]; then
  "$python_bin" -m reviewer_lineage_identity_response.run_analysis prepare
fi
"$python_bin" -m reviewer_lineage_identity_response.run_analysis simulate --mode all --workers "$workers"
"$python_bin" -m reviewer_lineage_identity_response.run_analysis fork --workers "$workers"
"$python_bin" -m reviewer_lineage_identity_response.run_analysis analyze
"$python_bin" -m reviewer_lineage_identity_response.run_analysis report
"$python_bin" -m reviewer_lineage_identity_response.run_analysis verify --full-replay
