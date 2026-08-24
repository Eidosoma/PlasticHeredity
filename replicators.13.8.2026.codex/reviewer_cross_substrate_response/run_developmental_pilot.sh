#!/usr/bin/env bash
set -euo pipefail

workers="${1:-14}"
campaign_python="${CAMPAIGN_PYTHON:-.venv/bin/python}"
module="reviewer_cross_substrate_response.run_experiment"

"${campaign_python}" -m "${module}" prepare
"${campaign_python}" -m "${module}" validate
"${campaign_python}" -m "${module}" calibrate --model all --workers "${workers}"
"${campaign_python}" -m "${module}" pilot --model all --workers "${workers}"
"${campaign_python}" -m "${module}" pilot-report
"${campaign_python}" -m "${module}" status
