#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run quarantine --workers "${1:-1}"
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run confirm --workers "${2:-8}"
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run report
.venv/bin/python -m reviewer_ca_lineage_renewal_replication.run verify
