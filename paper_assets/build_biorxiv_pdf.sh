#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${project_dir}/output/pdf"
qa_dir="${project_dir}/tmp/pdfs/biorxiv"
pdf_path="${output_dir}/plastic-heredity-biorxiv-v1.pdf"

mkdir -p "${output_dir}" "${qa_dir}"
cd "${project_dir}"

python3 paper_assets/derive_f12_descriptives.py --check

pandoc \
  --defaults paper_assets/biorxiv_pdf.yaml \
  PRE_PRINT_PAPER_DRAFT.md \
  --output "${pdf_path}"

qpdf --check "${pdf_path}"
pdfinfo "${pdf_path}" > "${qa_dir}/pdfinfo.txt"
pdffonts "${pdf_path}" > "${qa_dir}/pdffonts.txt"
pdftotext -layout "${pdf_path}" "${qa_dir}/extracted.txt"

if rg -n 'Preprint draft for human review|remain to be approved|\bTBD\b|\bTODO\b' "${qa_dir}/extracted.txt"; then
  echo "Submission placeholder detected in generated PDF." >&2
  exit 1
fi

echo "Built ${pdf_path}"
