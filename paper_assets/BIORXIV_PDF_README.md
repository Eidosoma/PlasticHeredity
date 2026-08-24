# bioRxiv PDF build

The publication PDF is generated from the canonical root manuscript with Pandoc and XeLaTeX:

```bash
bash paper_assets/build_biorxiv_pdf.sh
```

The aggregate synthesis figures, the two data-backed exemplar figures, and the Chapter 5 Phi-r figure can be regenerated separately:

```bash
python paper_assets/generate_synthesis_figures.py
python paper_assets/generate_trajectory_figures.py
python paper_assets/generate_chapter5_figure.py
```

The manuscript's F12 descriptive counts and ranges are independently re-derived from the retained confirmation bundles. Regenerate the checked provenance record, or verify that it is current, with:

```bash
python paper_assets/derive_f12_descriptives.py
python paper_assets/derive_f12_descriptives.py --check
```

The trajectory generator deterministically selects its F12 and strict-eight examples from retained result archives, verifies the F12 replay and strict-event geometry, and writes the selection contract and identifiers to `figures/trajectory_exemplar_provenance.json`. The exemplars are post-outcome visualizations and do not supply additional inferential evidence.

The stable output is `output/pdf/plastic-heredity-biorxiv-v1.pdf`. Quality-control text, font, and PDF metadata reports are written under `tmp/pdfs/biorxiv/` and are not publication artifacts.

The build uses TeX Gyre Pagella for body text and mathematics, TeX Gyre Heros for headings, an A4 single-column page, inline figures, embedded hyperlinks, and a conventional reading order suitable for bioRxiv's later HTML/XML conversion.

The manuscript exposes three project-level provenance entries in the public `Eidosoma/Research` repository: the originating Eidosoma AI Scientist code together with its step reports and retained artefacts, the Codex Sol replication/discovery record, and the Claude Fable replication/discovery record. Manuscript links are pinned to immutable commits rather than the moving branch.

The PDF lists Robert Bjarnason as the formal author and gives Eidosoma.ai as the affiliation. The research-team disclosure identifies Eidosoma AI Scientist, Codex Sol, and Claude Fable as AI research systems rather than formal authors.
