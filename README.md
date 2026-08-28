# Plastic Heredity

Code, retained results, preregistrations, and provenance supporting manuscript version 2.1:

> **Plastic heredity: Predicting and steering how inheritance breaks, returns, and persists**  
> Robert Bjarnason · Eidosoma.ai

The scientific evidence cutoff for version 2.1 is **24 August 2026**; the manuscript is dated **27 August 2026**. This repository is a computational research record, not a claim that the simulated assemblies are living systems or validated models of prebiotic chemistry.

## What the study asks

Can the present state and catalytic context of a simulated pre-genetic assembly predict—and can external interventions alter—its future hereditary dynamics?

The project began by reconstructing the Phi-r analysis reported by Pigozzi and Levin in *[Causal Architecture Dynamics Prior to Arrival of Self-replicators in a Model of Catalytic Networks Relevant to Origin-of-Life](https://arxiv.org/abs/2607.28250)*. Our explicit public-source reconstructions recovered several retrospective patterns, but not the reported prospective prediction or intervention ordering. Because the target-specific implementation was unavailable at the evidence cutoff, that result is implementation-bounded rather than evidence that the target study is wrong.

The programme then tested the broader prediction-and-control question using a different process-level endpoint.

## Main result

We use **plastic heredity** for a selected-lineage process in which parent-to-daughter compositional inheritance breaks and later renews without returning to the old molecular composition. Here the term concerns compositional continuity, not inheritance of developmental plasticity.

The primary **F12 break-and-renewal** endpoint asks whether, within twelve fissions, an inheritance break is followed strictly later by three consecutive inherited boundaries. Across the originating workflow and two separate clean-room codebase reimplementations:

- present-state and catalytic-context predictors improved on their registered history-only ridge comparators across six reconstructed GARD contracts;
- molecular edits and catalytic-network interventions shifted F12 probability and separately altered resistance to a break and recovery after one;
- repeated feedback maintained high inherited-boundary frequency while active;
- most three-fission renewals were not one coherent composition-space regime, but stricter coherent eight-fission episodes occurred in 1.70–2.11% of 512,000 sampled futures;
- break-and-renewal was also detected before carrier manipulation in cellular automata and unaugmented Wagner gene-regulatory networks;
- a frozen cellular-automaton motif reader controlled one generation without testing onward transmission, while an explicitly added Wagner lineage carrier supported multigenerational memory and passed reversal, ablation, and rescue tests; and
- several Phi-r-family readings responded when heredity was altered, but no tested reading supplied shared foresight or a transferable control gradient.

The strongest conclusion is bounded: hereditary loss-and-renewal in these reconstructed simulations is state-dependent, prospectively steerable, and actively maintainable. The results do not establish autonomous agency, a permanent self, autonomous evolution, or a universal origin-of-life mechanism. In every GARD experiment, the catalytic matrix and molecular reservoir remain fixed environmental constraints that the assembly neither rewrites nor inherits.

## Start here

| Goal | Entry point |
|---|---|
| Understand the main clean-room implementation and its evidence | [Clean-room test 1 README](replicators.13.8.2026.codex/README.md) and [results ledger](replicators.13.8.2026.codex/RESULTS_LEDGER.md) |
| Inspect the independent codebase reimplementation | [Clean-room test 2 README](replicators.13.8.2026.fable/replication/README.md) and [report](replicators.13.8.2026.fable/replication/REPORT.md) |
| Follow the originating reconstruction and discovery code | [Originating implementation README](original.1.8.2026.eidosoma-ai-scientist.code/arrival-of-self-replicators-eidosoma-groups-42/README.md) and [step-report archive](original.1.8.2026.eidosoma-ai-scientist.stepReports/) |
| Audit the direct target-paper reconstructions | [Codex reconstruction](paperFinalReplicationAttempt.15.8.2026.codex/README.md) and [Claude reconstruction report](paperFinalReplicationAttempt.15.8.2026.claude/REPORT.md) |
| Inspect manuscript figures and provenance | [Figures](figures/) and [paper assets](paper_assets/) |

Files named `PRE_PRINT_PAPER_DRAFT.md` inside implementation directories are historical branch-level drafts. They are retained as part of the research record and should not be treated as the canonical version 2.1 manuscript.

## Repository map

```text
.
├── figures/                                             # manuscript figures and provenance
├── paper_assets/                                        # figure/provenance scripts and PDF configuration
├── original.1.8.2026.eidosoma-ai-scientist.code/        # originating reconstruction and discovery code
├── original.1.8.2026.eidosoma-ai-scientist.stepReports/ # plans, reports, registrations, and artefacts
├── replicators.13.8.2026.codex/                         # clean-room test 1 and later adversarial extensions
├── replicators.13.8.2026.fable/                         # clean-room test 2
├── paperFinalReplicationAttempt.15.8.2026.codex/        # direct target-paper reconstruction
├── paperFinalReplicationAttempt.15.8.2026.claude/       # second direct reconstruction
└── LICENSE
```

### Originating research record

`original.1.8.2026.eidosoma-ai-scientist.code/` contains the initial public-source reconstruction and the workflow in which the alternative F12 coordinate was developed. `original.1.8.2026.eidosoma-ai-scientist.stepReports/` preserves the associated plans, registrations, diagnostics, and generated research artefacts.

### Clean-room tests and extensions

`replicators.13.8.2026.codex/` contains clean-room test 1, scale-ups, adversarial analyses, molecular and network interventions, feedback experiments, generative nulls, Phi-r integration, reviewer-response analyses, cellular-automaton motif tests, and Wagner-network programmes.

`replicators.13.8.2026.fable/replication/` contains clean-room test 2, including its separate reproduction and scale-up, coherence and strict-eight tests, interventions, feedback and policy-compression studies, and Phi-r analyses.

“Clean room” means a separate codebase reimplementation without reuse of the originating implementation or scientific data. These are implementation-level checks conducted under one human research director, not independent laboratories.

### Direct reconstruction of the target study

The two `paperFinalReplicationAttempt.*` directories contain separately built reconstructions of the target paper's reported GARD, Phi-r, prediction, and intervention pipeline. They address recurrence of that particular operational pipeline and are distinct from the positive plastic-heredity result obtained with the F12 endpoint.

## Evidence conventions

The programme used staged development and untouched confirmation cohorts, fixed seed domains, candidate-separated analysis, whole-matrix inference, preregistered gates where applicable, exact replay, readback audits, and SHA-256 manifests. Retained provenance varies by campaign; the relevant README, preregistration, and result ledger state what is available.

Important conventions:

- the originating workflow and the two clean-room implementations use materially different simulator contracts;
- local candidate labels `02` and `03` do not denote identical contracts across implementations;
- “shared” means that a result recurred in both clean-room tests;
- prospective confirmation, post-hoc robustness analysis, and exploratory follow-up remain distinct;
- failed required cells are retained rather than repaired through pooling; and
- active or incomplete campaigns are excluded from the manuscript's evidence cutoff.

Machine-readable JSON, CSV, manifests, hashes, preregistrations, and replay audits are stored beside the corresponding reports where retained. Large raw arrays and resumable scientific checkpoints were not retained uniformly; compact reports and verification records identify those boundaries.

## Reproducing the work

There is intentionally no root command that silently combines the different scientific implementations. Each codebase has its own environment, simulator contract, and execution instructions. Begin with the README and preregistration for the campaign you want to reproduce.

The checked manuscript-level descriptive statistics are recorded in
[machine-readable provenance](paper_assets/f12_descriptive_provenance.json),
and the calculation is documented in
[`derive_f12_descriptives.py`](paper_assets/derive_f12_descriptives.py). Running
that derivation requires the raw Codex and Fable confirmation bundles, which
are not included in this repository snapshot; the script therefore does not
run standalone from this checkout.

Manuscript figure generators and historical PDF-build assets are under `paper_assets/`. Those build files are retained as part of the research record and are not the current version 2.1 submission package.

Some full campaigns are computationally expensive. Do not infer a common environment or silently pool candidate contracts; follow the campaign-level documentation and fixed seed domains.

## Attribution

Robert Bjarnason conceived and directed the research programme and is the accountable author. The computational work used the Eidosoma AI Scientist, Codex Sol, and Claude Fable for code, simulation, analysis, documentation, visualisation, and drafting. These are credited as AI research systems, not formal authors.

## Licence

Except where a file or directory states otherwise, original project code and generated data are available under the [MIT License](LICENSE). Third-party materials and the manuscript/preprint retain their applicable rights and are not relicensed by that file.

Third-party papers are cited through official sources and are not redistributed here. Historical workflows may record original input paths and hashes; obtain those sources from their official locations when reproducing the corresponding work.

## Status

This repository is the public computational companion to manuscript version 2.1. It may be updated after release of the target study's implementation, substantive external review, or completion of later campaigns. For audit-stable use, cite a tagged release or immutable commit rather than a moving branch.
