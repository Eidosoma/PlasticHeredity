# Post-hoc strict-8 rulebook and holding-capacity probe

This folder is a separate mechanistic follow-up to the verified strict-8
prediction/mechanism diagnosis. It is **post hoc relative to the original
preprint and its preregistrations**, even though its protocol is frozen before
the new intervention futures are generated.

No file or result under `NewIdeas/` is an input. The hypothesis was motivated
by exploratory reasoning, but every analyzed quantity here is reconstructed
from the frozen replication simulator, sealed development/confirmation
artifacts, or newly generated futures.

## Questions

1. Does beta define a deterministic composition-space form whose alignment
   adds held-out information beyond history, concentration, and the retained
   state block?
2. Does holding measured under the *other* candidate implementation, pooled
   across its five landmarks and an independent branch half, predict
   strict-8/coherence under the target candidate?
3. Do equal mass- and richness-preserving molecule transfers toward versus
   away from the beta-derived form causally separate breaking from holding?

## Analysis structure

- `rulebook_core.py` contains outcome-blind expected-flow, fixed-point,
  feature, and intervention rules.
- `test_rulebook_core.py` tests solver, edit, gate, nesting, and leakage
  contracts.
- `run_analysis.py` freezes the source/protocol contract, builds rulebooks and
  cross-fitted holding features, seals development-only models, generates
  896,000 fresh common-stream futures, scores the results, and verifies exact
  deterministic replay.
- `artifacts/` contains the frozen protocol, derived inputs, resumable state
  checkpoints, output tables, figures, checksums, and manifests.

## Key inferential boundaries

- The cross-candidate holding score is a matrix calibration based on sibling
  futures. It is not a launch-time predictor available from one untouched
  state.
- The intervention reuses surviving, observable confirmation states and one
  selected daughter per fission.
- The registered cosine endpoint is primary for the causal rulebook test;
  calibrated Bray--Curtis endpoints are sensitivity readouts.
- A global causal endpoint is supported only if all four candidate-by-branch-
  half cells pass the frozen power, feasibility, interval, and Holm-adjusted
  criteria.

## Commands

```bash
python run_analysis.py prepare
python run_analysis.py rulebooks
python run_analysis.py holding
python run_analysis.py fit
python run_analysis.py score
python run_analysis.py intervention --workers 14
python run_analysis.py analyze-intervention
python run_analysis.py report
python run_analysis.py verify --workers 4
python run_analysis.py status
```

`python run_analysis.py all --workers 14` performs the complete resumable
workflow. The long intervention stage is designed to run detached.
