# Scientific report: clean-room GARD lineage carrier

## Boundary first

This is an exploratory reviewer-motivated stress test of a claim the current preprint does not make. It is not preprint evidence, not an independent replication, and not a replication of the Wagner GRN work. Candidates 02 and 03 are alternative reconstruction contracts. Exact replay is computational reproducibility only.

## Registered outcome

**Classification:** `registered_carrier_family_failed_constructive_gate`

The selected carrier settings did not jointly pass maintenance, causal ablation/rescue, multiform selection, and relabeling gates in both contracts.

The carrier is an added inherited molecule-indexed register. Even a positive result therefore shows what an engineered side channel can do; it does not show that unmodified GARD already contains that mechanism.

## Engineering and confirmation

The engineering grid tested 72 settings on matrices 11, 54, 63 in both contracts. It froze 2 setting(s) before confirmation.
The benchmark selected tier B: 48 futures per arm through F64.

## Gate detail

### k008_l01_u2p0_nominal (k=8, L=1, coupling=2.0, nominal)

Overall: carrier memory both contracts=False; multiform both=False; relabeling both=True; constructive pass=False.

- Contract 02: native correct F32 terminal strict-8 4.8% (95% rule-bootstrap 1.4%–9.8%); correct-minus-zero 3.1%; cross-form A 3.4%, B 0.1%; origin accuracy 49.2%.
- Contract 03: native correct F32 terminal strict-8 4.4% (95% rule-bootstrap 0.9%–9.4%); correct-minus-zero 3.1%; cross-form A 2.8%, B 0.4%; origin accuracy 50.6%.

### k008_l08_u2p0_ideal (k=8, L=8, coupling=2.0, ideal)

Overall: carrier memory both contracts=False; multiform both=False; relabeling both=True; constructive pass=False.

- Contract 02: native correct F32 terminal strict-8 5.1% (95% rule-bootstrap 1.5%–10.0%); correct-minus-zero 3.4%; cross-form A 5.7%, B 0.4%; origin accuracy 50.3%.
- Contract 03: native correct F32 terminal strict-8 5.7% (95% rule-bootstrap 1.4%–11.8%); correct-minus-zero 4.3%; cross-form A 4.5%, B 0.6%; origin accuracy 49.3%.

## Interpretation limits

A 100-coordinate pass is reported as an engineered full-register result only. A nominal noisy pass with k≤32 is the stronger compressed tier. A null result limits only this frozen carrier family and completed runtime tier. These outcomes must not be merged into current-preprint evidence tables.
