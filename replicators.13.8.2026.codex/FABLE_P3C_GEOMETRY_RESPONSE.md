Here is the full report for Codex. Everything is labeled by provenance: [SEALED] = part of the original G4 result, [REGENERATED] = deterministically recomputed from the sealed seeds in response to this request (byte-identical replay verified), [NEW] = computed for the first time in response to this request (exploratory, unregistered).

Conventions — both confirmed

- β[target, catalyst]: the kinetics are c = β @ n, bn = 1 + c/N, so β[i,j] = how much the presence of catalyst j boosts target i.
- Assembly catalytic throughput = xᵀβx: total support the present composition delivers to itself; with x supported on P, it equals x_Pᵀ β[P,P] x_P, so only the present-present block enters.

Per-state quantity changes, arm/none ratios (median [IQR] over 24 states per candidate) [REGENERATED except last row NEW]

Candidate 02 (median |P| = 19):

┌────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
│        arm         │          xᵀβx           │        Σ β[P,P]         │        ‖β[P,P]‖F        │    Perron ρ(β[P,P])     │
├────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ TIGHTEN ×1.5       │ 1.5000 exactly          │ 1.5000                  │ 1.5000                  │ 1.5000                  │
├────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ LOOSEN ÷1.5        │ 0.6667 exactly          │ 0.6667                  │ 0.6667                  │ 0.6667                  │
├────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ historical random  │ 1.0000 [0.9999, 1.0013] │ 1.0000 [0.9960, 1.0018] │ 1.0000 [1.0000, 1.0000] │ 1.0000 [0.9997, 1.1459] │
├────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ balanced-log (new) │ 1.3812 [0.6769, 1.4853] │ 1.3042 [0.8247, 1.4099] │ 1.4706 [0.5778, 1.4980] │ 1.0166 [0.9769, 1.2590] │
└────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘

Candidate 03 (median |P| = 18): TIGHTEN/LOOSEN identical by linearity; historical random 1.0000 on all four (Perron IQR [0.9996, 1.0004]); balanced-log 1.2489 / 1.1505 / 1.1883 / 1.1569 with similarly wide IQRs (e.g. Frobenius [0.6293, 1.4899]).

Three things these numbers establish. First, the TIGHTEN/LOOSEN rows are exact because scaling the block scales all four quantities linearly — a free implementation check for your clean room. Second, the historical random arm leaves every physically relevant block quantity essentially untouched (median ratio 1.0000 on all four; the occasional Perron excursion is a single scaled random edge landing inside P×P). This quantifies the locality caveat from the audit note: the historical null is largely a location null, and its whole-matrix Frobenius match does not translate into any block-level perturbation. Third, the balanced-log arm hits its Frobenius target exactly by construction (η bisected per state, median 0.411/0.478; achieved fraction = 0.5000 in all 48 states, no clipping, no duplicates) — but under the σ=4 lognormal, the signed quantities are dominated by a handful of giant entries, so fixed-norm balanced-log surgery is not throughput-neutral per state: xᵀβx moves anywhere from ×0.68 to ×1.49 depending on which signs the top entries draw.

JOINT_BREAK_RUN3 effects (whole-matrix bootstrap, 4,096 draws — descriptive confidence intervals, not formal equivalence tests; no TOST exists anywhere in this module)

┌─────────────────┬────────────────┬────────────────┬───────────────────────────────────────────────────────────────────────────┐
│    contrast     │    cand 02     │    cand 03     │                                provenance                                 │
├─────────────────┼────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ historical      │ −0.0156        │ −0.0026        │ point estimates [SEALED]; per-state values [REGENERATED], replay-exact to │
│ random − none   │ [−0.0833,      │ [−0.0485,      │  machine precision; CI re-drawn (sealed CI was [−0.0833, +0.0443] /       │
│                 │ +0.0469]       │ +0.0417]       │ [−0.0469, +0.0391] — difference is bootstrap draws only)                  │
├─────────────────┼────────────────┼────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ balanced-log −  │ +0.0417        │ −0.0443        │                                                                           │
│ none            │ [−0.0182,      │ [−0.1146,      │ [NEW], CRN branch streams identical to the sealed arms                    │
│                 │ +0.1016]       │ +0.0182]       │                                                                           │
└─────────────────┴────────────────┴────────────────┴───────────────────────────────────────────────────────────────────────────┘

The new within-block null: both CIs cross zero, but they are wide and opposite-signed between candidates — at 24 states × 16 branches this arm is underpowered relative to the raise−lowe its per-state heterogeneity is expected given thethroughput table above. An exploratory mechanism check — Spearman between each ballog state's realized xᵀβx ratio and its Δq — is uninformative at this n (−0.158 / +0.011). So the hup: coherent tightening/loosening moves heredityconsistently; norm-matched unstructured within-block perturbation shows no consistent directional effect at this scale, but that nuis noisy — if you want a decisive strict null, run better, stratify it by realized Δ(xᵀβx), which yourper-state heterogeneity makes possible for free.

One caution for interpretation: the balanced-log design your team proposed is a better control than our historical one, but as the table shows it is not a single control — it is a dis with sign determined by a few entries. A versionthat additionally constrains Σ or xᵀβx to be preserved (choose signs to zero the weighted sum, then bisect η on the residual norm) would isolate "unstructured disturbance at fixed th
