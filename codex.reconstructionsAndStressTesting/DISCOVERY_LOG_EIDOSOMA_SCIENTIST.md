# Discovery Log — Clean-Room Plastic Heredity in Cellular Automata

Date: 2026-08-20

Status: golden-trace reconciliation complete

Current run: [`results/golden-reconciliation/`](results/golden-reconciliation/)

## Executive verdict

The code-free golden traces were sufficient to turn the earlier structural
replication into an **exact quantitative reconstruction of the retained raw ECA
atlas**. Across all 88 canonical rules and 180,224 futures, strict frequency,
break-by-8 frequency, median first-generation time, mean survival, and every
form-library ID match the retained CSV exactly. Every numerical error is zero
and every rank correlation is 1.0.

This exactness is scoped. The phase experiment used an undisclosed fresh random
stream and therefore agrees strongly rather than cell-for-cell. Particle-domain
launch rows remain undisclosed, and Life remains numerically nonidentical even
after applying the supplied pooling convention. Those limitations do not weaken
the exact raw-atlas result; they define the remaining clean-room boundary.

## Golden-trace reconciliation, 2026-08-20

### Hard replay prerequisite

An integer replay independent of the NumPy campaign engine reproduced:

- all 16 authoritative launch rows and populations;
- all **907/907** post-sweep rows and activity increments;
- all **15/15** terminal final4 spectra; and
- every disclosed copy mask and offspring row.

The vendored trace is accompanied by the upstream evidence hashes in
[`MANIFEST.json`](results/golden-reconciliation/MANIFEST.json). The exact engine
pins NumPy 2.5.2 and PCG64.

### What fixed the raw atlas

The high-impact conventions were not cosmetic:

1. the same heterogeneous 16-row launch library is shared across rules, with no
   burn-in or noiseless preparation;
2. composition zero is the first completed generation;
3. noise is applied after the rule, and activity counts the realized post-noise
   row difference;
4. failure to reach activity 256 by sweep 128 is death even when the terminal
   row is not monochrome;
5. the terminal row is observed before copying error, but the whole generation
   batch consumes an unconditional copy draw;
6. each rule/seed has one PCG64 stream whose per-sweep arrays shrink as futures
   finish, and strict-positive futures stop entering later generations;
7. the retained `break_by_8` name covers generations 0–7, hence seven observable
   fidelity boundaries; its strict comparison anchor is the break-causing
   daughter; and
8. ECA forms use each broken future's last completed composition, with the
   break-causing daughter satisfying the later-generation requirement. Form IDs
   are assigned in first-discovery order.

Items 7–8 were inferred from the retained outputs after the trace pack fixed the
physical trajectory. They are now regression-pinned: changing either loses
exact equality.

### Raw ECA atlas: exact

| Quantity | Exact matches | Error / agreement |
|---|---:|---:|
| Strict frequency | 88/88 | MAE 0; Spearman 1.0 |
| Break-by-8 | 88/88 | MAE 0; Spearman 1.0 |
| Median first-generation sweeps | 88/88 | MAE 0; Spearman 1.0 |
| Mean survival | 88/88 | MAE 0; Spearman 1.0 |
| Form libraries | 88/88 | 217/217 global forms |

The retained scientific fingerprint is recovered: the five strict champions
are rules 35 (`0.43457`), 43 (`0.34863`), 11 (`0.33984`), 57 (`0.29297`), and
184 (`0.16797`); rule 110 and all raw class-4 representatives have strict rate
zero; the undisputed class-3 core has minimum break rate `0.94238`, above the
clean class-1/2 median `0.27124`.

The clean-room re-adjudication gets the same class separation, rule-110,
heavy-tail, and smoothness directions. Its descriptor-vs-Hamming metric
calculation still differs from the retained summary despite identical atlas
rows, locating that residual specifically in the downstream null/metric
calculation rather than the simulator or atlas.

### Five-point phase: strong, not bit-exact

All 440 cells ran. Exact-cell counts are 190/440 for strict, 100/440 for break,
and 405/440 for the median clock. Agreement is nevertheless very strong:

| Endpoint | Spearman | MAE |
|---|---:|---:|
| Strict | 0.96362 | 0.00207 |
| Break-by-8 | 0.99877 | 0.00478 |
| Median sweeps | 0.99905 | 0.0773 |

All registered phase conclusions match: every noise cell excludes the class-3
core from the capability band, no class-1 rule awakens, 58 rules are capable
somewhere, every capable profile is unimodal within tolerance, and raw rule 110
has zero strict events at all five points. The remaining cell differences are
consistent with the retained phase using a fresh stream whose seed tag was not
included in the code-free pack.

### Particle, Life, and evolution

- **Particle:** the corrected batch lifecycle gives rule 110 strict
  `37/2048 = 0.01807`, nearly the retained `38/2048 = 0.01855`; the redemption
  gate passes and the chaotic-control gate fails in both. The clean-room
  champion-stability gate passes while the retained one fails. Numeric equality
  is not claimed because the four domain-dictionary launch rows were not given.
- **Life:** the disclosed hierarchical round-5 pooling is implemented exactly
  as prose: normalize each completed generation, average through the
  break-causing generation within each future, then average futures equally.
  Fidelity, persistence, and distinct-form gates all pass. Supports are glider
  `159`, blinker `667`, and toad `2715`, versus retained `2715`, `667`, and
  `671`; only blinker is exact, so execution/RNG details beyond pooling remain.
- **Evolution:** on the exact atlas, selection again finds the class-2 boundary
  and repertoire stickiness again fails, matching both retained gate outcomes.
  GPS remains out of scope because no independent fresh-stream truth table was
  available.

### Verification and artifacts

All **30 tests pass**. The reference cascade completed every stage with no
software errors. Key artifacts are:

- [`REPORT.md`](results/golden-reconciliation/REPORT.md)
- [`RESULTS.json`](results/golden-reconciliation/RESULTS.json)
- [`eca_rules.csv`](results/golden-reconciliation/atlas/eca_rules.csv)
- [`reference_comparison.json`](results/golden-reconciliation/atlas/reference_comparison.json)
- [`phase.csv`](results/golden-reconciliation/phase/phase.csv)
- [`particle_gates.json`](results/golden-reconciliation/particle/particle_gates.json)
- [`observer.json`](results/golden-reconciliation/life/observer.json)

## Superseded first-round record

The remainder of this document preserves the 2026-08-19 pre-trace run as an
audit trail. Its “partial replication” verdict describes the underspecified
first implementation, not the reconciled result above.

## Clean-room boundary

The implementation was written from the Plastic Heredity preprint, sibling
prose contracts, and retained result JSON/CSV files. No sibling `src/`,
`tests/`, or executable script was opened or reused. The implementation and
the unresolved semantic choices are recorded in
[`CLEANROOM_PROTOCOL.md`](CLEANROOM_PROTOCOL.md). All 14 independent unit
tests pass.

## What the strict endpoint means

A lineage first has to stop resembling its old anchor at cosine similarity
`<= 0.9`. Strict plastic heredity is counted only if, later within the
32-boundary horizon, it produces eight consecutively inherited daughters, all
28 daughter pairs are mutually coherent above `0.9`, and every daughter remains
at most `0.85` similar to the old anchor. Thus ordinary persistence and mere
change do not count: the lineage must change and then inherit the changed form.

## Run scale

- Raw atlas: 88 ECA orbits × 16 seeds × 128 futures = **180,224 lineages**.
- Noise phase: 88 orbits × 5 noise levels × 2,048 futures = **901,120
  lineages**.
- Particle observer: 25 registered rules × 2,048 futures = **51,200
  lineages**.
- Life: four named-object cells × 256 futures plus three random-control cells ×
  2,048 futures = **7,168 lineages**.
- Evolution: 32 repeats of four arms, 24 individuals, and 40 generations.
- Total simulated lineage futures: **1,139,712**.
- Sixteen workers were used for the ECA experiments. Measured stage time was
  267.70 s atlas, 1,243.02 s phase, 108.64 s particle, 139.32 s Life, and
  6.59 s evolution: about **29 min 25 s** of stage wall time.

## Gate-level scorecard

| Claim or diagnostic | Clean room | Retained result | Adjudication |
|---|---:|---:|---|
| Raw class-3 break separation | pass | pass | reproduced |
| Raw class-4/rule-110 strict frequency negligible | pass | pass | reproduced |
| Raw strict events have a heavy tail | pass | pass | reproduced |
| Neighboring ECA rules have smoother repertoires than random pairs | pass | pass | reproduced |
| Exact raw rule ordering and rates | no | reference table | not reproduced |
| Descriptor metric beats Hamming distance | pass | fail | opposite outcomes |
| Noise ridge/unimodality | pass | pass | reproduced |
| Class-2-only phase regime is robust across noise | fail | pass | not reproduced |
| No class-1 awakening | fail: six awaken | pass: none awaken | not reproduced |
| Particle observer redeems rule 110 | pass | pass | reproduced, weaker rate |
| Particle observer keeps all chaotic controls out | fail | fail | same negative result |
| Particle observer preserves all raw champions | pass | fail | different outcome |
| Named Life objects have high absolute fidelity | pass | pass | reproduced |
| Named Life objects out-persist random controls | pass | pass | reproduced |
| Glider, blinker, and toad have distinct form supports | fail | pass | not reproduced |
| Evolutionary selection finds the boundary robustly | fail | pass | not reproduced |
| Evolved walks are repertoire-sticky | pass | fail | opposite outcomes |
| Goal-directed planning search (GPS) | omitted | pass | not tested clean-room |

## 1. Raw-texture ECA atlas

### Structural outcomes

The clean-room atlas recovered all 88 reflection/conjugation orbits. The
break-by-8 medians were `0.2246`, `0.8916`, `1.0000`, and `0.9995` for Wolfram
classes 1–4. The minimum undisputed class-3 break rate was `0.9839`, above the
clean class-1/2 median of `0.8481`; the registered separation gate passes.

Raw class-4 strict heredity remained negligible:

| Rule | Clean-room strict | Retained strict |
|---:|---:|---:|
| 41 | 0 | 0 |
| 54 | 0.000488 | 0 |
| 106 | 0 | 0 |
| 110 | 0 | 0 |

Rule 110 ranked `79.5` of 88 in the clean-room strict ordering and did not enter
the top decile, matching the retained negative result.

The other landscape-level diagnostics were close:

| Diagnostic | Clean room | Retained result |
|---|---:|---:|
| Heavy-tail share | 0.6714 | 0.6455 |
| Neighbor/random repertoire smoothness ratio | 2.8656 | 3.1534 |
| Number of registered forms | 188 | 217 |
| Descriptor-distance Spearman | 0.1237 | 0.1350 |
| Hamming-distance Spearman | 0.0822 | 0.1534 |

Heavy-tail and smoothness gates pass in both runs. The metric diagnostic flips:
descriptor distance beats Hamming in the clean room, whereas the retained run
found the reverse. This is a mismatch, even though the clean-room gate itself
is positive.

### Numerical mismatch

Across all 88 rules, break-by-8 has Spearman `0.7113` and mean absolute error
`0.2545` against the retained table. Strict heredity has Spearman `0.5483` and
mean absolute error `0.1441`. This is meaningful rank agreement at the broad
class level, but not a numerical reconstruction.

The top five lists show the problem directly:

| Rank | Clean-room rule | Clean-room strict | Retained rule | Retained strict |
|---:|---:|---:|---:|---:|
| 1 | 28 | 0.9097 | 35 | 0.4346 |
| 2 | 14 | 0.8555 | 43 | 0.3486 |
| 3 | 156 | 0.8511 | 11 | 0.3398 |
| 4 | 13 | 0.8140 | 57 | 0.2930 |
| 5 | 172 | 0.7935 | 184 | 0.1680 |

In the retained table, clean-room leaders 28, 14, 156, 13, and 172 score only
`0.00684`, `0.0327`, `0.00293`, `0.000488`, and `0`, respectively. Conversely,
all five retained champions remain nonzero in the clean room, but at altered
rates: rule 35 `0.3013`, 43 `0.5215`, 11 `0.6611`, 57 `0.7246`, and 184
`0.4585`.

The retained directional checks—class separation, negligible raw class 4,
negligible raw rule 110, and at least three nonzero class-2 champions—all pass.
The exact atlas fingerprint does not.

Raw files:
[`atlas_summary.json`](results/reference-suite/atlas/atlas_summary.json),
[`eca_rules.csv`](results/reference-suite/atlas/eca_rules.csv), and
[`reference_comparison.json`](results/reference-suite/atlas/reference_comparison.json).

## 2. Five-point noise phase

The grid used process-noise values `0.0025`, `0.005`, `0.01`, `0.02`, and
`0.04`, with copy error `1.5 × eta`. Rule profiles are almost always unimodal:
`0.9855` of ever-capable rules in the clean room versus `1.0` retained. The
ridge-law gate therefore reproduces. The number of rules ever entering the
registered capability band is 69 versus 58 retained.

The regime composition does not reproduce:

| eta | Clean band size | Clean class-2 share | Clean core-3 intruders | Retained band size | Retained class-2 share | Retained core-3 intruders |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0025 | 35 | 91.4% | 2 | 36 | 97.2% | 0 |
| 0.005 | 42 | 95.2% | 2 | 34 | 100% | 0 |
| 0.01 | 41 | 95.1% | 2 | 36 | 100% | 0 |
| 0.02 | 44 | 84.1% | 1 | 35 | 100% | 0 |
| 0.04 | 36 | 88.9% | 0 | 25 | 100% | 0 |

Only the highest-noise clean-room cell passes the class-purity criterion, so
the registered regime-robustness gate fails. Six nominal class-1 rules awaken
somewhere on the grid—8, 32, 40, 136, 160, and 168—whereas the retained run has
none. Rule 110 remains effectively absent in raw texture: one event in 2,048
at `eta=0.0025` and zero at the other four cells; the retained run has zero at
all five.

Result files:
[`phase_summary.json`](results/reference-suite/phase/phase_summary.json) and
[`phase.csv`](results/reference-suite/phase/phase.csv).

## 3. Particle/figure-ground observer

The observer learned a rule-specific noiseless 3×3 spacetime-domain dictionary
and measured 4-mer composition only on cells outside that dictionary. Its
learned coverage is strikingly close to the retained observer:

| Wolfram class | Clean coverage | Retained coverage |
|---:|---:|---:|
| 1 | 1.0000 | 1.0000 |
| 2 | 0.9113 | 0.9173 |
| 3 | 0.8679 | 0.8656 |
| 4 | 0.9061 | 0.9064 |

The central observer-dependence result reproduces. Raw rule 110 has zero
strict events, while the particle observer gives 15/2,048 = `0.00732`, inside
the registered redemption band. The retained observer gives 38/2,048 =
`0.01855`. The clean-room Rule-110 break rate is `0.4385` versus `0.1084`
retained, and its mean above-threshold inheritance is `0.9104` versus `0.9766`;
the rescue is therefore real but quantitatively weaker.

Class-4 rates are all positive in both runs:

| Rule | Clean strict | Retained strict |
|---:|---:|---:|
| 41 | 0.01563 | 0.10303 |
| 54 | 0.04346 | 0.05420 |
| 106 | 0.00732 | 0.04248 |
| 110 | 0.00732 | 0.01855 |

The class-3 control still fails in both runs. Its maximum strict rate is
`0.02295` clean-room versus `0.06836` retained: less contamination, but still
above the registered exclusion threshold. The five raw champions all clear
the clean-room particle threshold (strict rates 11 `0.0151`, 35 `0.0244`, 43
`0.1084`, 57 `0.0127`, 184 `0.0200`), so champion stability passes here. It
fails in the retained run because rule 57 falls to `0.00342`.

Result file:
[`particle_gates.json`](results/reference-suite/particle/particle_gates.json).

## 4. Conway Life named objects

The registered 16×16 Life observer uses a generation-averaged live 2×2 census.
Absolute fidelity and persistence reproduce closely:

| Object | Clean mean H | Retained mean H | Clean survival-8 | Retained survival-8 | Clean random control | Retained random control |
|---|---:|---:|---:|---:|---:|---:|
| Blinker | 0.96969 | 0.96681 | 0.5586 | 0.5078 | 0.0688 | 0.0601 |
| Glider | 0.97062 | 0.97015 | 0.5742 | 0.5664 | 0.0869 | 0.0596 |
| Toad | 0.97035 | 0.97027 | 0.6211 | 0.5703 | 0.0464 | 0.0337 |

The named structures persist roughly 6.6–13.4 times as often as their
density-matched random controls in the clean room. The descriptive block has
mean fidelity `0.9691` and survival-8 `0.3906`, versus `0.9583` and `0.3398`
retained.

Break rates are glider `0.1563`, blinker `0.4102`, toad `0.2695`, and block
`0.2266`; retained values are `0.1445`, `0.3086`, `0.1758`, and `0.2148`.
Strict rates are small—glider `0`, blinker `0.0156`, toad `0.00391`, and block
`0.0117`—because persistence without a qualifying break is deliberately not a
strict plastic-heredity event.

Form distinctness does not reproduce. Clean-room form supports are blinker
`2715`, glider `2719`, and toad `2719`, so glider and toad merge. The retained
supports `667`, `2715`, and `671` are all distinct. Between-object cosine
similarities are nevertheless nearly identical across runs (`0.990`–`0.997`
clean-room and `0.990`–`0.998` retained), locating the mismatch in the discrete
support classification rather than overall fidelity or persistence.

Result file: [`observer.json`](results/reference-suite/life/observer.json).

## 5. Evolution on the ECA rule hypercube

Selection raises mean final fitness, but not enough to clear the clean-room
control gate:

| Arm | Clean mean | Clean 95% interval | Retained mean | Retained 95% interval |
|---|---:|---:|---:|---:|
| Selection | 0.2642 | 0.1485–0.4379 | 0.1066 | 0.0492–0.2160 |
| Drift | 0.1600 | 0.0720–0.2742 | 0.0367 | 0.00858–0.0689 |
| Random walk | 0.1777 | 0.0943–0.3180 | 0.0302 | 0.00633–0.0725 |
| Fitness shuffled | 0.1472 | 0.0485–0.2745 | 0.0252 | 0.00510–0.0640 |

The clean-room selection lower bound (`0.1485`) does not exceed the drift and
random-walk means, so `gate_selection_finds_boundary` fails; the retained gate
passes. This follows naturally from the inflated and redistributed raw atlas:
the controls encounter many high-scoring rules without selection.

Selection visits the class-3 core `0.0852` of the time versus `0.1007` for a
random walk, a modest avoidance. The retained comparison is `0.0620` versus
`0.1018`, a stronger effect.

Repertoire stickiness reverses. Clean-room evolved one-bit transitions have
mean Jaccard `0.1447` versus random-pair `0.0547` (nonempty reanalysis ratio
`2.655`), so the sticky-walk gate passes. The retained run reports `0.0201`
versus `0.0511`, so it fails.

The goal-directed planning-search (GPS) result was deliberately omitted. A
clean-room GPS adjudication needs an independent fresh-stream truth table; using
the same atlas for planning and truth would be circular. The retained GPS gate
passes, but this reconstruction makes no claim about it.

Result file:
[`evolution_summary.json`](results/reference-suite/evolution/evolution_summary.json).

## Scientific interpretation

### What is independently supported

1. A strict change-then-inheritance event can be realized in noisy cellular
   automata without importing the sibling implementation.
2. Raw-texture plastic heredity is strongly structured by dynamical regime and
   is concentrated in a heavy tail of mostly class-2 rules.
3. Rule 110 demonstrates observer dependence: it is null in raw texture and
   positive under a rule-conditioned figure/ground observer.
4. Named Life structures retain high fidelity and multi-generation persistence
   far beyond density-matched random initial conditions.
5. Local rule-space geometry is nonrandom: neighboring rules share repertoire
   structure more than unrelated pairs.

### What is not independently pinned down

1. The exact ECA champion identities, ranking, or strict-event probabilities.
2. A class-2-only capability band across the full noise grid.
3. The absence of class-1 awakenings.
4. A unique discrete form identity for each named Life object.
5. Robust evolutionary localization of the boundary landscape.
6. Goal-directed planning performance.

The large raw-atlas discrepancies cannot be attributed to seed noise at 2,048
futures per rule. The most consequential unresolved axes are the launch-anchor
timing, seed-row ensemble, process-noise ordering, terminal/death bookkeeping,
and post-break pooling convention. Form pooling directly affects repertoire
and Life-form results; the first four axes can also alter strict lineage rates.

The clean-room result therefore strengthens the paper's broad claim that
heredity is observer-relative and localized near dynamical organization, while
also showing that the retained prose/data do not specify a unique executable
experiment closely enough to recover the exact numerical atlas.

<!-- causal-heredity-round-1:start -->
## Causal heredity round 1

Completed `1787313673.1448975` under design `51710c4b87bdf0f154c67741daff6b15c7ff5d97fc2a68af62fe79da354f3e8b`.

- `causal_transmission`: `False`
- `structure_matters`: `False`
- `dose_response`: `False`
- `pedigree_persistence`: `False`
- `observer_robustness`: `False`
- `environmental_memory`: `False`
- `rule_specificity`: `True`

See `results/causal-heredity-round-1/REPORT.md` and `LAY_SUMMARY.md` for the full evidence boundary.
<!-- causal-heredity-round-1:end -->

<!-- life-carrier-round-2:start -->
## Life carrier round 2

Completed under design `9565528d988ec8b79c797103da560f239b7f681cb2768d70d5baed9b04705582`.

- Rule-125398 audit: `SATURATION_LEAD_NOT_REPLICATED`
- Sealed holdout candidates: `[]`
- Overall verdict: `NO_CAUSAL_CARRIER_FOUND`

See `results/life-carrier-round-2/REPORT.md` and `LAY_SUMMARY.md`.
<!-- life-carrier-round-2:end -->

<!-- ca-carrier-round-3:start -->
## CA carrier round 3

State `complete` under design `041fd850aa50248722cd190c03d59409560b8d91939d0729b48ee7fa2cce61f3`.
Narrow rule-31649 verdict: `DURABLE_LOCAL_PLASTIC_HEREDITY`.
Broad holdouts adjudicated: `8`.

See `results/ca-carrier-round-3/REPORT.md` for the continuous-form evidence ladder.
<!-- ca-carrier-round-3:end -->

<!-- ca-lineage-field-round-4:start -->
## CA lineage-field round 4

Overall verdict: `NO_RENEWED_LINEAGE_CARRIER`.
Selected timing profile: `reference`; elapsed `0.07` wall hours.
See `results/ca-lineage-field-round-4/REPORT.md` and `LAY_SUMMARY.md`.
<!-- ca-lineage-field-round-4:end -->

<!-- ca-motif-lineage-stage-1:start -->
## CA motif-lineage Stage 1

Upper-bound verdict: `ROBUST_LOCAL_MOTIF_CONTROLLABILITY`.
Profile: `reference`; elapsed `0.040` wall hours.
See `results/ca-motif-lineage-stage-1/REPORT.md` and `LAY_SUMMARY.md`.
<!-- ca-motif-lineage-stage-1:end -->

<!-- ca-motif-lineage-stage-2:start -->
## CA motif-lineage Stage 2

Frozen-reader verdict: `DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL`.
Profile: `reference`; elapsed `0.211` wall hours.
See `results/ca-motif-lineage-stage-2/REPORT.md` and `LAY_SUMMARY.md`.
<!-- ca-motif-lineage-stage-2:end -->

<!-- ca-motif-lineage-stage-3:start -->
## CA motif-lineage Stage 3

Renewed-lineage verdict: `NO_RENEWED_CA_PLASTIC_HEREDITY`.
Profile: `reference`; elapsed `0.357` wall hours.
See `results/ca-motif-lineage-stage-3/REPORT.md` and `LAY_SUMMARY.md`.
<!-- ca-motif-lineage-stage-3:end -->

<!-- ca-motif-lineage-stage-3r:start -->
## CA motif-lineage Stage 3R

State: `awaiting_human_review`; verdict: `AWAITING_HUMAN_REVIEW`.
Elapsed `0.001` wall hours.
See `results/ca-motif-lineage-stage-3r/REPORT.md` and `LAY_SUMMARY.md`.
<!-- ca-motif-lineage-stage-3r:end -->

<!-- STAGE4_COMPRESSION_START -->
## CA Stage 4 — compressed texture heredity

State: `complete`; verdict: `ROBUST_COMPACT_RENEWED_CA_PLASTIC_HEREDITY`.
Elapsed `2.071` wall hours.
See `results/ca-motif-lineage-stage-4/REPORT.md` and `LAY_SUMMARY.md`.
<!-- STAGE4_COMPRESSION_END -->

<!-- CA_MOTIF_STAGE6_IMPLEMENTATION_START -->
## CA motif-lineage Stage 6 — bounded minimality programme

Implemented on 2026-08-24 after the sealed Stage-5R verdict
`ROBUST_REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY`. Stage 6 does not treat
that success as the end point: the Stage-5R one-site seed still filled the
entire 16x16 torus, and its local writer circuit still aggregated the entire
lattice. The new question is whether the history-specific recovery survives
when both operations have genuinely finite causal reach.

The clean-room implementation adds five separately invoked gates:

1. `locality`: 20 finite-hop/finite-writer-window mechanisms, strict causal
   qualification to generation 32, and endurance to generation 64;
2. `scale`: fixed and diameter-scaled communication on 16, 32, and 64 cells,
   exact light-cone checks, distance-band phenotypes, and an 18-neighbour rule
   panel without refitting;
3. `compression`: 64-, 36-, 24-, 16-, 8-, and 4-bit carriers, full ordinary
   and damaged causal ladders, corruption/drift curves, information lower
   bounds, and a cost/effect Pareto frontier;
4. `ecology`: paired carrier-wave collisions plus descent and de-novo sparse
   writer evolution, with proxy winners counted only after disjoint full-CA
   validation; and
5. `audit`: a separately authorized, model-sealed test on all 62 remaining
   untouched founder pairs.

Every visible daughter is bitwise reset before each generation. Only the
quantized boundary payload crosses that reset. Germination uses synchronous
nearest-neighbour copying, reading is local to the occupied light cone, and
rewriting is an explicitly audited directed shift-and-accumulate rectangle.
Inherited bits, temporary field/writer storage, routing work, and shared
parameters remain separate accounting categories.

The complete five-round smoke execution passed its engineering checks,
including 64x64 lattices, all checkpoint/resume paths, round decisions, final
authorization, and pre-audit model sealing. Smoke effects are not scientific
results: its two-pair gates are explicitly marked non-adjudicative. The
reference Round 6A is the only round authorized to launch initially; later
rounds wait for human review and no successor launches automatically.

Protocol: `CA_MOTIF_LINEAGE_STAGE6_PROTOCOL.md`. Implementation:
`plastic_ca/motif_minimality.py`. Tests: `tests/test_motif_minimality.py`.
<!-- CA_MOTIF_STAGE6_IMPLEMENTATION_END -->

<!-- CA_MOTIF_STAGE6AR_IMPLEMENTATION_START -->
## CA motif-lineage Stage 6A and Stage 6A-R correction

Reference Stage 6A completed on 2026-08-24. Its historical global and exact
controls reproduced strong generation-8 lineage crossover (0.949 and 0.959),
but no bounded candidate passed; the selected diagnostic reached crossover
zero at generations 16, 32, and 64. This initially appeared to limit the
positive Stage-5R result to organism-wide coordination.

Forensic inspection then located a specific v1 normalization defect. The
directed rectangle reducer already returned a spatial mean, but the writer
divided it by the rectangle area again. At full 16x16 span this attenuated the
empirical history term by exactly **1/256**. The first rewritten compact-anchor
payload consequently had zero lineage-centroid distance even though its
absolute bias remained nonzero. Independently randomizing the next seed origin
also allowed a bounded writer to sample outside the parent's causal region.
The negative Stage 6A record remains immutable, but it is not a clean rejection
of bounded locality.

Stage 6A-R implements a separately hashed correction and five manually gated
phases: arithmetic/equivalence audit, a 60-cell hop/window/origin bridge
matrix, local causal screening, strict qualification, and generation-64
endurance. It adds co-located and adjacent-budding origin policies, preserves
independent relocation as a non-promotable control, uses candidate-order-
independent paired random streams, and records entry/exit carrier centroids,
writer moments, causal overlap, decoder ties, and spatial phenotype effects.

The complete smoke workflow passed its engineering gates. In that two-pair
diagnostic, the corrected full bridge retained 0.75 of the historical compact
anchor while the frozen legacy calculation again produced exactly zero. Smoke
effects are not scientific evidence. The official reference correction audit
then passed under frozen design
`7be77a6e42c2a2eafdb3ec99cd106313164acebd5773d8b6b4952855a6e67225`:
the maximum writer-moment error was `6.94e-18`, full-span quantized payloads
matched Stage 5R exactly, the historical attenuation reproduced as
`0.00390625`, and the original zero-centroid collapse reproduced. Origin
geometry, label-independent random streams, tie-aware scoring, frozen Stage 6A
hashes, and reserve isolation all passed. This is an implementation audit, not
yet a biological positive result. The reference bridge is queued but remains
blocked pending explicit review; all 62 final-audit trajectories are unloaded.

Protocol: `CA_MOTIF_LINEAGE_STAGE6AR_PROTOCOL.md`. Implementation:
`plastic_ca/motif_minimality_repair.py`. Tests:
`tests/test_motif_minimality_repair.py`.
<!-- CA_MOTIF_STAGE6AR_IMPLEMENTATION_END -->

<!-- CA_MOTIF_STAGE6BR_IMPLEMENTATION_START -->
## CA motif-lineage Stage 6B-R — renewal, coverage, and scale

Implemented on 2026-08-25 as a new clean-room campaign. The frozen Stage-6A-R
result remains negative: the corrected whole-lattice anchor worked, but none of
the registered radius-at-most-five objects survived generation eight. The
posthoc traces nevertheless exposed a narrower lead. Several local carriers
affected generations one through four, founder clamping restored the effect,
and carrier magnitude remained nonzero while family identity rotated toward
chance. Stage 6B-R asks whether renewal geometry—not the existence of a local
signal—is the missing ingredient.

The registered search crosses five writer windows, three shared label-blind
writer fits, and four slow-latch turnover rates. Six diverse candidates are
then compared with raw and Hamming(8,4) transmission, including a paired 1%
carrier-corruption condition. Survivors cross radii five through eight and
one-, two-, and four-seed constellations. Multi-seed designs distinguish full
payload replication from a fixed-total-bit partition; zero-valued encoded
channels remain physical channels. Scale tests use nearest-seed distance,
verify exact absence of carrier support outside the union of local light cones,
and separately report inherited bits, temporary field/writer values, routing
work, seed density, and shared parameters.

All visible daughters are reset to the same matched board before every
generation. Random streams are semantic and independent of candidate labels,
ordering, worker count, and resume boundaries. Development uses disjoint
already-exposed cohorts. The 62 protected pairs remain unloaded unless at most
two finite candidates pass strict generation-16 causality and generation-64
endurance; only then is `FINAL_AUDIT_DESIGN.json` sealed and the automatically
authorized reserve opened without retuning.

The final end-to-end engineering smoke run completed all five corrected phases,
230 checkpoints, and 16x16/32x32/64x64 scale cases in 6 minutes 10 seconds. Its
protected reserve remained closed. Fifteen focused Stage-6B-R tests exercise
the positive sealing branch as well as the ordinary negative branch, and the
full 199-test repository suite passes. Smoke effects are not scientific
results. The reference output and its immutable design digest are recorded only
by the detached registered run.

Protocol: `CA_MOTIF_LINEAGE_STAGE6BR_PROTOCOL.md`. Implementation:
`plastic_ca/motif_renewal_repair.py`. Tests:
`tests/test_motif_renewal_repair.py`.
<!-- CA_MOTIF_STAGE6BR_IMPLEMENTATION_END -->
