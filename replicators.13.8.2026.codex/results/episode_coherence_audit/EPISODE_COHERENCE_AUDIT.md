# Exploratory episode-coherence audit

## Status and outcome

This is a post-hoc descriptive audit of existing positive `JOINT_BREAK_RUN3` futures. It does not redefine or prospectively confirm the endpoint. Every reported episode is still, first and foremost, a break followed by three inherited fissions.

The table below describes the first qualifying episode. Similarities are cosine values; larger pairwise values mean tighter episode geometry, while smaller old-anchor values mean greater separation from the pre-break composition.

| Cohort | Candidate | Episodes | Mean minimum pairwise H [95% CI] | Mean first-to-last H [95% CI] | Mean maximum old-anchor H [95% CI] | Same-run persist-5, resolved [95% CI] | Second renewal after later break in F12 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| scaled5 | 02 | 23752 | 0.6869 [0.6824, 0.6916] | 0.6924 [0.6880, 0.6967] | 0.6850 [0.6763, 0.6945] | 0.7728 [0.7532, 0.7921] | 0.1054 [0.0978, 0.1124] |
| scaled5 | 03 | 26671 | 0.7003 [0.6960, 0.7048] | 0.7059 [0.7017, 0.7102] | 0.6801 [0.6717, 0.6890] | 0.7594 [0.7411, 0.7778] | 0.1155 [0.1092, 0.1214] |
| MECHCONF | 02 | 22868 | 0.6805 [0.6751, 0.6859] | 0.6864 [0.6813, 0.6917] | 0.6801 [0.6697, 0.6900] | 0.7752 [0.7553, 0.7962] | 0.1045 [0.0971, 0.1119] |
| MECHCONF | 03 | 25494 | 0.6960 [0.6910, 0.7009] | 0.7017 [0.6966, 0.7066] | 0.6729 [0.6636, 0.6822] | 0.7601 [0.7404, 0.7800] | 0.1166 [0.1082, 0.1245] |
| MECHCONF2 | 02 | 22033 | 0.6884 [0.6833, 0.6936] | 0.6941 [0.6889, 0.6991] | 0.6969 [0.6879, 0.7061] | 0.7891 [0.7706, 0.8090] | 0.1021 [0.0950, 0.1090] |
| MECHCONF2 | 03 | 24698 | 0.7040 [0.6986, 0.7094] | 0.7096 [0.7046, 0.7149] | 0.6936 [0.6840, 0.7033] | 0.7796 [0.7614, 0.7992] | 0.1077 [0.1007, 0.1143] |

## Threshold sensitivity

These cutoffs were chosen after the original result and are sensitivity views, not discovery gates. The least restrictive view requires minimum pairwise daughter similarity `>0.90` and maximum similarity to the old anchor `<=0.90`.

| Cohort | Candidate | Coherent [95% CI] | Distinct [95% CI] | Both [95% CI] | Both + persist-5 among resolved [95% CI] |
|---|---:|---:|---:|---:|---:|
| scaled5 | 02 | 0.0469 [0.0406, 0.0535] | 0.9426 [0.9343, 0.9496] | 0.0395 [0.0339, 0.0458] | 0.0364 [0.0311, 0.0422] |
| scaled5 | 03 | 0.0583 [0.0517, 0.0656] | 0.9407 [0.9328, 0.9475] | 0.0485 [0.0425, 0.0552] | 0.0463 [0.0402, 0.0530] |
| MECHCONF | 02 | 0.0475 [0.0414, 0.0542] | 0.9491 [0.9416, 0.9556] | 0.0414 [0.0357, 0.0476] | 0.0382 [0.0326, 0.0443] |
| MECHCONF | 03 | 0.0650 [0.0579, 0.0729] | 0.9394 [0.9310, 0.9469] | 0.0552 [0.0488, 0.0621] | 0.0507 [0.0444, 0.0574] |
| MECHCONF2 | 02 | 0.0453 [0.0395, 0.0520] | 0.9359 [0.9274, 0.9437] | 0.0379 [0.0326, 0.0440] | 0.0338 [0.0287, 0.0398] |
| MECHCONF2 | 03 | 0.0634 [0.0553, 0.0719] | 0.9324 [0.9236, 0.9403] | 0.0532 [0.0464, 0.0605] | 0.0489 [0.0425, 0.0562] |

## Branch-half consistency

The two preassigned branch halves are descriptive consistency checks. No cohort, candidate, or half is pooled to rescue disagreement.

| Cohort | Candidate | Half | Coherent >0.90 | Distinct <=0.90 | Same-run persist-5, resolved | Second renewal after later break |
|---|---:|---:|---:|---:|---:|---:|
| scaled5 | 02 | A | 0.0474 | 0.9418 | 0.7690 | 0.1038 |
| scaled5 | 02 | B | 0.0464 | 0.9433 | 0.7766 | 0.1070 |
| scaled5 | 03 | A | 0.0572 | 0.9404 | 0.7614 | 0.1174 |
| scaled5 | 03 | B | 0.0594 | 0.9410 | 0.7574 | 0.1136 |
| MECHCONF | 02 | A | 0.0462 | 0.9518 | 0.7756 | 0.1057 |
| MECHCONF | 02 | B | 0.0488 | 0.9464 | 0.7747 | 0.1033 |
| MECHCONF | 03 | A | 0.0640 | 0.9388 | 0.7595 | 0.1175 |
| MECHCONF | 03 | B | 0.0661 | 0.9400 | 0.7607 | 0.1158 |
| MECHCONF2 | 02 | A | 0.0447 | 0.9378 | 0.7873 | 0.1039 |
| MECHCONF2 | 02 | B | 0.0460 | 0.9340 | 0.7909 | 0.1003 |
| MECHCONF2 | 03 | A | 0.0619 | 0.9337 | 0.7804 | 0.1058 |
| MECHCONF2 | 03 | B | 0.0649 | 0.9311 | 0.7788 | 0.1095 |

## Interpretation boundary

Even a high descriptive coherence rate would not make the original target a registered regime test: the endpoint never required coherence, distinctness, persistence, or recurrence. The F12 second-renewal summary is not a test of return to the same composition. Conversely, weak geometry directly argues against regime language.

The defensible claim remains that a break followed by renewed short-run inheritance has a reproducible, predictable probability. A distinct new hereditary regime requires a frozen coherence/distinctness/persistence endpoint on another untouched cohort.

## Replay audit

| Cohort | Archived futures | Positive episodes replayed | State reconstruction | Target/process replay |
|---|---:|---:|---:|---:|
| scaled5 | 128000 | 50423 | True | True / continuous within 1e-14: True |
| MECHCONF | 128000 | 48362 | True | True / continuous within 1e-14: True |
| MECHCONF2 | 128000 | 46731 | True | True / continuous within 1e-14: True |
