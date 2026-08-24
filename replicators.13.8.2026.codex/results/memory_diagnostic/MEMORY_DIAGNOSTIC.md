# Retrospective IID-support diagnostic

## Outcome

This regenerates the existing clean-room 12-fission futures and measures the reviewer-identified support mismatch. It is diagnostic only and does not confirm a memory claim.

| Candidate | Legacy mismatched Markov gain | Corrected Markov gain | Legacy minus corrected IID loss |
|---|---:|---:|---:|
| 02 | 0.057430 | 0.056881 | 0.000550 |
| 03 | 0.038494 | 0.038103 | 0.000391 |

Positive values in the last column mean that the mismatched IID fit makes the apparent Markov advantage larger; negative values mean it makes it smaller.

## Audit boundary

All 128000 retained branch rows and the original batch digest matched exactly: **True**.

The unavailable original L44 sequences are not present here, so this cannot repair or reproduce the preprint's numerical 0.015–0.022-bit result.
