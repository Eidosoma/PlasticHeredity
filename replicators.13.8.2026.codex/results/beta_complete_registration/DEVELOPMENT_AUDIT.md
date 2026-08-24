# Beta-completeness development audit

The retained fivefold development outcomes were used without resimulation. All legacy coordinates and 64,000 target rows were reconstructed before fitting.

| Check | Result |
|---|---:|
| Legacy state/history/beta arrays exact | True |
| Retained target rows validated | 64000 |
| Provenance records cover every raw feature | True |
| Added-block PCA components | 0 |
| Whole-matrix CV folds | 5 |
| Portable predictions within 1e-12 | True |

## Registered dimensions and penalties

| Candidate | H10 | State | Beta | Interaction | Lambda state/beta/interaction |
|---|---:|---:|---:|---:|---|
| 02 | 9 | 11 | 240 | 64 | 0.1 / 10 / 10 |
| 03 | 10 | 12 | 240 | 64 | 0.01 / 1 / 10 |
