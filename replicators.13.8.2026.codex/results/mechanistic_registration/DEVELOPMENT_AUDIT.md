# Mechanistic ablation development audit

No development futures were resimulated. The retained 5× targets were used only after exact trajectory-derived feature reconstruction and branch-table validation.

| Check | Result |
|---|---:|
| Reconstructed state/graph array exact | True |
| Reconstructed direct-history array exact | True |
| Reconstructed beta-only array exact | True |
| Retained target rows validated | 64000 |
| Direct trailing-run duplicate exact | True |
| Portable registered predictions within 1e-12 | True |

Raw compositions were not retained by the earlier campaign, so they cannot be compared byte-for-byte with a prior file. They are reconstructed from the unchanged seed contract, and all 399 retained state/history/beta coordinates match exactly.
