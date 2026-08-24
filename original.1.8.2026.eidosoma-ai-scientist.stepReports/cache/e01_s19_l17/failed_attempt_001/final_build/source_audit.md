# BreakingGRNMemories source audit

BreakingGRNMemories was pinned at commit `afe44231ad3ce915172cdb53a6b234bd76fcb6a5` (tree `56f66ab8b57a2c60e830370842926708eee0767d`). No license file, test suite, Phi fixture, or raw Phi input bundle was present. Source is not redistributed.

The numerical information core is the corrected IIGR lineage: corrected z-scoring, global-signal regression, lag-one residualization, slow bidirectional Gaussian MI summed across directions, an additive `1e-6` graph floor, unnormalized Fiedler strict-sign partition, arithmetic partition means, unregularized Gaussian entropy, the shared PhiID lattice, and both `emergence` and corrected `local_phi_r` identities. Current `phi.py` retains only nonfinite-to-zero `emergence`.

The tracked `info.txt` is not regenerable from any visible exact script state: the original script declared six phases and two measures but stopped before its loop; the current script declares six phases and one measure; the tracked file contains five phases and two measures. The repository does not specify a GARD adapter or a past-only/prefix refit. L17 therefore treats it as related-team lineage inspiration, never as proof of the unavailable author implementation.
