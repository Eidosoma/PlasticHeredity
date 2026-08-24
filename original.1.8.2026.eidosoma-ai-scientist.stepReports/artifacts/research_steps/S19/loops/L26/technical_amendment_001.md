# S19-L26 technical-only amendment 001

Attempt 001 completed the frozen feature extraction and development-label null, then stopped before validation-label permutation aggregation with `IndexError: index 588 is out of bounds for axis 0 with size 588`.

The filtered validation table retained inherited row labels and the permutation helper treated them as positional NumPy indices. The repair reset the helper index before grouped permutation. It also reused the already frozen deterministic 15-neighbor identities for the 512 development-label permutations instead of recomputing identical distances.

No target, cohort, firewall, landmark, horizon, representation, normalization, neighbor count, tie rule, label, random stream, statistic, gate, or classification changed. The authoritative run started from fresh caches and passed exact representation, library, prediction, suffix, regeneration, and artifact-hash checks.

