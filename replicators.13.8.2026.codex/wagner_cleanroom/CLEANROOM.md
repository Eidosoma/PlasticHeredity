# Clean-room boundary

This project is a source-clean reimplementation. Its executable code was written
from the parent preprint, prose protocols, documented data schemas, and ordinary
numerical-library documentation. It must never import, execute, copy, or read the
earlier experimental implementation at runtime.

Permitted design inputs were prose (`.md`) and machine-readable scientific
artifacts (`.json`, `.jsonl.gz`, checksum and runtime manifests). Earlier Python,
shell, notebooks, tests, and generated source were out of bounds. Earlier result
values are not runtime inputs and cannot select seeds, cohorts, thresholds,
models, or gates.

The `validate` command statically checks executable imports and path literals.
Scientific commands also require the current registration digest and refuse an
output directory registered to another protocol. A completed fresh run is sealed
before any numerical comparison with earlier results.

This is source-code clean, not outcome-blind: the direction of the earlier Wagner
result was already known. The safeguards are therefore fresh semantic seed
domains, frozen gates, untouched evaluation rulebooks, complete reporting, and
no outcome-conditioned substitutions.

