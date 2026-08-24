# Clean-room boundary

This directory is an independent implementation of the registered Wagner
memory tests. It must not import, execute, copy, inspect, or translate source
code, tests, scripts, notebooks, or executable configuration from `NewIdeas`.

The only NewIdeas inputs permitted during design were the prose and retained
non-executable result data listed, with hashes, in `SOURCE_BOUNDARY.md`. The
implementation below was written from those scientific descriptions and an
independently specified Wagner contract. NewIdeas paths are forbidden runtime
dependencies. A validation test scans the installed source and protocol for
such imports or path access.

The existing `grn_f12_realistic` project and its completed run are separate
evidence. They are not imported by this package and are not rerun here.

Every scientific run copies the canonical protocol, records its SHA-256 digest,
records source and environment hashes, and refuses to resume after a digest
change. Smoke tests and admission benchmarks use discarded semantic seed
domains and cannot receive a scientific verdict.

