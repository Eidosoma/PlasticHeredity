# Clean-room boundary

This directory is an independent implementation. It must not import, execute,
copy, or inspect source code from `NewIdeas` or from any earlier PH simulator.
Only prose/data supplied as scientific context may be consulted. At runtime the
package imports only its own modules and declared third-party dependencies.

The existing `wagner_cleanroom` campaign is an external result, not a code or
data dependency. Comparisons with its sealed summaries happen only after this
campaign is sealed. Random streams, network sampling, simulators, endpoints,
predictors, controls, inference, storage, and replay are implemented here.

Every scientific run copies the full protocol, records its SHA-256 digest,
records source and environment hashes, and refuses to resume if registration
has changed. Benchmark and smoke profiles are explicitly non-scientific and
cannot receive a PH verdict.

