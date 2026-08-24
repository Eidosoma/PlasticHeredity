# Repository relocation compatibility audit

The Codex replication directory was moved from

```text
/home/robert/Projects/replications/replicators.13.8.2026.codex
```

to

```text
/home/robert/Projects/replications/PlasticHeredity/replicators.13.8.2026.codex
```

The scientific data already reside in the latter, correct location. Several
older checksum-sealed JSON records contain the former absolute path because it
was part of their original provenance and canonical digest. Those archived
files must not be silently rewritten: doing so would change registration IDs
and checksum manifests even though no scientific value changed.

The relocation repair therefore has two strict rules:

1. sealed registrations, protocols, manifests, results, checkpoints, and
   checksum files remain byte-for-byte unchanged; and
2. active verification maps a known historical repository root to the current
   root only for filesystem access, and ignores only that root prefix when
   comparing a stored protocol with its reconstructed form.

It does not ignore relative paths, filenames, IDs, hashes, cohort parameters,
seeds, arms, outcomes, or any other protocol field. A mapped target must exist.

The audit identified historical absolute paths in three classes:

- active P1, P2, and P3 recovery protocols, which require relocation-aware
  verification;
- completed result and registration manifests, where the paths are preserved
  historical metadata and are not execution inputs; and
- captured pytest/log text, which is immutable diagnostic output.

The repository-wide JSON scan found 58 historical path strings across 37
files. Fifty-seven map to existing objects under the current repository root.
The sole missing target is the disposable hidden checkpoint directory named in
the already sealed P3c confirmation manifest; the checksum-sealed P3c result
and replay audit remain present. No scientific result, registration, model, or
required replay artifact is missing.

Four older, non-intervention verification entry points (the v1 mechanistic,
beta-completeness, regime-development, and regime-ensemble bundles) also
dereference their archived absolute paths directly. Their sealed artifacts are
present, but those legacy verification commands are not relocation-portable.
They are outside the CR3 execution path and are recorded here for a future
standalone provenance-maintenance amendment rather than silently changing
their historical source closures during CR3.

Relocation-specific tests cover current-to-current access, legacy-to-current
mapping, rejection of missing targets, root-normalized protocol equivalence,
and detection of every non-path protocol change. The complete repository suite
must pass after the compatibility alias is removed.
