# P3 inference-routing recovery

Recovery amendment: `86e0cda5d5403fed3601fcd65110472ed76305ab9f5a1b139a0347abf238e185`.
Prospective lifecycle amendment: `679449881a33f0f40a50ca7e9de8849a1996321492a1b8190f8007f6cc22637c`.

The original P3 execution completed all primary and replay futures, then stopped before inference because its generic caller looked for `RANDOM` instead of the registered `RANDOM_SURGERY` arm. This checksum-sealed recovery explicitly routed `RANDOM_SURGERY`, loaded all completed checkpoints, generated zero futures, verified exact replay and unchanged checkpoint hashes, round-tripped the complete inference, and sealed the result.

No scientific design, data, estimator, threshold, gate, or claim boundary changed. This remains a developmental pilot and stops before confirmation.
