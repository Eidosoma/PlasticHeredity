# Methods and audit map

- `../protocol/protocol.json`: sealed scientific design and implementation hashes
- `../protocol/confirmation_selection.json`: outcome-dependent engineering selection frozen before confirmation
- `calibration_rule_metrics.csv`: equal-rule engineering summaries
- `primary_rule_metrics.csv`: single-form confirmation summaries (when authorized)
- `multiform_rule_metrics.csv`: reciprocal two-form summaries (when authorized)
- `primary_summary.json`: registered gates and classification
- `../verification/verification_audit.json`: exact replay and checksum audit

No full daughter trajectories are discarded without an audit trace: each future checkpoint contains its SHA-256 daughter-trajectory digest plus the complete boundary-H, target-H, and carrier-decoding traces. Full verification regenerates every future and compares those arrays.
