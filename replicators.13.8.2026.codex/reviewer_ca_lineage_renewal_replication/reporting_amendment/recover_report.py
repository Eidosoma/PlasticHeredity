"""Serialize the sealed adjudication after a nonfinite diagnostic ratio.

This file is deliberately outside the registered simulation module manifest.
It does not simulate, alter checkpoints, or change adjudication.  It replaces
nonfinite diagnostic values with JSON null and records that transformation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from reviewer_ca_lineage_renewal_replication.campaign import (
    _checkpoint_payloads,
    _load_registration,
    _raw_rows,
    _write_csv,
    _write_status,
)
from reviewer_ca_lineage_renewal_replication.contract import (
    CONTRACT,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_json,
)
from reviewer_ca_lineage_renewal_replication.inference import adjudicate


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PACKAGE_ROOT / "artifacts"


def _sanitize(value: Any, path: str = "$") -> tuple[Any, list[dict[str, str]]]:
    if isinstance(value, float) and not math.isfinite(value):
        return None, [{"path": path, "original_value": str(value)}]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        changes: list[dict[str, str]] = []
        for key, child in value.items():
            clean_child, child_changes = _sanitize(child, f"{path}.{key}")
            clean[str(key)] = clean_child
            changes.extend(child_changes)
        return clean, changes
    if isinstance(value, list):
        clean_list: list[Any] = []
        changes = []
        for index, child in enumerate(value):
            clean_child, child_changes = _sanitize(child, f"{path}[{index}]")
            clean_list.append(clean_child)
            changes.extend(child_changes)
        return clean_list, changes
    return value, []


def _metric_text(metric: Mapping[str, Any]) -> str:
    return f"{metric['mean']:.4f} [{metric['ci'][0]:.4f}, {metric['ci'][1]:.4f}]"


def main() -> None:
    registration = _load_registration(ARTIFACTS)
    confirmation = ARTIFACTS / "confirmation"
    payloads, missing = _checkpoint_payloads(
        confirmation / "checkpoints",
        registration["cohorts"]["confirmation"],
        registration["design_digest"],
    )
    if missing or len(payloads) != 96:
        raise RuntimeError("reporting amendment requires all 96 sealed checkpoints")
    raw_adjudication = adjudicate(
        payloads,
        complete=True,
        expected_pairs=int(registration["profile"]["confirmation_pairs"]),
        resamples=int(registration["profile"]["bootstrap_resamples"]),
    )
    adjudication, changes = _sanitize(raw_adjudication, "$.adjudication")
    if changes != [
        {
            "path": "$.adjudication.no_rewrite_loss_fraction",
            "original_value": "-inf",
        }
    ]:
        raise RuntimeError(f"unexpected nonfinite adjudication fields: {changes}")

    amendment = {
        "schema_version": SCHEMA_VERSION,
        "kind": "reporting_serialization_amendment",
        "date_utc": "2026-08-23",
        "design_digest": registration["design_digest"],
        "trigger": (
            "The registered safe-fraction diagnostic returned negative infinity "
            "because intact generation-8 crossover was non-positive; strict JSON "
            "serialization rejects nonfinite numbers."
        ),
        "transformation": "replace the listed nonfinite diagnostic with JSON null",
        "changed_fields": changes,
        "checkpoint_count": 96,
        "trajectory_rerun": False,
        "checkpoint_change": False,
        "estimand_change": False,
        "gate_change": False,
        "verdict_change": False,
        "sealed_verdict_before_serialization": raw_adjudication["verdict"],
        "registered_inference_sha256": registration["implementation_manifest"][
            "inference.py"
        ],
        "recovery_script_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(confirmation / "REPORTING_AMENDMENT.json", amendment)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "independent_ca_lineage_renewal_stage3r_replication",
        "state": "complete",
        "complete": True,
        "design_digest": registration["design_digest"],
        "pair_count": 96,
        "missing_pair_ids": [],
        "profile": registration["profile"],
        "adjudication": adjudication,
        "reporting_amendment": "REPORTING_AMENDMENT.json",
    }
    atomic_write_json(confirmation / "RESULTS.json", result)
    _write_csv(confirmation / "RAW_OUTCOMES.csv", _raw_rows(payloads))

    verdict = str(adjudication["verdict"])
    report_text = f"""# Independent CA lineage-renewal replication

Verdict: `{verdict}`.

The fixed `motif_energy512-w32-s025-d32` reader and preregistered strict-49--64
daughter writer with universal gain 0.5 were tested on 96 fully fresh matched
founder pairs, 64 futures per history, and 16 visibly reset generations. The
independent unit was the founder pair; intervals use 10,000 pair-cluster
bootstrap draws at alpha 0.0125.

## Original-form persistence

- Generation 4: {_metric_text(adjudication['intact_generation4'])}
- Generation 8: {_metric_text(adjudication['intact_generation8'])}
- Generation 16: {_metric_text(adjudication['intact_generation16'])}
- Terminal observer at generation 8: {_metric_text(adjudication['terminal_generation8'])}

The intact lineage missed the registered original-form gates. Its crossover was
already small at generation 4 and was negative at generations 8 and 16.

## Causal renewal

- No-rewrite generation 8: {_metric_text(adjudication['no_rewrite_generation8'])}
- Active-rewrite advantage at generation 8: {_metric_text(adjudication['active_rewrite_advantage_generation8'])}
- Opposite rescue at generation 4: {_metric_text(adjudication['opposite_rescue_generation4'])}
- Opposite founder at generation 8: {_metric_text(adjudication['opposite_founder_generation8'])}
- One-percent corruption at generation 8: {_metric_text(adjudication['carrier_corruption_generation8'])}

The fading, non-rewritten founder carrier retained a strong signal, while active
daughter rewriting destroyed rather than renewed that signal. The registered
no-rewrite loss fraction is undefined because the intact generation-8
denominator was non-positive; it is encoded as JSON `null`. The directly paired
active-rewrite advantage was negative, so the causal-renewal gate clearly
failed independently of that diagnostic ratio.

The opposite-history rescue reversed generation 4 in the predicted direction,
showing short-horizon steering, but same-history rescue could not establish
durable renewal. The complete gate table is in `RESULTS.json`.

## Reporting amendment

The original report command failed only while serializing negative infinity in
the undefined loss-ratio diagnostic. `REPORTING_AMENDMENT.json` records the
single conversion to JSON `null`. No trajectory, checkpoint, estimand, gate, or
verdict was changed.

Claim boundary: {CONTRACT['claim_boundary']}.
"""
    lay_text = f"""# Lay summary

The replication finished, and the result is `{verdict}`.

The carrier from the original founder could still steer descendants when it
was merely allowed to fade. But when each daughter tried to measure its own
pattern and write a fresh carrier for the next generation, the inherited A/B
signal rapidly collapsed: it was weak by generation 4 and reversed slightly by
generations 8 and 16. In simple terms, this implementation preserved a fading
message better than it copied that message.

Some short-term steering worked—the opposite rescue pushed generation 4 in the
opposite direction—but the full registered causal and durability requirements
did not pass. This clean-room run therefore does not replicate the claimed
self-renewing carrier.

This is a synthetic cellular-automaton result, not a claim about biological
heredity, life, agency, or memory outside the automaton.
"""
    atomic_write_text(confirmation / "REPORT.md", report_text)
    atomic_write_text(confirmation / "LAY_SUMMARY.md", lay_text)
    atomic_write_json(
        confirmation / "STAGE_DECISION.json",
        {
            "design_digest": registration["design_digest"],
            "verdict": verdict,
            "decision": "stop_after_failed_strict_replication_and_review",
            "review_required": True,
            "automatic_continuation": False,
            "reporting_amendment": "REPORTING_AMENDMENT.json",
        },
    )
    atomic_write_json(
        confirmation / "QUEUE.json",
        {
            "state": "awaiting_review",
            "automatic_launch": False,
            "added_experiments": [],
        },
    )
    atomic_write_text(
        confirmation / "REGISTRATION.json",
        (ARTIFACTS / "REGISTRATION.json").read_text(encoding="utf-8"),
    )
    filenames = [
        "REGISTRATION.json",
        "RESULTS.json",
        "RAW_OUTCOMES.csv",
        "REPORT.md",
        "LAY_SUMMARY.md",
        "STAGE_DECISION.json",
        "QUEUE.json",
        "REPORTING_AMENDMENT.json",
    ]
    files = {name: sha256_file(confirmation / name) for name in filenames}
    atomic_write_json(
        confirmation / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": files,
            "seal_digest": sha256_json(files),
            "reporting_amendment": "REPORTING_AMENDMENT.json",
        },
    )
    _write_status(ARTIFACTS)


if __name__ == "__main__":
    main()
