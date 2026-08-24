#!/usr/bin/env python3
"""Regenerate and verify one canonical S06 trajectory in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_gard_reproducibility import (
    capture_identity_from_workspace,
    deserialize_envelope,
    make_envelope,
    regenerate_independent_trajectory,
    registry_boundary_from_workspace,
    serialize_envelope,
    validate_trajectory_invariants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_bytes = args.input.read_bytes()
    reference = deserialize_envelope(reference_bytes, require_canonical=True)
    validate_trajectory_invariants(reference["payload"])
    identity = capture_identity_from_workspace(
        repository_root=REPOSITORY_ROOT,
        artifacts_root=args.artifacts_dir.resolve(),
    )
    registry = registry_boundary_from_workspace(
        artifacts_root=args.artifacts_dir.resolve()
    )
    regenerated_payload = regenerate_independent_trajectory(
        reference_payload=reference["payload"],
        capture_identity=identity,
        registry_boundary=registry,
    )
    regenerated = make_envelope(regenerated_payload)
    regenerated_bytes = serialize_envelope(regenerated)
    exact = regenerated_bytes == reference_bytes
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(regenerated_bytes)
    result = {
        "researchStepId": "S06",
        "success": exact,
        "status": "exact_same_engine_regeneration_pass" if exact else "failure",
        "input": str(args.input),
        "output": None if args.output is None else str(args.output),
        "expectedPayloadSha256": reference["payloadSha256"],
        "regeneratedPayloadSha256": regenerated["payloadSha256"],
        "canonicalBytesExact": exact,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
