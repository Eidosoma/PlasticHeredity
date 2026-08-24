from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "grn-f12-v1.json"
TIERS = ("continuous", "molecular")
COHORTS = ("calibration", "development", "confirmation")


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_PROTOCOL
    with target.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("format") != "grn-f12-realistic-v1":
        raise ValueError("unsupported protocol format")
    if tuple(protocol.get("landmarks", ())) != (0, 2, 4, 8, 12):
        raise ValueError("the registered landmark panel changed")
    if set(protocol.get("tiers", {})) != set(TIERS):
        raise ValueError("both registered tiers are required")
    if int(protocol["endpoint"]["horizon"]) != 12:
        raise ValueError("the primary endpoint must remain F12")
    if int(protocol["endpoint"]["stable_run"]) != 3:
        raise ValueError("the primary endpoint must remain RUN3")
    if int(protocol["predictor"]["folds"]) != 5:
        raise ValueError("five whole-network folds are registered")
    for tier in TIERS:
        values = protocol["tiers"][tier]
        for key in ("genes", "calibration_networks", "development_networks", "confirmation_networks", "futures"):
            if int(values[key]) <= 0:
                raise ValueError(f"{tier}.{key} must be positive")
        if int(values["futures"]) % 2:
            raise ValueError(f"{tier}.futures must split into equal frozen halves")


def protocol_for_profile(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return an explicit non-scientific reduction or the untouched full contract."""
    if profile not in {"full", "quick", "smoke"}:
        raise ValueError(f"unknown profile: {profile}")
    result = copy.deepcopy(protocol)
    result["profile"] = profile
    if profile == "full":
        return result
    result["scientific"] = False
    if profile == "smoke":
        counts = {
            "continuous": (5, 10, 10, 16, 16, 4),
            "molecular": (5, 10, 10, 16, 16, 4),
        }
        result["predictor"].update(max_epochs=8, patience=3, batch_states=32, width=16)
        result["inference"].update(bootstrap_repetitions=128, permutation_repetitions=64)
        result["operations"].update(shard_networks=2, disk_admission_gib=1, disk_running_gib=1)
    else:
        counts = {
            "continuous": (16, 32, 48, 64, 48, 12),
            "molecular": (12, 24, 32, 48, 32, 12),
        }
        result["predictor"].update(max_epochs=80, patience=12, batch_states=128, width=32)
        result["inference"].update(bootstrap_repetitions=1024, permutation_repetitions=512)
        result["operations"].update(shard_networks=4, disk_admission_gib=5, disk_running_gib=4)
    for tier, values in counts.items():
        calibration, development, confirmation, futures, calibration_futures, controls = values
        result["tiers"][tier].update(
            calibration_networks=calibration,
            development_networks=development,
            confirmation_networks=confirmation,
            futures=futures,
            calibration_futures=calibration_futures,
            control_networks=controls,
        )
    validate_protocol(result)
    return result


def cohort_size(protocol: dict[str, Any], tier: str, cohort: str) -> int:
    if tier not in TIERS or cohort not in COHORTS:
        raise ValueError((tier, cohort))
    return int(protocol["tiers"][tier][f"{cohort}_networks"])

