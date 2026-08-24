from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .continuous import simulate_continuous_futures
from .endpoint import event_summary
from .molecular import simulate_molecular_futures
from .network import sample_network
from .rng import array_digest, generator, jax_key
from .statistics import bootstrap_mean
from .storage import load_npz, write_json_atomic, write_npz_atomic


ARMS = ("self", "exact_transplant", "basal_reset", "node_shuffle", "inheritance_erased")


def _continuous_arm(protocol, network, shard, state_index, arm, futures, key, permutation):
    state = shard["states_x"][state_index]
    baseline = shard["baseline_x"]
    erase = None
    if arm == "basal_reset":
        state = baseline
    elif arm == "node_shuffle":
        state = state[permutation]
    elif arm == "inheritance_erased":
        erase = baseline
    similarities, endpoint, trajectory = simulate_continuous_futures(
        network, state, protocol, futures, key,
        horizon=int(protocol["endpoint"]["horizon"]), executor="scan", erase_state=erase,
    )
    return similarities, array_digest(endpoint, trajectory, decimals=6)


def _molecular_arm(protocol, network, shard, state_index, arm, futures, key, permutation):
    mrna = shard["states_mrna"][state_index]
    protein = shard["states_protein"][state_index]
    baseline = (shard["baseline_mrna"], shard["baseline_protein"])
    erase = None
    if arm == "basal_reset":
        mrna, protein = baseline
    elif arm == "node_shuffle":
        mrna, protein = mrna[permutation], protein[permutation]
    elif arm == "inheritance_erased":
        erase = baseline
    similarities, endpoint_mrna, endpoint, mrna_trajectory, protein_trajectory = simulate_molecular_futures(
        network, mrna, protein, protocol, futures, key,
        horizon=int(protocol["endpoint"]["horizon"]), executor="scan", erase_state=erase,
    )
    return similarities, array_digest(endpoint_mrna, endpoint, mrna_trajectory, protein_trajectory, decimals=6)


def run_controls_tier(run_dir: str | Path, tier: str, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    predictions = load_npz(root / "predictions" / f"{tier}.npz")
    count = int(protocol["tiers"][tier]["control_networks"])
    futures = int(protocol["tiers"][tier]["futures"])
    threshold_data = __import__("json").loads((root / "calibration" / f"{tier}.json").read_text(encoding="utf-8"))
    threshold = float(threshold_data["thresholds"]["q05"])
    event_counts = np.empty((count, 2, len(ARMS)), dtype=np.int32)
    selected_states = np.empty((count, 2), dtype=np.int8)
    digests = np.empty((count, 2, len(ARMS)), dtype="U64")
    master = str(protocol["master_seed_label"])
    for network_row in range(count):
        network_index = int(predictions["network_index"][network_row])
        probability = predictions["full_event"][network_row]
        selected_states[network_row] = (int(np.argmin(probability)), int(np.argmax(probability)))
        shard = load_npz(root / "data" / "confirmation" / tier / f"network_{network_index:04d}.npz")
        network = sample_network(protocol, tier, "confirmation", network_index)
        permutation = generator(master, "control-node-permutation", tier, network_index).permutation(
            int(protocol["tiers"][tier]["genes"])
        )
        key = jax_key(master, "control-common-randomness", tier, network_index)
        for risk_index, state_index in enumerate(selected_states[network_row]):
            self_events: np.ndarray | None = None
            self_digest = ""
            for arm_index, arm in enumerate(ARMS):
                if tier == "continuous":
                    similarities, digest = _continuous_arm(
                        protocol, network, shard, int(state_index), arm, futures, key, permutation
                    )
                else:
                    similarities, digest = _molecular_arm(
                        protocol, network, shard, int(state_index), arm, futures, key, permutation
                    )
                summary = event_summary(similarities, threshold)
                events = np.asarray(summary["event"], dtype=np.uint8)
                event_counts[network_row, risk_index, arm_index] = int(events.sum())
                digests[network_row, risk_index, arm_index] = digest
                if arm == "self":
                    self_events, self_digest = events, digest
                if arm == "exact_transplant" and (
                    self_events is None or not np.array_equal(events, self_events) or digest != self_digest
                ):
                    raise RuntimeError("exact transplant failed pathwise common-random-number identity")
    output = root / "controls" / f"{tier}.npz"
    write_npz_atomic(
        output, network_index=predictions["network_index"][:count], selected_states=selected_states,
        event_counts=event_counts, trajectory_digest=digests, arms=np.asarray(ARMS),
        futures=np.asarray(futures), threshold=np.asarray(threshold),
    )
    gaps = (event_counts[:, 1, :] - event_counts[:, 0, :]) / float(futures)
    repetitions = int(protocol["inference"]["bootstrap_repetitions"])
    summaries = {
        arm: bootstrap_mean(gaps[:, index], repetitions, master, f"control|{tier}|{arm}", len(ARMS))
        for index, arm in enumerate(ARMS)
    }
    self_gap = summaries["self"]["estimate"]
    denominator = max(abs(self_gap), 1e-12)
    retention = {arm: summaries[arm]["estimate"] / denominator for arm in ARMS}
    gates = protocol["gates"]
    checks = {
        "self_gap": self_gap >= float(gates["control_gap_minimum"]),
        "self_positive_bound": summaries["self"]["adjusted_lower"] > 0.0,
        "exact_transplant": retention["exact_transplant"] >= float(gates["transplant_retention_minimum"]),
        "basal_reset": abs(retention["basal_reset"]) <= float(gates["null_retention_maximum"]),
        "node_shuffle": abs(retention["node_shuffle"]) <= float(gates["null_retention_maximum"]),
        "inheritance_erased": abs(retention["inheritance_erased"]) <= float(gates["null_retention_maximum"]),
    }
    result = {
        "format": "grn-f12-controls-tier-v1", "tier": tier, "networks": count,
        "summaries": summaries, "retention": retention, "checks": checks,
        "pass": bool(all(checks.values())),
    }
    write_json_atomic(root / "analysis" / f"controls_{tier}.json", result)
    return result


def combine_controls(run_dir: str | Path) -> dict[str, Any]:
    import json

    root = Path(run_dir)
    tiers = {
        tier: json.loads((root / "analysis" / f"controls_{tier}.json").read_text(encoding="utf-8"))
        for tier in ("continuous", "molecular")
    }
    result = {
        "format": "grn-f12-controls-v1", "tiers": tiers,
        "pass": bool(tiers["continuous"]["pass"] and tiers["molecular"]["pass"]),
    }
    write_json_atomic(root / "analysis" / "controls.json", result)
    return result

