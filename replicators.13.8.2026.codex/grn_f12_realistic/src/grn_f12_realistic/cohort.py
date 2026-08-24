from __future__ import annotations

from typing import Any

import numpy as np

from .continuous import acquire_continuous_history, simulate_continuous_futures
from .endpoint import calibrated_threshold, classify_f12, event_summary, phenotype_similarity
from .features import continuous_features, molecular_features
from .molecular import acquire_molecular_history, simulate_molecular_futures
from .network import Network, network_arrays, sample_network
from .rng import array_digest, jax_key


def _future_key(protocol: dict[str, Any], tier: str, cohort: str, network_index: int, cue_index: int, age: int, arm: str = "observational"):
    return jax_key(
        str(protocol["master_seed_label"]), "future", tier, cohort, int(network_index),
        int(cue_index), int(age), arm,
    )


def calibrate_one(protocol: dict[str, Any], tier: str, network_index: int, executor: str = "scan") -> dict[str, np.ndarray]:
    network = sample_network(protocol, tier, "calibration", network_index)
    cfg = protocol["tiers"][tier]
    futures = int(cfg["calibration_futures"])
    horizon = int(protocol["endpoint"]["horizon"])
    key = jax_key(str(protocol["master_seed_label"]), "calibration-future", tier, network_index)
    if tier == "continuous":
        baseline, _ = acquire_continuous_history(network, protocol)
        similarities, endpoint, trajectory = simulate_continuous_futures(
            network, baseline, protocol, futures, key, horizon=horizon, executor=executor
        )
        digest = array_digest(endpoint, trajectory, similarities, decimals=6)
    elif tier == "molecular":
        baseline, _, _ = acquire_molecular_history(network, protocol)
        similarities, endpoint_mrna, endpoint, mrna_trajectory, protein_trajectory = simulate_molecular_futures(
            network, baseline[0], baseline[1], protocol, futures, key, horizon=horizon, executor=executor
        )
        digest = array_digest(endpoint_mrna, endpoint, mrna_trajectory, protein_trajectory, similarities, decimals=6)
    else:
        raise ValueError(tier)
    quantiles = np.quantile(similarities.ravel(), [0.025, 0.05, 0.1])
    return {
        "network_index": np.asarray(network_index, dtype=np.int32),
        "network_uid": np.asarray(network.uid),
        "quantiles": quantiles.astype(np.float64),
        "similarity_count": np.asarray(similarities.size, dtype=np.int64),
        "trajectory_digest": np.asarray(digest),
    }


def combine_calibration(rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["network_index"]))
    quantiles = np.stack([row["quantiles"] for row in ordered])
    if len(np.unique([int(row["network_index"]) for row in ordered])) != len(ordered):
        raise RuntimeError("duplicate calibration network coordinate")
    return {
        "networks": len(ordered),
        "thresholds": {
            "q025": float(np.median(quantiles[:, 0])),
            "q05": float(np.median(quantiles[:, 1])),
            "q10": float(np.median(quantiles[:, 2])),
        },
        "per_network_quantiles": quantiles.tolist(),
    }


def _state_coordinates(protocol: dict[str, Any]):
    for cue_index in range(2):
        for landmark_index, age in enumerate(protocol["landmarks"]):
            yield cue_index, landmark_index, int(age)


def simulate_one(
    protocol: dict[str, Any],
    tier: str,
    cohort: str,
    network_index: int,
    threshold: float,
    *,
    executor: str = "scan",
    sensitivity_thresholds: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    network = sample_network(protocol, tier, cohort, network_index)
    futures = int(protocol["tiers"][tier]["futures"])
    horizon12 = int(protocol["endpoint"]["horizon"])
    horizon24 = int(protocol["endpoint"]["secondary_horizon"])
    states = 2 * len(protocol["landmarks"])
    genes = int(protocol["tiers"][tier]["genes"])
    history_features = np.empty((states, 10), dtype=np.float32)
    structural_features = np.empty((states, 16), dtype=np.float32)
    node_features = np.empty((states, genes, 8), dtype=np.float32)
    event_count = np.empty(states, dtype=np.int32)
    break_count = np.empty(states, dtype=np.int32)
    event_half0 = np.empty(states, dtype=np.int32)
    event_half1 = np.empty(states, dtype=np.int32)
    break_half0 = np.empty(states, dtype=np.int32)
    break_half1 = np.empty(states, dtype=np.int32)
    run5_count = np.empty(states, dtype=np.int32)
    f24_count = np.empty(states, dtype=np.int32)
    event_count_q025 = np.empty(states, dtype=np.int32)
    event_count_q10 = np.empty(states, dtype=np.int32)
    coherence_mean = np.empty(states, dtype=np.float32)
    mean_similarity = np.empty(states, dtype=np.float32)
    old_anchor_separation = np.empty(states, dtype=np.float32)
    cue_indices = np.empty(states, dtype=np.int8)
    ages = np.empty(states, dtype=np.int8)
    events = np.empty((states, futures), dtype=np.uint8)
    breaks = np.empty((states, futures), dtype=np.uint8)
    endpoint_digests = np.empty(states, dtype="U64")
    trajectory_digests = np.empty(states, dtype="U64")

    result = network_arrays(network)
    if tier == "continuous":
        baseline, history_states = acquire_continuous_history(network, protocol)
        endpoints = np.empty((states, futures, genes), dtype=np.float32)
        result["baseline_x"] = baseline.astype(np.float32)
        result["states_x"] = history_states.reshape(states, genes).astype(np.float32)
    else:
        baseline, history_mrna, history_protein = acquire_molecular_history(network, protocol)
        endpoints = np.empty((states, futures, genes), dtype=np.int32)
        endpoint_mrna = np.empty((states, futures, genes), dtype=np.int32)
        result["baseline_mrna"] = baseline[0].astype(np.int32)
        result["baseline_protein"] = baseline[1].astype(np.int32)
        result["states_mrna"] = history_mrna.reshape(states, genes).astype(np.int32)
        result["states_protein"] = history_protein.reshape(states, genes).astype(np.int32)

    for row, (cue_index, landmark_index, age) in enumerate(_state_coordinates(protocol)):
        key = _future_key(protocol, tier, cohort, network_index, cue_index, age)
        if tier == "continuous":
            start = history_states[cue_index, landmark_index]
            similarities, endpoint, trajectory = simulate_continuous_futures(
                network, start, protocol, futures, key, horizon=horizon24, executor=executor
            )
            endpoints[row] = endpoint
            history_features[row], structural_features[row], node_features[row] = continuous_features(
                network, start, cue_index, age, protocol
            )
            old_similarity = phenotype_similarity(np.broadcast_to(start, endpoint.shape), endpoint, tier)
        else:
            start_mrna = history_mrna[cue_index, landmark_index]
            start_protein = history_protein[cue_index, landmark_index]
            similarities, final_mrna, endpoint, mrna_trajectory, protein_trajectory = simulate_molecular_futures(
                network, start_mrna, start_protein, protocol, futures, key, horizon=horizon24, executor=executor
            )
            endpoint_mrna[row] = final_mrna
            endpoints[row] = endpoint
            history_features[row], structural_features[row], node_features[row] = molecular_features(
                network, start_mrna, start_protein, cue_index, age, protocol
            )
            old_similarity = phenotype_similarity(np.broadcast_to(start_protein, endpoint.shape), endpoint, tier)
        primary = event_summary(similarities[:, :horizon12], threshold)
        broken24, event24, _ = classify_f12(similarities, threshold, run=3)
        event_count[row] = int(primary["event_count"])
        break_count[row] = int(primary["break_count"])
        event_half0[row] = int(primary["event_half0"])
        event_half1[row] = int(primary["event_half1"])
        break_half0[row] = int(primary["break_half0"])
        break_half1[row] = int(primary["break_half1"])
        run5_count[row] = int(np.asarray(primary["run5"]).sum())
        f24_count[row] = int(event24.sum())
        sensitivity = sensitivity_thresholds or {"q025": threshold, "q10": threshold}
        event_count_q025[row] = int(classify_f12(
            similarities[:, :horizon12], float(sensitivity["q025"]), run=3
        )[1].sum())
        event_count_q10[row] = int(classify_f12(
            similarities[:, :horizon12], float(sensitivity["q10"]), run=3
        )[1].sum())
        coherence_mean[row] = float(np.mean(np.asarray(primary["maximum_run"])) / horizon12)
        mean_similarity[row] = float(primary["mean_similarity"])
        old_anchor_separation[row] = float(1.0 - np.mean(old_similarity))
        cue_indices[row] = cue_index
        ages[row] = age
        events[row] = np.asarray(primary["event"], dtype=np.uint8)
        breaks[row] = np.asarray(primary["break"], dtype=np.uint8)
        # The cross-executor trace digest is deliberately based on the registered
        # thresholded phenotype trace. Floating kernels are compared separately
        # with a tight tolerance; a one-ulp fusion difference must not masquerade
        # as a biological replay failure.
        trajectory_digests[row] = array_digest(
            (similarities > threshold).astype(np.uint8), events[row], breaks[row]
        )
        endpoint_digests[row] = array_digest(endpoint, events[row], breaks[row], decimals=6)

    result.update({
        "history_features": history_features,
        "structural_features": structural_features,
        "node_features": node_features,
        "event_count": event_count,
        "break_count": break_count,
        "event_half0": event_half0,
        "event_half1": event_half1,
        "break_half0": break_half0,
        "break_half1": break_half1,
        "run5_count": run5_count,
        "f24_count": f24_count,
        "event_count_q025": event_count_q025,
        "event_count_q10": event_count_q10,
        "coherence_mean": coherence_mean,
        "mean_similarity": mean_similarity,
        "old_anchor_separation": old_anchor_separation,
        "cue_index": cue_indices,
        "age": ages,
        "events": events,
        "breaks": breaks,
        "endpoints": endpoints,
        "endpoint_digest": endpoint_digests,
        "trajectory_digest": trajectory_digests,
        "futures": np.asarray(futures, dtype=np.int32),
        "threshold": np.asarray(threshold, dtype=np.float64),
    })
    if tier == "molecular":
        result["endpoint_mrna"] = endpoint_mrna
    return result


def audit_one(
    protocol: dict[str, Any], tier: str, network_index: int, threshold: float,
    stored: dict[str, np.ndarray], sensitivity_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    regenerated = simulate_one(
        protocol, tier, "confirmation", network_index, threshold, executor="loop",
        sensitivity_thresholds=sensitivity_thresholds,
    )
    endpoint_equal = np.array_equal(regenerated["endpoints"], stored["endpoints"])
    endpoint_close = bool(np.allclose(regenerated["endpoints"], stored["endpoints"], rtol=1e-6, atol=1e-6))
    event_equal = np.array_equal(regenerated["events"], stored["events"])
    break_equal = np.array_equal(regenerated["breaks"], stored["breaks"])
    state_key = "states_x" if tier == "continuous" else "states_protein"
    state_equal = np.array_equal(regenerated[state_key], stored[state_key])
    digest_equal = np.array_equal(regenerated["trajectory_digest"], stored["trajectory_digest"])
    secondary_equal = all(np.array_equal(regenerated[name], stored[name]) for name in (
        "run5_count", "f24_count", "event_count_q025", "event_count_q10",
    ))
    return {
        "network_index": int(network_index),
        "state_equal": bool(state_equal),
        "endpoint_equal": bool(endpoint_equal),
        "endpoint_close": endpoint_close,
        "event_equal": bool(event_equal),
        "break_equal": bool(break_equal),
        "trajectory_digest_equal": bool(digest_equal),
        "secondary_equal": bool(secondary_equal),
        "pass": bool(state_equal and endpoint_close and event_equal and break_equal and digest_equal and secondary_equal),
    }
