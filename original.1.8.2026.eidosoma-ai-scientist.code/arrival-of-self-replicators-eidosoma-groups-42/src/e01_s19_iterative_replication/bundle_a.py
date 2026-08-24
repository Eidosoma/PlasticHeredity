"""Locked Bundle A feature reconstruction for S19-L01."""

from __future__ import annotations

import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_iterative_replication.core import derive_seed32

DYNAMIC_SPECIFICATIONS = (
    "A_DYNAMICS_DIRECT_SELECTED_CLOCK",
    "A_DYNAMICS_SOURCE_WINDOW100",
)
NETWORK_SPECIFICATIONS = (
    "A_GRAPH_UNWEIGHTED_POSITIVE_SUPPORT",
    "A_GRAPH_WEIGHTED_BETA",
)


def relative_compositions(states: np.ndarray) -> np.ndarray:
    array = np.asarray(states, dtype=np.float64)
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("empty state in frozen trajectory")
    return array / totals


def source_preprocess(compositions: np.ndarray, specification: str) -> np.ndarray:
    """Apply one of the two locked source-grounded time preprocessors."""

    values = np.asarray(compositions, dtype=np.float64)
    if specification == "A_DYNAMICS_DIRECT_SELECTED_CLOCK":
        sampled = values
    elif specification == "A_DYNAMICS_SOURCE_WINDOW100":
        sampled = np.column_stack(
            [
                np.nan_to_num(uniform_filter1d(values[:, column], size=100)[::100], copy=False)
                for column in range(values.shape[1])
            ]
        )
    else:
        raise ValueError(f"unknown dynamics specification: {specification}")
    return StandardScaler().fit_transform(sampled).astype(np.float64, copy=False)


def _safe_scalar(function: Any) -> tuple[float | None, str | None, list[str]]:
    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function()
        captured = [str(item.message) for item in caught]
        array = np.asarray(value, dtype=np.float64)
        if array.size != 1 or not np.isfinite(array.reshape(-1)[0]):
            return None, "undefined_or_nonfinite", captured
        return float(array.reshape(-1)[0]), None, captured
    except Exception as error:  # source behavior is retained as a status-bearing failure
        return None, f"{type(error).__name__}:{error}", captured


def dynamical_metrics(data: np.ndarray, *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the pinned same-author/nolds metric calls with retained failures."""

    import nolds  # loaded from the pinned cache path by the scientific runner

    np.random.seed(int(seed))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    warning_text: list[str] = []

    value, reason, caught = _safe_scalar(lambda: nolds.sampen(data))
    warning_text.extend(caught)
    rows.append({"metricId": "C008_SAMPLE_ENTROPY", "summaryId": "PRIMARY", "value": value, "failureReason": reason})

    correlation_dimensions: list[float] = []
    correlation_failures = 0
    lyapunov_values: list[float] = []
    lyapunov_failures = 0
    for series in data.T:
        cd, _, caught = _safe_scalar(lambda series=series: nolds.corr_dim(series, emb_dim=2))
        warning_text.extend(caught)
        # The public source substitutes 0 for correlation-dimension exceptions.
        if cd is None:
            correlation_failures += 1
            correlation_dimensions.append(0.0)
        else:
            correlation_dimensions.append(cd)
        le, _, caught = _safe_scalar(
            lambda series=series: np.max(nolds.lyap_e(series.astype(np.float32)))
        )
        warning_text.extend(caught)
        if le is None:
            lyapunov_failures += 1
        else:
            lyapunov_values.append(le)
    rows.extend(
        [
            {
                "metricId": "C009_CORRELATION_DIMENSION",
                "summaryId": "PRIMARY_MEAN",
                "value": float(np.mean(correlation_dimensions)),
                "failureReason": None,
            },
            {
                "metricId": "C009_CORRELATION_DIMENSION",
                "summaryId": "SOURCE_STD_COMPANION",
                "value": float(np.std(correlation_dimensions)),
                "failureReason": None,
            },
            {
                "metricId": "C010_LYAPUNOV_EXPONENT",
                "summaryId": "PRIMARY_MEAN",
                "value": float(np.mean(lyapunov_values)) if lyapunov_values else None,
                "failureReason": None if lyapunov_values else "all_component_lyapunov_calls_failed",
            },
            {
                "metricId": "C010_LYAPUNOV_EXPONENT",
                "summaryId": "SOURCE_STD_COMPANION",
                "value": float(np.std(lyapunov_values)) if lyapunov_values else None,
                "failureReason": None if lyapunov_values else "all_component_lyapunov_calls_failed",
            },
            {
                "metricId": "C010_LYAPUNOV_EXPONENT",
                "summaryId": "SOURCE_MAX_COMPANION",
                "value": float(np.max(lyapunov_values)) if lyapunov_values else None,
                "failureReason": None if lyapunov_values else "all_component_lyapunov_calls_failed",
            },
        ]
    )
    dfa, reason, caught = _safe_scalar(lambda: nolds.dfa(data))
    warning_text.extend(caught)
    rows.append({"metricId": "C011_DFA", "summaryId": "PRIMARY", "value": dfa, "failureReason": reason})
    ghe, reason, caught = _safe_scalar(lambda: nolds.mfhurst_b(data)[0])
    warning_text.extend(caught)
    rows.append({"metricId": "C012_GENERALIZED_HURST", "summaryId": "PRIMARY", "value": ghe, "failureReason": reason})
    diagnostics = {
        "inputTimeCount": int(data.shape[0]),
        "inputVariableCount": int(data.shape[1]),
        "correlationDimensionSubstitutionCount": correlation_failures,
        "lyapunovFailureCount": lyapunov_failures,
        "warningCount": len(warning_text),
        "warnings": " | ".join(sorted(set(warning_text)))[:4000] or None,
        "runtimeSeconds": time.perf_counter() - started,
        "strictSourceEligible": bool(lyapunov_failures == 0 and all(row["value"] is not None for row in rows)),
    }
    return rows, diagnostics


def dynamic_task(task: tuple[str, int, str, str, bool]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Multiprocessing-safe trajectory feature task."""

    candidate_id, matrix_index, cache_path, specification, replay = task
    started = time.perf_counter()
    with Path(cache_path).open("rb") as handle:
        trajectory = pickle.load(handle)
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    compositions = relative_compositions(states)
    data = source_preprocess(compositions, specification)
    rows, diagnostics = dynamical_metrics(
        data,
        seed=derive_seed32("bundle_a", candidate_id, matrix_index, specification, "metric_rng"),
    )
    for row in rows:
        row.update(
            {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "specificationId": specification,
                "replay": replay,
                "trajectoryLength": len(states),
            }
        )
    diagnostics.update(
        {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "specificationId": specification,
            "replay": replay,
            "taskRuntimeSeconds": time.perf_counter() - started,
        }
    )
    return rows, diagnostics


def _summary_rows(values: dict[Any, float], prefix: str) -> list[dict[str, Any]]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return [
        {"metricId": prefix, "summaryId": "PRIMARY_MEAN", "value": float(np.mean(array))},
        {"metricId": prefix, "summaryId": "SOURCE_STD_COMPANION", "value": float(np.std(array))},
    ]


def network_metrics(beta: np.ndarray, specification: str) -> list[dict[str, Any]]:
    """Apply the public NetworkX summary function to the locked GARD graph."""

    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (100, 100) or np.any(matrix <= 0):
        raise ValueError("frozen beta is not a positive 100x100 matrix")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(100))
    if specification == "A_GRAPH_UNWEIGHTED_POSITIVE_SUPPORT":
        graph.add_edges_from((i, j) for i in range(100) for j in range(100))
    elif specification == "A_GRAPH_WEIGHTED_BETA":
        graph.add_weighted_edges_from(
            (i, j, float(matrix[i, j])) for i in range(100) for j in range(100)
        )
    else:
        raise ValueError(f"unknown graph specification: {specification}")
    indegree = dict(graph.in_degree())
    outdegree = dict(graph.out_degree())
    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)
    hubs, authorities = nx.hits(graph)
    rows: list[dict[str, Any]] = [
        {"metricId": "C001_NUMBER_OF_NODES", "summaryId": "PRIMARY", "value": float(graph.number_of_nodes())},
        {"metricId": "C002_NUMBER_OF_EDGES", "summaryId": "PRIMARY", "value": float(graph.number_of_edges())},
    ]
    rows.extend(_summary_rows(indegree, "C003_IN_DEGREE"))
    rows.extend(_summary_rows(outdegree, "C004_OUT_DEGREE"))
    rows.extend(_summary_rows(betweenness, "C005_BETWEENNESS"))
    rows.extend(_summary_rows(pagerank, "C006_PAGERANK"))
    rows.extend(_summary_rows(hubs, "C007_HITS_HUB"))
    rows.extend(
        [
            {"metricId": "C007_HITS_AUTHORITY_COMPANION", "summaryId": "PRIMARY_MEAN", "value": float(np.mean(list(authorities.values())))},
            {"metricId": "C007_HITS_AUTHORITY_COMPANION", "summaryId": "SOURCE_STD_COMPANION", "value": float(np.std(list(authorities.values())))},
        ]
    )
    for row in rows:
        row["failureReason"] = None if math.isfinite(row["value"]) else "nonfinite"
    return rows


__all__ = [
    "DYNAMIC_SPECIFICATIONS",
    "NETWORK_SPECIFICATIONS",
    "dynamic_task",
    "dynamical_metrics",
    "network_metrics",
    "relative_compositions",
    "source_preprocess",
]
