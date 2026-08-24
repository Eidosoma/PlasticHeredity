"""Run-level reconstruction of the paper's established-metric comparisons."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.spatial.distance import cdist, pdist

from .analysis import AnalyzedRun
from .composition import relative_composition


FloatArray = NDArray[np.float64]


def _standardize(values: ArrayLike) -> FloatArray:
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return series
    scale = float(np.std(series, ddof=1)) if series.size > 1 else 0.0
    return (series - np.mean(series)) / scale if scale > 0 else np.zeros_like(series)


def sample_entropy(values: ArrayLike, *, dimension: int = 2) -> float:
    """Chebyshev sample entropy with the conventional 0.2-SD tolerance."""

    series = _standardize(values)
    if series.size < dimension + 4 or np.std(series) == 0:
        return float("nan")

    def match_probability(width: int) -> float:
        embedded = np.lib.stride_tricks.sliding_window_view(series, width)
        distances = pdist(embedded, metric="chebyshev")
        return float(np.mean(distances <= 0.2))

    lower = match_probability(dimension)
    upper = match_probability(dimension + 1)
    return float(-np.log(upper / lower)) if lower > 0 and upper > 0 else float("nan")


def _delay_embed(values: FloatArray, dimension: int = 3) -> FloatArray:
    if values.size < dimension:
        return np.empty((0, dimension), dtype=float)
    return np.lib.stride_tricks.sliding_window_view(values, dimension)


def correlation_dimension(values: ArrayLike) -> float:
    """Grassberger–Procaccia slope on a three-dimensional delay embedding."""

    embedded = _delay_embed(_standardize(values), 3)
    if embedded.shape[0] < 20:
        return float("nan")
    if embedded.shape[0] > 512:
        indices = np.linspace(0, embedded.shape[0] - 1, 512).astype(int)
        embedded = embedded[indices]
    distances = pdist(embedded)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size < 20:
        return float("nan")
    radii = np.unique(np.quantile(distances, np.linspace(0.1, 0.6, 10)))
    correlation_sum = np.asarray([np.mean(distances <= radius) for radius in radii])
    valid = (radii > 0) & (correlation_sum > 0) & (correlation_sum < 1)
    if valid.sum() < 3:
        return float("nan")
    return float(stats.linregress(np.log(radii[valid]), np.log(correlation_sum[valid])).slope)


def largest_lyapunov(values: ArrayLike) -> float:
    """Short-horizon Rosenstein estimate of the largest Lyapunov exponent."""

    embedded = _delay_embed(_standardize(values), 3)
    count = embedded.shape[0]
    if count < 30:
        return float("nan")
    if count > 512:
        indices = np.linspace(0, count - 1, 512).astype(int)
        embedded = embedded[indices]
        count = embedded.shape[0]
    distances = cdist(embedded, embedded)
    theiler = max(3, count // 20)
    row, column = np.indices(distances.shape)
    distances[np.abs(row - column) <= theiler] = np.inf
    neighbor = np.argmin(distances, axis=1)
    initial = distances[np.arange(count), neighbor]
    valid_origin = np.isfinite(initial) & (initial > 0)
    horizon = min(10, count // 10)
    divergence = []
    times = []
    for offset in range(horizon):
        origins = np.flatnonzero(
            valid_origin
            & (np.arange(count) + offset < count)
            & (neighbor + offset < count)
        )
        if origins.size < 5:
            continue
        separation = np.linalg.norm(
            embedded[origins + offset] - embedded[neighbor[origins] + offset], axis=1
        )
        separation = separation[separation > 0]
        if separation.size:
            divergence.append(float(np.mean(np.log(separation))))
            times.append(offset)
    if len(times) < 3:
        return float("nan")
    return float(stats.linregress(times, divergence).slope)


def detrended_fluctuation(values: ArrayLike) -> float:
    """First-order detrended fluctuation-analysis scaling exponent."""

    series = _standardize(values)
    if series.size < 32 or np.std(series) == 0:
        return float("nan")
    integrated = np.cumsum(series - np.mean(series))
    scales = np.unique(
        np.geomspace(4, max(5, series.size // 4), num=10).astype(int)
    )
    fluctuations = []
    kept_scales = []
    for scale in scales:
        segments = series.size // scale
        if segments < 2:
            continue
        blocks = integrated[: segments * scale].reshape(segments, scale)
        x = np.arange(scale)
        residuals = []
        for block in blocks:
            trend = np.polyval(np.polyfit(x, block, 1), x)
            residuals.append(np.mean((block - trend) ** 2))
        fluctuation = float(np.sqrt(np.mean(residuals)))
        if fluctuation > 0:
            kept_scales.append(scale)
            fluctuations.append(fluctuation)
    if len(kept_scales) < 3:
        return float("nan")
    return float(
        stats.linregress(np.log(kept_scales), np.log(fluctuations)).slope
    )


def generalized_hurst(values: ArrayLike, *, order: float = 2.0) -> float:
    """Generalized Hurst exponent from order-q structure functions."""

    series = _standardize(values)
    if series.size < 32 or np.std(series) == 0:
        return float("nan")
    lags = np.unique(np.geomspace(2, max(3, series.size // 4), num=12).astype(int))
    moments = []
    kept_lags = []
    for lag in lags:
        differences = np.abs(series[lag:] - series[:-lag])
        moment = float(np.mean(differences**order) ** (1.0 / order))
        if moment > 0:
            kept_lags.append(lag)
            moments.append(moment)
    if len(kept_lags) < 3:
        return float("nan")
    return float(stats.linregress(np.log(kept_lags), np.log(moments)).slope)


def _network_metrics(run: AnalyzedRun, *, catalytic_threshold: float = 1.0) -> Dict[str, float]:
    active = np.flatnonzero(np.any(run.trace.counts > 0, axis=0))
    beta = run.trace.beta[np.ix_(active, active)]
    adjacency = beta > catalytic_threshold
    np.fill_diagonal(adjacency, False)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(active.size))
    for affected, catalyst in np.argwhere(adjacency):
        strength = float(beta[affected, catalyst])
        graph.add_edge(
            int(catalyst),
            int(affected),
            strength=strength,
            distance=1.0 / strength,
        )
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    if nodes == 0:
        return {name: float("nan") for name in (
            "network_nodes", "network_edges", "network_indegree", "network_outdegree",
            "network_betweenness", "network_pagerank", "network_hits_hub", "network_hits_authority"
        )}
    indegree = np.asarray([degree for _, degree in graph.in_degree()], dtype=float)
    outdegree = np.asarray([degree for _, degree in graph.out_degree()], dtype=float)
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    pagerank = nx.pagerank(graph, weight="strength")
    try:
        hubs, authorities = nx.hits(graph, max_iter=1000, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs = {node: float("nan") for node in graph}
        authorities = hubs
    return {
        "network_nodes": float(nodes),
        "network_edges": float(edges),
        "network_indegree": float(np.mean(indegree)),
        "network_outdegree": float(np.mean(outdegree)),
        "network_betweenness": float(np.mean(list(betweenness.values()))),
        # Maxima quantify centralization; means of these normalized scores are
        # nearly fixed by node count and would be uninformative.
        "network_pagerank": float(np.max(list(pagerank.values()))),
        "network_hits_hub": float(np.nanmax(list(hubs.values()))),
        "network_hits_authority": float(np.nanmax(list(authorities.values()))),
    }


def established_metric_values(runs: Sequence[AnalyzedRun]) -> pd.DataFrame:
    """Compute one transparent scalar per run for each named comparator."""

    rows: List[Dict[str, float | int]] = []
    for run in runs:
        relative = relative_composition(run.trace.counts)
        dynamics = np.linalg.norm(np.diff(relative, axis=0), axis=1)
        row: Dict[str, float | int] = {
            "run_index": run.run_index,
            "phi_mean": float(np.mean(run.causal.values)),
            "phi_std": float(np.std(run.causal.values, ddof=1)),
            "dynamic_sample_entropy": sample_entropy(dynamics),
            "dynamic_correlation_dimension": correlation_dimension(dynamics),
            "dynamic_lyapunov": largest_lyapunov(dynamics),
            "dynamic_dfa": detrended_fluctuation(dynamics),
            "dynamic_generalized_hurst": generalized_hurst(dynamics),
        }
        row.update(_network_metrics(run))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_correlations(x: FloatArray, y: FloatArray) -> Tuple[float, float, float, float, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    first = x[finite]
    second = y[finite]
    if first.size < 3 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(first.size)
    pearson = stats.pearsonr(first, second)
    spearman = stats.spearmanr(first, second)
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
        int(first.size),
    )


def established_metric_correlations(values: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman run-level correlations with mean Phi-r."""

    rows = []
    response = values.phi_mean.to_numpy(dtype=float)
    for metric in values.columns:
        if metric in {"run_index", "phi_mean", "phi_std"}:
            continue
        pearson_r, pearson_p, spearman_rho, spearman_p, count = _safe_correlations(
            response, values[metric].to_numpy(dtype=float)
        )
        rows.append(
            {
                "metric": metric,
                "family": "network" if metric.startswith("network_") else "dynamic",
                "n": count,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )
    result = pd.DataFrame(rows)
    for p_column, q_column in (("pearson_p", "pearson_q_bh"), ("spearman_p", "spearman_q_bh")):
        pvalues = result[p_column].to_numpy(dtype=float)
        finite = np.isfinite(pvalues)
        qvalues = np.full(pvalues.shape, np.nan)
        if finite.any():
            order = np.argsort(pvalues[finite])
            ranked = pvalues[finite][order]
            adjusted = np.minimum.accumulate(
                (ranked * ranked.size / np.arange(1, ranked.size + 1))[::-1]
            )[::-1]
            restored = np.empty_like(adjusted)
            restored[order] = np.clip(adjusted, 0, 1)
            qvalues[finite] = restored
        result[q_column] = qvalues
    return result
