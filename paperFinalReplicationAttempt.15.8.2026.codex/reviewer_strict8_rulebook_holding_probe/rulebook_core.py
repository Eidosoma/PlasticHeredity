"""Pure mechanics for the post-hoc strict-8 rulebook/holding probe.

Nothing in this module reads outcome data.  The deterministic rulebook form is
the fixed point of the expected composition flow implied by the frozen GARD
join/leave equations.  Composition edits preserve total mass and the occupied
set so that alignment is not confounded with a richness intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

RULEBOOK_FEATURE_NAMES = (
    "nearest_book_cosine",
    "nearest_book_l1_distance",
    "state_log_self_support",
    "book_log_self_support",
    "local_flow_log_l1",
    "local_flow_toward_book_cosine",
    "one_step_book_cosine_gain",
    "rulebook_stability_margin",
    "book_effective_species",
    "book_top1_share",
)

EDIT_ARMS = (
    "NOOP",
    "TOWARD_BOOK_D1",
    "TOWARD_BOOK_D4",
    "AWAY_BOOK_D1",
    "AWAY_BOOK_D4",
    "RANDOM_MATCHED_D1",
    "RANDOM_MATCHED_D4",
)


def normalized(values: NDArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0.0):
        raise ValueError("composition must be one-dimensional and nonnegative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("composition must be nonempty")
    return np.ascontiguousarray(array / total)


def cosine(left: NDArray, right: NDArray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def mean_field_flow(
    composition: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> FloatArray:
    """Expected simplex flow from the simulator's join/leave equations."""

    x = normalized(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.shape != (x.size, x.size):
        raise ValueError("beta shape differs from composition dimension")
    rho = 1.0 / x.size
    boost = 1.0 + matrix @ x
    raw = (k_join * rho - k_leave * x) * boost
    flow = raw - x * float(raw.sum())
    # Numerical roundoff must not introduce a mass direction.
    flow -= float(flow.sum()) / flow.size
    return np.ascontiguousarray(flow)


def _stationary_update(
    values: FloatArray,
    beta: FloatArray,
    k_join: float,
    k_leave: float,
) -> FloatArray:
    """Picard update obtained by solving the stationary equations for x|beta x."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be a start-by-type matrix")
    n_types = x.shape[1]
    basal = k_join / n_types
    boost = 1.0 + x @ np.asarray(beta, dtype=np.float64).T
    low = np.zeros(x.shape[0], dtype=np.float64)
    high = 2.0 * basal * boost.sum(axis=1) + k_leave * boost.max(axis=1)
    # For fixed boost, sum_i basal*b_i/(lambda+k_leave*b_i) is monotone.
    for _ in range(60):
        middle = (low + high) / 2.0
        total = (basal * boost / (middle[:, None] + k_leave * boost)).sum(axis=1)
        low = np.where(total > 1.0, middle, low)
        high = np.where(total > 1.0, high, middle)
    updated = basal * boost / (high[:, None] + k_leave * boost)
    return updated / updated.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class RulebookSolution:
    forms: FloatArray
    iterations: int
    maximum_update: float
    maximum_flow_residual: float
    starts: int


def solve_rulebook(
    beta: NDArray,
    k_join: float,
    k_leave: float,
    starts: int,
    seed: int,
    maximum_iterations: int = 1_000,
    tolerance: float = 1e-11,
    damping: float = 0.5,
    merge_cosine: float = 0.95,
) -> RulebookSolution:
    """Solve and cluster beta-derived deterministic composition fixed points."""

    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("beta must be square")
    if starts < 2:
        raise ValueError("at least two starts are required")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    values = np.vstack(
        (
            np.full(matrix.shape[0], 1.0 / matrix.shape[0]),
            rng.dirichlet(np.ones(matrix.shape[0]), size=starts - 1),
        )
    )
    maximum_update = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        updated = _stationary_update(values, matrix, k_join, k_leave)
        maximum_update = float(np.max(np.abs(updated - values)))
        values = (1.0 - damping) * values + damping * updated
        values /= values.sum(axis=1, keepdims=True)
        if maximum_update < tolerance:
            break
    else:
        raise RuntimeError(
            f"rulebook solver did not converge in {maximum_iterations} iterations; "
            f"maximum update={maximum_update:.3g}"
        )

    leaders: list[FloatArray] = []
    for value in values:
        if not any(cosine(value, leader) >= merge_cosine for leader in leaders):
            leaders.append(value.copy())
    forms = np.vstack(leaders)
    residual = max(
        float(np.max(np.abs(mean_field_flow(form, matrix, k_join, k_leave))))
        for form in forms
    )
    return RulebookSolution(
        forms=forms,
        iterations=iteration,
        maximum_update=maximum_update,
        maximum_flow_residual=residual,
        starts=starts,
    )


def nearest_form(composition: NDArray, forms: NDArray) -> tuple[int, FloatArray, float]:
    x = normalized(composition)
    candidates = np.asarray(forms, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != x.size or len(candidates) == 0:
        raise ValueError("forms must be a nonempty form-by-type matrix")
    scores = np.asarray([cosine(x, form) for form in candidates])
    index = int(np.argmax(scores))
    return index, candidates[index].copy(), float(scores[index])


def flow_jacobian(
    composition: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> FloatArray:
    """Analytic Jacobian of the normalized expected composition flow."""

    x = normalized(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    rho = 1.0 / x.size
    basal = k_join * rho
    boost = 1.0 + matrix @ x
    raw = (basal - k_leave * x) * boost
    derivative = (basal - k_leave * x)[:, None] * matrix
    derivative[np.diag_indices_from(derivative)] -= k_leave * boost
    total_derivative = derivative.sum(axis=0)
    jacobian = derivative - np.outer(x, total_derivative)
    jacobian[np.diag_indices_from(jacobian)] -= float(raw.sum())
    return jacobian


def tangent_stability_margin(
    form: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> float:
    """Negative spectral abscissa on the simplex tangent space; positive is stable."""

    x = normalized(form)
    basis_seed = np.vstack((np.eye(x.size - 1), -np.ones(x.size - 1)))
    basis, _ = np.linalg.qr(basis_seed)
    reduced = basis.T @ flow_jacobian(x, beta, k_join, k_leave) @ basis
    maximum_real = float(np.max(np.linalg.eigvals(reduced).real))
    return -maximum_real


def effective_species(profile: NDArray) -> float:
    p = normalized(profile)
    positive = p[p > 0.0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def rulebook_features(
    composition: NDArray,
    beta: NDArray,
    forms: NDArray,
    k_join: float,
    k_leave: float,
    stability_margin: float | None = None,
) -> FloatArray:
    x = normalized(composition)
    _, book, similarity = nearest_form(x, forms)
    flow = mean_field_flow(x, beta, k_join, k_leave)
    direction = book - x
    denominator = float(np.linalg.norm(flow) * np.linalg.norm(direction))
    alignment = float(np.dot(flow, direction) / denominator) if denominator else 0.0
    scale = float(np.max(np.abs(flow)))
    if scale:
        stepped = np.clip(x + 1e-4 * flow / scale, 0.0, None)
        stepped /= stepped.sum()
        step_gain = cosine(stepped, book) - similarity
    else:
        step_gain = 0.0
    matrix = np.asarray(beta, dtype=np.float64)
    state_support = float(x @ matrix @ x)
    book_support = float(book @ matrix @ book)
    if stability_margin is None:
        stability_margin = tangent_stability_margin(book, matrix, k_join, k_leave)
    values = np.asarray(
        (
            similarity,
            float(np.abs(x - book).sum()),
            float(np.log1p(state_support)),
            float(np.log1p(book_support)),
            float(np.log1p(np.abs(flow).sum())),
            alignment,
            step_gain,
            float(stability_margin),
            effective_species(book),
            float(book.max()),
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("nonfinite rulebook feature")
    return values


@dataclass(frozen=True)
class EditResult:
    composition: IntArray
    requested_dose: int
    achieved_dose: int
    occupied_before: int
    occupied_after: int
    cosine_before: float
    cosine_after: float
    transfers: tuple[tuple[int, int], ...]


def _parse_arm(arm: str) -> tuple[str, int]:
    if arm == "NOOP":
        return "NOOP", 0
    if arm not in EDIT_ARMS:
        raise ValueError(f"unknown edit arm: {arm}")
    stem, dose = arm.rsplit("_D", 1)
    return stem, int(dose)


def apply_rulebook_edit(
    composition: NDArray,
    target_form: NDArray,
    arm: str,
    selection_seed: int,
) -> EditResult:
    """Move molecules toward/away/randomly within the original occupied set."""

    values = np.asarray(composition, dtype=np.int64).copy()
    if values.ndim != 1 or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("composition must be a nonempty nonnegative integer vector")
    target = normalized(target_form)
    if target.shape != values.shape:
        raise ValueError("target form shape differs")
    stem, dose = _parse_arm(arm)
    before = values.copy()
    before_occupied = int(np.count_nonzero(before))
    rng = np.random.default_rng(selection_seed)
    transfers: list[tuple[int, int]] = []
    for _ in range(dose):
        donors = np.flatnonzero(values > 1)
        recipients = np.flatnonzero(values > 0)
        pairs = [(int(i), int(j)) for i in donors for j in recipients if i != j]
        if not pairs:
            break
        if stem == "RANDOM_MATCHED":
            selected = pairs[int(rng.integers(0, len(pairs)))]
        else:
            scored: list[tuple[float, int, int]] = []
            for donor, recipient in pairs:
                candidate = values.copy()
                candidate[donor] -= 1
                candidate[recipient] += 1
                scored.append((cosine(candidate, target), donor, recipient))
            if stem == "TOWARD_BOOK":
                best = max(value[0] for value in scored)
                if best <= cosine(values, target) + 1e-15:
                    break
                selected = min(
                    (donor, recipient)
                    for score, donor, recipient in scored
                    if np.isclose(score, best, rtol=0.0, atol=1e-15)
                )
            elif stem == "AWAY_BOOK":
                best = min(value[0] for value in scored)
                if best >= cosine(values, target) - 1e-15:
                    break
                selected = min(
                    (donor, recipient)
                    for score, donor, recipient in scored
                    if np.isclose(score, best, rtol=0.0, atol=1e-15)
                )
            else:
                raise ValueError(stem)
        donor, recipient = selected
        values[donor] -= 1
        values[recipient] += 1
        transfers.append(selected)

    result = EditResult(
        composition=values,
        requested_dose=dose,
        achieved_dose=len(transfers),
        occupied_before=before_occupied,
        occupied_after=int(np.count_nonzero(values)),
        cosine_before=cosine(before, target),
        cosine_after=cosine(values, target),
        transfers=tuple(transfers),
    )
    if int(values.sum()) != int(before.sum()) or np.any(values < 0):
        raise AssertionError("rulebook edit violated mass/nonnegativity")
    if result.occupied_after != result.occupied_before:
        raise AssertionError("rulebook edit changed occupied-type count")
    if stem == "TOWARD_BOOK" and result.achieved_dose and result.cosine_after <= result.cosine_before:
        raise AssertionError("toward-book edit did not improve target cosine")
    if stem == "AWAY_BOOK" and result.achieved_dose and result.cosine_after >= result.cosine_before:
        raise AssertionError("away-book edit did not reduce target cosine")
    return result


def aggregate_transitions(gates: NDArray) -> tuple[IntArray, IntArray]:
    """Aggregate break/run/coherence/anchor successes and eligible trials."""

    values = np.asarray(gates, dtype=np.int8)
    if values.ndim != 2:
        raise ValueError("gates must be state-by-branch")
    successes = np.empty((values.shape[0], 4), dtype=np.int64)
    trials = np.empty_like(successes)
    successes[:, 0] = np.count_nonzero(values >= 1, axis=1)
    trials[:, 0] = values.shape[1]
    for transition in range(1, 4):
        successes[:, transition] = np.count_nonzero(values >= transition + 1, axis=1)
        trials[:, transition] = np.count_nonzero(values >= transition, axis=1)
    return successes, trials


def smoothed_rate(successes: NDArray, trials: NDArray) -> FloatArray:
    success = np.asarray(successes, dtype=np.float64)
    total = np.asarray(trials, dtype=np.float64)
    return (success + 0.5) / (total + 1.0)
