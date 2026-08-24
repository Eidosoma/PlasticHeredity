from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


STATUS_LIMIT = 0
STATUS_POINT = 1
STATUS_CYCLE = 2


def signed_update(field: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return np.where(field > 0.0, 1, np.where(field < 0.0, -1, previous)).astype(np.int8)


def sequential_sweep_numpy(
    weights: np.ndarray,
    state: np.ndarray,
    external_field: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(state, dtype=np.int8).copy()
    field = np.zeros_like(result, dtype=np.float64) if external_field is None else np.asarray(external_field, dtype=np.float64)
    matrix = np.asarray(weights, dtype=np.float64)
    for gene in range(result.shape[-1]):
        value = result @ matrix[gene] + field[..., gene]
        result[..., gene] = signed_update(value, result[..., gene])
    return result


def states_from_int(values: np.ndarray | list[int], genes: int) -> np.ndarray:
    integers = np.asarray(values, dtype=np.uint16).reshape(-1)
    bits = ((integers[:, None] >> np.arange(genes, dtype=np.uint16)) & 1).astype(np.int8)
    return np.where(bits == 1, 1, -1).astype(np.int8)


def states_to_int(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.int8)
    powers = (1 << np.arange(values.shape[-1], dtype=np.uint16))[None, :]
    return np.sum((values > 0).astype(np.uint16) * powers, axis=-1).astype(np.uint16)


def _jax_modules() -> tuple[Any, Any]:
    import jax

    # The registered Wagner genotype is float64. Enabling x64 makes the matrix
    # used on CPU/CUDA identical to the enumerated deterministic landscape.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


def _sequential_sweep_jax(jax, jnp, weights, state, field, hard_mask, hard_values):
    def one_gene(gene, current):
        value = jnp.sum(current * weights[:, gene, :], axis=1) + field[:, gene]
        updated = jnp.where(value > 0.0, 1, jnp.where(value < 0.0, -1, current[:, gene]))
        updated = jnp.where(hard_mask[gene], hard_values[:, gene], updated)
        return current.at[:, gene].set(updated.astype(jnp.int8))

    return jax.lax.fori_loop(0, state.shape[1], one_gene, state)


def _state_codes_jax(jnp, state):
    powers = (1 << jnp.arange(state.shape[1], dtype=jnp.uint16))[None, :]
    return jnp.sum((state > 0).astype(jnp.uint16) * powers, axis=1).astype(jnp.int32)


def _develop_core(jax, jnp, weights, state, field, hard_mask, hard_values, max_sweeps):
    batch = state.shape[0]
    state_count = 1 << state.shape[1]
    state = jnp.where(hard_mask[None, :], hard_values, state).astype(jnp.int8)
    rows = jnp.arange(batch, dtype=jnp.int32)
    visited = jnp.zeros((batch, state_count), dtype=jnp.bool_)
    visited = visited.at[rows, _state_codes_jax(jnp, state)].set(True)
    active = jnp.ones(batch, dtype=jnp.bool_)
    status = jnp.zeros(batch, dtype=jnp.int8)
    steps = jnp.zeros(batch, dtype=jnp.int16)

    def condition(carry):
        _, _, current_active, _, _, iteration = carry
        return jnp.logical_and(iteration < max_sweeps, jnp.any(current_active))

    def body(carry):
        current, current_visited, current_active, current_status, current_steps, iteration = carry
        candidate = _sequential_sweep_jax(
            jax, jnp, weights, current, field, hard_mask, hard_values
        )
        candidate = jnp.where(current_active[:, None], candidate, current)
        codes = _state_codes_jax(jnp, candidate)
        point = jnp.all(candidate == current, axis=1)
        repeated = current_visited[rows, codes]
        newly_point = current_active & point
        newly_cycle = current_active & (~point) & repeated
        resolved = newly_point | newly_cycle
        next_status = jnp.where(
            newly_point,
            STATUS_POINT,
            jnp.where(newly_cycle, STATUS_CYCLE, current_status),
        ).astype(jnp.int8)
        next_visited = current_visited.at[rows, codes].set(
            current_visited[rows, codes] | current_active
        )
        return (
            candidate,
            next_visited,
            current_active & (~resolved),
            next_status,
            current_steps + current_active.astype(jnp.int16),
            iteration + 1,
        )

    result = jax.lax.while_loop(
        condition,
        body,
        (state, visited, active, status, steps, jnp.asarray(0, dtype=jnp.int16)),
    )
    return result[0], result[3], result[4]


@lru_cache(maxsize=None)
def _compiled_development(max_sweeps: int):
    jax, jnp = _jax_modules()

    @jax.jit
    def run(base_weights, initial, field, gamma_variance, flip_probability, key_data, hard_mask, hard_values):
        root = jax.random.wrap_key_data(key_data)
        flip_key, gamma_key = jax.random.split(root)
        flips = jax.random.uniform(flip_key, initial.shape, dtype=jnp.float64) < flip_probability
        state = jnp.where(flips, -initial, initial).astype(jnp.int8)
        safe_variance = jnp.maximum(gamma_variance, jnp.asarray(1e-12, dtype=jnp.float64))

        def noisy(_):
            multipliers = jax.random.gamma(
                gamma_key,
                1.0 / safe_variance,
                shape=(initial.shape[0], base_weights.shape[0], base_weights.shape[1]),
                dtype=jnp.float64,
            ) * safe_variance
            return base_weights[None, :, :] * multipliers

        def deterministic(_):
            return jnp.broadcast_to(base_weights[None, :, :], (initial.shape[0],) + base_weights.shape)

        weights = jax.lax.cond(gamma_variance > 0.0, noisy, deterministic, operand=None)
        return _develop_core(jax, jnp, weights, state, field, hard_mask, hard_values, max_sweeps)

    return run


def develop_one_cycle_jax(
    weights: np.ndarray,
    initial: np.ndarray,
    *,
    external_field: np.ndarray | None,
    gamma_variance: float,
    expression_flip_probability: float,
    key_data: np.ndarray,
    max_sweeps: int,
    hard_mask: np.ndarray | None = None,
    hard_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, jnp = _jax_modules()
    state = np.asarray(initial, dtype=np.int8)
    batch, genes = state.shape
    field = np.zeros((batch, genes), dtype=np.float64) if external_field is None else np.asarray(external_field, dtype=np.float64)
    mask = np.zeros(genes, dtype=bool) if hard_mask is None else np.asarray(hard_mask, dtype=bool)
    values = state if hard_values is None else np.asarray(hard_values, dtype=np.int8)
    adult, status, steps = _compiled_development(int(max_sweeps))(
        jnp.asarray(weights, dtype=jnp.float64),
        jnp.asarray(state, dtype=jnp.int8),
        jnp.asarray(field, dtype=jnp.float64),
        jnp.asarray(gamma_variance, dtype=jnp.float64),
        jnp.asarray(expression_flip_probability, dtype=jnp.float64),
        jnp.asarray(key_data, dtype=jnp.uint32),
        jnp.asarray(mask, dtype=jnp.bool_),
        jnp.asarray(values, dtype=jnp.int8),
    )
    return np.asarray(adult, dtype=np.int8), np.asarray(status, dtype=np.int8), np.asarray(steps, dtype=np.int16)


@lru_cache(maxsize=None)
def _compiled_deterministic_cycles(cycles: int, read_mode: str):
    jax, jnp = _jax_modules()
    if read_mode not in {"none", "first", "recurrent"}:
        raise ValueError(read_mode)

    @jax.jit
    def run(adult_table, initial, initial_mark, flip_probability, coupling, rho, read_enabled, write_enabled, key_data):
        root = jax.random.wrap_key_data(key_data)

        def one_cycle(carry, cycle_index):
            state, mark = carry
            flip_key, read_key = jax.random.split(jax.random.fold_in(root, cycle_index))
            flips = jax.random.uniform(flip_key, state.shape, dtype=jnp.float64) < flip_probability
            prepared = jnp.where(flips, -state, state).astype(jnp.int8)
            if read_mode == "none":
                active_read = jnp.asarray(False)
            elif read_mode == "first":
                active_read = cycle_index == 0
            else:
                active_read = jnp.asarray(True)
            probabilities = jnp.clip(coupling * jnp.abs(mark), 0.0, 1.0)
            draws = jax.random.uniform(read_key, state.shape, dtype=jnp.float64)
            mark_values = jnp.where(mark > 0.0, 1, jnp.where(mark < 0.0, -1, prepared)).astype(jnp.int8)
            use_mark = active_read & read_enabled & (draws < probabilities) & (mark != 0.0)
            prepared = jnp.where(use_mark, mark_values, prepared).astype(jnp.int8)
            adult = adult_table[_state_codes_jax(jnp, prepared)]
            if read_mode == "recurrent":
                updated_mark = jnp.where(
                    write_enabled,
                    rho * mark + (1.0 - rho) * adult.astype(jnp.float64),
                    mark,
                )
            else:
                updated_mark = mark
            return (adult, updated_mark), adult

        (adult, mark), adults = jax.lax.scan(
            one_cycle,
            (initial, initial_mark),
            jnp.arange(cycles, dtype=jnp.int32),
        )
        return adults, adult, mark

    return run


def rollout_adult_cycles_jax(
    adult_table: np.ndarray,
    initial: np.ndarray,
    *,
    cycles: int,
    expression_flip_probability: float,
    key_data: np.ndarray,
    read_mode: str = "none",
    mark: np.ndarray | None = None,
    coupling: float = 0.0,
    half_life: int = 0,
    read_enabled: bool = True,
    write_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, jnp = _jax_modules()
    state = np.asarray(initial, dtype=np.int8)
    initial_mark = np.zeros_like(state, dtype=np.float64) if mark is None else np.asarray(mark, dtype=np.float64)
    rho = 0.0 if half_life <= 0 else 2.0 ** (-1.0 / float(half_life))
    adults, adult, final_mark = _compiled_deterministic_cycles(int(cycles), read_mode)(
        jnp.asarray(adult_table, dtype=jnp.int8),
        jnp.asarray(state, dtype=jnp.int8),
        jnp.asarray(initial_mark, dtype=jnp.float64),
        jnp.asarray(expression_flip_probability, dtype=jnp.float64),
        jnp.asarray(coupling, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(read_enabled, dtype=jnp.bool_),
        jnp.asarray(write_enabled, dtype=jnp.bool_),
        jnp.asarray(key_data, dtype=jnp.uint32),
    )
    return np.asarray(adults, dtype=np.int8), np.asarray(adult, dtype=np.int8), np.asarray(final_mark, dtype=np.float64)


@lru_cache(maxsize=None)
def _compiled_latch_cycles(cycles: int):
    jax, jnp = _jax_modules()

    @jax.jit
    def run(
        adult_table, initial, carrier, ttl, pending, streak,
        flip_probability, coupling, retention, threshold,
        read_enabled, rewrite, key_data,
    ):
        root = jax.random.wrap_key_data(key_data)

        def one_cycle(carry, cycle_index):
            state, current_carrier, current_ttl, current_pending, current_streak = carry
            flip_key, read_key = jax.random.split(jax.random.fold_in(root, cycle_index))
            flips = jax.random.uniform(flip_key, state.shape, dtype=jnp.float64) < flip_probability
            prepared = jnp.where(flips, -state, state).astype(jnp.int8)
            draws = jax.random.uniform(read_key, state.shape, dtype=jnp.float64)
            active = (
                read_enabled
                & (current_carrier != 0)
                & (current_ttl > 0)
                & (draws < coupling)
            )
            prepared = jnp.where(active, current_carrier, prepared).astype(jnp.int8)
            adult = adult_table[_state_codes_jax(jnp, prepared)]

            next_ttl = jnp.maximum(current_ttl - 1, 0).astype(jnp.int16)

            def update_latch(_):
                same_pending = current_pending == adult
                next_pending = adult
                next_streak = jnp.where(same_pending, current_streak + 1, 1).astype(jnp.int16)
                matching = (current_carrier != 0) & (adult == current_carrier)
                renewed_ttl = jnp.where(matching, retention, next_ttl).astype(jnp.int16)
                writable = (
                    ((current_carrier == 0) | (renewed_ttl <= 0))
                    & (next_streak >= threshold)
                )
                next_carrier = jnp.where(writable, adult, current_carrier).astype(jnp.int8)
                renewed_ttl = jnp.where(writable, retention, renewed_ttl).astype(jnp.int16)
                next_carrier = jnp.where(renewed_ttl > 0, next_carrier, 0).astype(jnp.int8)
                return next_carrier, renewed_ttl, next_pending, next_streak

            def decay_only(_):
                next_carrier = jnp.where(next_ttl > 0, current_carrier, 0).astype(jnp.int8)
                return next_carrier, next_ttl, current_pending, current_streak

            next_carrier, next_ttl, next_pending, next_streak = jax.lax.cond(
                rewrite, update_latch, decay_only, operand=None
            )
            return (
                adult, next_carrier, next_ttl, next_pending, next_streak
            ), adult

        final, adults = jax.lax.scan(
            one_cycle,
            (initial, carrier, ttl, pending, streak),
            jnp.arange(cycles, dtype=jnp.int32),
        )
        return adults, *final

    return run


def rollout_latch_cycles_jax(
    adult_table: np.ndarray,
    initial: np.ndarray,
    carrier: np.ndarray,
    ttl: np.ndarray,
    pending: np.ndarray,
    streak: np.ndarray,
    *,
    cycles: int,
    expression_flip_probability: float,
    coupling: float,
    retention: int,
    threshold: int,
    read_enabled: bool,
    rewrite: bool,
    key_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a latch while its per-coordinate retention clock remains active."""
    _, jnp = _jax_modules()
    result = _compiled_latch_cycles(int(cycles))(
        jnp.asarray(adult_table, dtype=jnp.int8),
        jnp.asarray(initial, dtype=jnp.int8),
        jnp.asarray(carrier, dtype=jnp.int8),
        jnp.asarray(ttl, dtype=jnp.int16),
        jnp.asarray(pending, dtype=jnp.int8),
        jnp.asarray(streak, dtype=jnp.int16),
        jnp.asarray(expression_flip_probability, dtype=jnp.float64),
        jnp.asarray(coupling, dtype=jnp.float64),
        jnp.asarray(retention, dtype=jnp.int16),
        jnp.asarray(threshold, dtype=jnp.int16),
        jnp.asarray(read_enabled, dtype=jnp.bool_),
        jnp.asarray(rewrite, dtype=jnp.bool_),
        jnp.asarray(key_data, dtype=jnp.uint32),
    )
    adults, adult, final_carrier, final_ttl, final_pending, final_streak = result
    return (
        np.asarray(adults, dtype=np.int8),
        np.asarray(adult, dtype=np.int8),
        np.asarray(final_carrier, dtype=np.int8),
        np.asarray(final_ttl, dtype=np.int16),
        np.asarray(final_pending, dtype=np.int8),
        np.asarray(final_streak, dtype=np.int16),
    )


@lru_cache(maxsize=None)
def _compiled_noisy_cycles(cycles: int, max_sweeps: int):
    jax, jnp = _jax_modules()

    @jax.jit
    def run(base_weights, initial, gamma_variance, flip_probability, key_data):
        root = jax.random.wrap_key_data(key_data)
        zero_field = jnp.zeros(initial.shape, dtype=jnp.float64)
        hard_mask = jnp.zeros(initial.shape[1], dtype=jnp.bool_)

        def one_cycle(state, cycle_index):
            flip_key, gamma_key = jax.random.split(jax.random.fold_in(root, cycle_index))
            flips = jax.random.uniform(flip_key, state.shape, dtype=jnp.float64) < flip_probability
            prepared = jnp.where(flips, -state, state).astype(jnp.int8)
            safe_variance = jnp.maximum(gamma_variance, jnp.asarray(1e-12, dtype=jnp.float64))
            multipliers = jax.random.gamma(
                gamma_key,
                1.0 / safe_variance,
                shape=(state.shape[0], base_weights.shape[0], base_weights.shape[1]),
                dtype=jnp.float64,
            ) * safe_variance
            noisy_weights = base_weights[None, :, :] * multipliers
            adult, status, _ = _develop_core(
                jax, jnp, noisy_weights, prepared, zero_field, hard_mask, prepared, max_sweeps
            )
            return adult, (adult, status)

        adult, (adults, statuses) = jax.lax.scan(
            one_cycle, initial, jnp.arange(cycles, dtype=jnp.int32)
        )
        return adults, adult, statuses

    return run


def rollout_noisy_adult_cycles_jax(
    weights: np.ndarray,
    initial: np.ndarray,
    *,
    cycles: int,
    max_sweeps: int,
    gamma_variance: float,
    expression_flip_probability: float,
    key_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, jnp = _jax_modules()
    adults, adult, statuses = _compiled_noisy_cycles(int(cycles), int(max_sweeps))(
        jnp.asarray(weights, dtype=jnp.float64),
        jnp.asarray(initial, dtype=jnp.int8),
        jnp.asarray(gamma_variance, dtype=jnp.float64),
        jnp.asarray(expression_flip_probability, dtype=jnp.float64),
        jnp.asarray(key_data, dtype=jnp.uint32),
    )
    return (
        np.asarray(adults, dtype=np.int8),
        np.asarray(adult, dtype=np.int8),
        np.asarray(statuses, dtype=np.int8),
    )


def exact_match(states: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.all(np.asarray(states) == np.asarray(target, dtype=np.int8), axis=-1)


def longest_true_run(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.ndim != 2:
        raise ValueError("values must be cycles by futures")
    current = np.zeros(values.shape[1], dtype=np.int16)
    longest = np.zeros(values.shape[1], dtype=np.int16)
    for row in values:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def strict_destinations(
    adult_history: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    strict_run: int,
    valid_point_history: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.ones(np.asarray(adult_history).shape[:2], dtype=bool)
        if valid_point_history is None
        else np.asarray(valid_point_history, dtype=bool)
    )
    if valid.shape != np.asarray(adult_history).shape[:2]:
        raise ValueError("valid point history is not aligned with adult history")
    return (
        longest_true_run(exact_match(adult_history, target_a) & valid) >= strict_run,
        longest_true_run(exact_match(adult_history, target_b) & valid) >= strict_run,
    )


def primary_destinations(
    adult_history: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    point_states: np.ndarray,
    stable_run: int,
    valid_point_history: np.ndarray | None = None,
) -> np.ndarray:
    """Classify the first deterministic point state held for ``stable_run`` cycles.

    The prediction endpoint and the strict-retention endpoint are deliberately
    different.  Prediction uses the first deterministic point attractor held
    for three consecutive *adult developmental cycles*.  The two registered
    forms are encoded 1/2; every other trajectory is the pooled ``other``
    destination encoded 3.  Requiring the same exact point state prevents a
    sequence of different off-target points from falsely certifying a
    destination.
    """
    history = np.asarray(adult_history, dtype=np.int8)
    if history.ndim != 3:
        raise ValueError("adult_history must be cycles by futures by genes")
    futures = history.shape[1]
    valid = (
        np.ones(history.shape[:2], dtype=bool)
        if valid_point_history is None
        else np.asarray(valid_point_history, dtype=bool)
    )
    if valid.shape != history.shape[:2]:
        raise ValueError("valid point history is not aligned with adult history")
    point_codes = states_to_int(np.asarray(point_states, dtype=np.int8)).astype(np.int32)
    point_lookup = np.zeros(1 << history.shape[2], dtype=bool)
    point_lookup[point_codes] = True
    target_codes = states_to_int(np.stack((target_a, target_b))).astype(np.int32)
    pending = np.full(futures, -1, dtype=np.int32)
    streak = np.zeros(futures, dtype=np.int16)
    destination = np.zeros(futures, dtype=np.uint8)
    for cycle_index, adults in enumerate(history):
        codes = states_to_int(adults).astype(np.int32)
        active_point = point_lookup[codes] & valid[cycle_index]
        same = active_point & (codes == pending)
        streak = np.where(active_point, np.where(same, streak + 1, 1), 0).astype(np.int16)
        pending = np.where(active_point, codes, -1).astype(np.int32)
        newly_resolved = (destination == 0) & (streak >= stable_run)
        category = np.where(
            codes == target_codes[0],
            1,
            np.where(codes == target_codes[1], 2, 3),
        ).astype(np.uint8)
        destination = np.where(newly_resolved, category, destination).astype(np.uint8)
    # The registered committor has exactly A/B/other categories.  A trajectory
    # without a three-cycle deterministic point destination belongs to the
    # pooled other category rather than being silently dropped.
    return np.where(destination == 0, 3, destination).astype(np.uint8)


def apply_challenge(
    states: np.ndarray,
    challenge: str,
    *,
    forced_state: np.ndarray,
    neutral_damage_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(states, dtype=np.int8).copy()
    if challenge == "release_only":
        return result
    if challenge == "forced_break":
        return np.repeat(np.asarray(forced_state, dtype=np.int8)[None, :], result.shape[0], axis=0)
    if challenge == "neutral_damage":
        count = max(1, int(round(result.shape[1] * neutral_damage_fraction)))
        for future in range(result.shape[0]):
            indices = rng.choice(result.shape[1], size=count, replace=False)
            result[future, indices] *= -1
        return result
    raise ValueError(f"unknown challenge: {challenge}")
