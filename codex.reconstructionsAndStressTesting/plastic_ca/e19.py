"""Bit-exact clean-room implementation of the frozen E19 ECA contract.

This module is deliberately separate from :mod:`plastic_ca.eca`.  The older
integer engine records the convention search; this NumPy engine implements the
code-free execution trace supplied after that search, including its batched
PCG64 draw order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from .eca import DISPUTED, rule_descriptors, wolfram_class


PINNED_NUMPY = "2.5.2"
GOLDEN_FIXTURE = Path(__file__).with_name("data") / "e19_golden_traces.json"

LAUNCH_HEX = (
    "0000000080000000",
    "0000000180000000",
    "00000003c0000000",
    "0000000ff0000000",
    "aaaaaaaaaaaaaaaa",
    "ffffffff00000000",
    "23a031300051000a",
    "6420581808011102",
    "4450081105301018",
    "01c1385c048a0f68",
    "2349a69c817d7b9b",
    "a13c035191c53554",
    "9d65705fb9e2a777",
    "89f7b38cfeff37ed",
    "27deeff5fff7aefd",
    "dfd73effdfef7dea",
)


@dataclass(frozen=True)
class E19Contract:
    """Frozen, explicit contract reconstructed from the golden trace pack."""

    width: int = 64
    activity_budget: int = 256
    min_sweeps: int = 4
    max_sweeps: int = 128
    flip_noise: float = 0.01
    copy_error: float = 0.015
    n_seeds: int = 16
    futures_per_seed: int = 128
    horizon: int = 32
    break_horizon: int = 8
    inherit: float = 0.9
    coherence: float = 0.9
    distinct: float = 0.85
    strict_run: int = 8
    form_mass_quantile: float = 0.5
    rng_tag: str = "eca-traj-v1"
    launch_rows_hex: tuple[str, ...] = field(default=LAUNCH_HEX)

    def __post_init__(self) -> None:
        if self.width != 64:
            raise ValueError("the frozen E19 contract has width 64")
        if self.n_seeds > len(self.launch_rows_hex):
            raise ValueError("the frozen launch library contains only 16 rows")
        if self.futures_per_seed <= 0 or self.horizon <= 0:
            raise ValueError("future and horizon counts must be positive")
        for probability in (self.flip_noise, self.copy_error):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("noise probabilities must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "launch_preparation": "none",
                "composition_zero": "first_completed_generation",
                "process_noise_order": "post_rule",
                "activity_count": "realized_post_noise",
                "boundary_death": "timeout_or_terminal_monochrome",
                "observed_daughter": "terminal_pre_copy",
                "copy_draw": "unconditional_generation_batch",
                "cell_zero_bit_order": "most_significant_bit",
                "form_pooling": (
                    "per seed, equal mean of each broken future's last completed composition; "
                    "the break-causing daughter satisfies the later-generation requirement"
                ),
            }
        )
        return value

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class E19RuleResult:
    rule: int
    wolfram_class: int
    disputed: bool
    strict: float
    break_by_8: float
    median_gen_sweeps: float
    mean_survival: float
    form_supports: tuple[int, ...]
    descriptors: dict[str, float]
    n_futures: int
    total_sweeps: int
    death_counts: dict[str, int]
    per_seed: tuple[dict[str, Any], ...]

    @property
    def library(self) -> frozenset[int]:
        return frozenset(value for value in self.form_supports if value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["form_supports"] = list(self.form_supports)
        value["per_seed"] = list(self.per_seed)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "E19RuleResult":
        return cls(
            rule=int(value["rule"]),
            wolfram_class=int(value["wolfram_class"]),
            disputed=bool(value["disputed"]),
            strict=float(value["strict"]),
            break_by_8=float(value["break_by_8"]),
            median_gen_sweeps=float(value["median_gen_sweeps"]),
            mean_survival=float(value["mean_survival"]),
            form_supports=tuple(int(item) for item in value["form_supports"]),
            descriptors={str(k): float(v) for k, v in value["descriptors"].items()},
            n_futures=int(value["n_futures"]),
            total_sweeps=int(value["total_sweeps"]),
            death_counts={str(k): int(v) for k, v in value["death_counts"].items()},
            per_seed=tuple(dict(item) for item in value["per_seed"]),
        )


def require_pinned_numpy() -> None:
    if np.__version__ != PINNED_NUMPY:
        raise RuntimeError(
            f"E19 requires NumPy {PINNED_NUMPY} for an auditable RNG environment; "
            f"found {np.__version__}"
        )


def _hex_to_row(value: str, width: int = 64) -> np.ndarray:
    integer = int(value, 16)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((np.uint64(integer) >> shifts) & np.uint64(1)).astype(np.bool_)


def _row_to_hex(row: np.ndarray) -> str:
    value = 0
    for bit in row:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _integer_step(row: int, rule: int, width: int = 64) -> int:
    """MSB-cell-zero scalar step used only by the independent golden replay."""

    cells = [((row >> (width - 1 - index)) & 1) for index in range(width)]
    updated = 0
    for index in range(width):
        neighbourhood = 4 * cells[index - 1] + 2 * cells[index] + cells[(index + 1) % width]
        updated = (updated << 1) | ((rule >> neighbourhood) & 1)
    return updated


def e19_step(states: np.ndarray, rule: int) -> np.ndarray:
    """Apply a numbered ECA rule to MSB-first rows shaped ``(n, 64)``."""

    if states.ndim != 2 or states.shape[1] != 64:
        raise ValueError("E19 state batches must have shape (n, 64)")
    left = np.roll(states, 1, axis=1).astype(np.uint8, copy=False)
    centre = states.astype(np.uint8, copy=False)
    right = np.roll(states, -1, axis=1).astype(np.uint8, copy=False)
    index = (left << 2) | (centre << 1) | right
    return (((np.uint16(rule) >> index) & 1) != 0)


def final4_counts(states: np.ndarray) -> np.ndarray:
    """Return the E19 16-bin cyclic final4 census for each input row."""

    if states.ndim != 2 or states.shape[1] != 64:
        raise ValueError("E19 state batches must have shape (n, 64)")
    codes = (
        (np.roll(states, 1, axis=1).astype(np.uint8) << 3)
        | (states.astype(np.uint8) << 2)
        | (np.roll(states, -1, axis=1).astype(np.uint8) << 1)
        | np.roll(states, -2, axis=1).astype(np.uint8)
    )
    counts = np.zeros((states.shape[0], 16), dtype=np.float64)
    for code in range(16):
        counts[:, code] = np.count_nonzero(codes == code, axis=1)
    return counts


def spacetime_codes(histories: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Return row-major 3x3 spacetime codes shaped ``(n, 64)``."""

    n = histories[0].shape[0]
    codes = np.zeros((n, 64), dtype=np.uint16)
    bit = 0
    for row in histories:
        for delta in (-1, 0, 1):
            codes |= np.roll(row, -delta, axis=1).astype(np.uint16) << bit
            bit += 1
    return codes


def figure_states(
    histories: tuple[np.ndarray, np.ndarray, np.ndarray],
    domain_codes: frozenset[int],
) -> np.ndarray:
    """Mark cells whose local spacetime word is outside a rule domain."""

    codes = spacetime_codes(histories)
    domain = np.fromiter(sorted(domain_codes), dtype=np.uint16)
    return ~np.isin(codes, domain)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _mass_support(vector: np.ndarray, quantile: float) -> int:
    total = float(np.maximum(vector, 0.0).sum())
    if total <= 0.0:
        return 0
    order = sorted(range(len(vector)), key=lambda index: (-float(vector[index]), index))
    threshold = quantile * total
    cumulative = 0.0
    support = 0
    for index in order:
        if vector[index] <= 0.0:
            continue
        cumulative += float(vector[index])
        support |= 1 << index
        if cumulative >= threshold:
            break
    return support


def trajectory_seed(rule: int, seed_index: int, tag: str = "eca-traj-v1") -> int:
    digest = hashlib.sha256(f"{tag}:{rule}:{seed_index}".encode()).digest()
    return int.from_bytes(digest[:16], "little")


def load_golden_fixture(path: Path | None = None) -> dict[str, Any]:
    with (path or GOLDEN_FIXTURE).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_golden_fixture(path: Path | None = None) -> dict[str, Any]:
    """Replay every disclosed mask without using the NumPy campaign engine."""

    fixture = load_golden_fixture(path)
    errors: list[str] = []
    launch_checks = 0
    sweep_checks = 0
    spectrum_checks = 0
    for expected_index, launch in enumerate(fixture["launch_rows"]):
        launch_checks += 1
        if int(launch["seed_index"]) != expected_index:
            errors.append(f"launch index {expected_index}: seed_index mismatch")
        value = int(launch["hex"], 16)
        if value.bit_count() != int(launch["ones"]):
            errors.append(f"launch index {expected_index}: population mismatch")
        if launch["hex"].lower() != LAUNCH_HEX[expected_index]:
            errors.append(f"launch index {expected_index}: authoritative row mismatch")

    for rule_text, settings in fixture["traces"].items():
        rule = int(rule_text)
        for setting_name, setting in settings.items():
            for generation in setting["generations"]:
                row = int(generation["start_row_hex"], 16)
                activity = 0
                masks = generation["process_masks_nonzero"]
                expected_rows = generation["rows_hex_per_sweep"]
                expected_changes = generation["changed_per_sweep"]
                for sweep, (expected_row, expected_change) in enumerate(
                    zip(expected_rows, expected_changes, strict=True), start=1
                ):
                    previous = row
                    row = _integer_step(previous, rule) ^ int(masks.get(str(sweep), "0"), 16)
                    changed = (row ^ previous).bit_count()
                    activity += changed
                    sweep_checks += 1
                    label = f"rule {rule} {setting_name} gen {generation['generation']} sweep {sweep}"
                    if row != int(expected_row, 16):
                        errors.append(f"{label}: row mismatch")
                    if changed != int(expected_change):
                        errors.append(f"{label}: activity increment mismatch")
                label = f"rule {rule} {setting_name} gen {generation['generation']}"
                if row != int(generation["terminal_row_hex"], 16):
                    errors.append(f"{label}: terminal mismatch")
                if activity != int(generation["activity_count"]):
                    errors.append(f"{label}: total activity mismatch")
                counts = final4_counts(_hex_to_row(f"{row:016x}")[None, :])[0].astype(int).tolist()
                spectrum_checks += 1
                if counts != generation["final4_counts_of_64"]:
                    errors.append(f"{label}: final4 mismatch")
                offspring = row ^ int(generation["copy_mask_hex"], 16)
                if offspring != int(generation["offspring_row_hex"], 16):
                    errors.append(f"{label}: copy replay mismatch")

    return {
        "passed": not errors,
        "launch_checks": launch_checks,
        "sweep_checks": sweep_checks,
        "spectrum_checks": spectrum_checks,
        "total_checks": launch_checks + sweep_checks + spectrum_checks,
        "expected_sweep_checks": 907,
        "expected_spectrum_checks": 15,
        "errors": errors,
        "fixture_sha256": hashlib.sha256((path or GOLDEN_FIXTURE).read_bytes()).hexdigest(),
    }


def _update_observer_state(
    compositions: np.ndarray,
    lengths: np.ndarray,
    first_break: np.ndarray,
    broke_by_8: np.ndarray,
    strict_positive: np.ndarray,
    future: int,
    composition: np.ndarray,
    contract: E19Contract,
) -> None:
    position = int(lengths[future])
    compositions[future, position] = composition
    lengths[future] += 1
    length = position + 1
    if length < 2:
        return

    boundary = length - 2
    similarity = _cosine(compositions[future, boundary], compositions[future, boundary + 1])
    if first_break[future] < 0 and similarity <= contract.inherit:
        first_break[future] = boundary
        # The retained E19 column named ``break_by_8`` uses completed
        # generation indices 0..7, hence only the seven observable fidelity
        # boundaries among those generations (boundary indices 0..6).
        if boundary < contract.break_horizon - 1:
            broke_by_8[future] = True

    break_index = int(first_break[future])
    if break_index < 0 or length < contract.strict_run + 2:
        return
    start = length - contract.strict_run - 1
    if start < break_index + 1:
        return
    if not all(
        _cosine(compositions[future, index], compositions[future, index + 1]) > contract.inherit
        for index in range(start, start + contract.strict_run)
    ):
        return
    daughters = compositions[future, start + 1 : start + contract.strict_run + 1]
    for left in range(contract.strict_run):
        for right in range(left):
            if _cosine(daughters[left], daughters[right]) <= contract.coherence:
                return
    # E19 freezes the break-causing daughter (the composition on the right of
    # the failed boundary) as its comparison anchor.  This historical index
    # convention is counterintuitive but is numerically pinned by the atlas.
    anchor = compositions[future, break_index + 1]
    if any(_cosine(daughter, anchor) > contract.distinct for daughter in daughters):
        return
    strict_positive[future] = True


def _simulate_seed(
    rule: int,
    seed_index: int,
    contract: E19Contract,
    domain_codes: frozenset[int] | None = None,
    *,
    capture: bool = False,
) -> dict[str, Any]:
    n = contract.futures_per_seed
    launch = _hex_to_row(contract.launch_rows_hex[seed_index])
    states = np.repeat(launch[None, :], n, axis=0)
    alive = np.ones(n, dtype=np.bool_)
    strict_positive = np.zeros(n, dtype=np.bool_)
    broke_by_8 = np.zeros(n, dtype=np.bool_)
    first_break = np.full(n, -1, dtype=np.int16)
    lengths = np.zeros(n, dtype=np.int16)
    compositions = np.zeros((n, contract.horizon, 16), dtype=np.float64)
    first_generation_times = np.zeros(n, dtype=np.int16)
    total_sweeps = 0
    deaths: Counter[str] = Counter()
    captures: list[list[dict[str, Any]]] | None = ([[] for _ in range(n)] if capture else None)
    rng = np.random.default_rng(trajectory_seed(rule, seed_index, contract.rng_tag))

    for generation in range(contract.horizon):
        batch = np.flatnonzero(alive & ~strict_positive)
        if not len(batch):
            break
        current = states[batch].copy()
        history = [current.copy(), current.copy(), current.copy()]
        batch_size = len(batch)
        activity = np.zeros(batch_size, dtype=np.int32)
        sweeps = np.zeros(batch_size, dtype=np.int16)
        reached_budget = np.zeros(batch_size, dtype=np.bool_)

        for sweep in range(1, contract.max_sweeps + 1):
            active = np.flatnonzero(~reached_budget)
            if not len(active):
                break
            previous = current[active]
            terminal = e19_step(previous, rule)
            if contract.flip_noise > 0.0:
                terminal ^= rng.random((len(active), contract.width)) < contract.flip_noise
            activity[active] += np.count_nonzero(terminal != previous, axis=1)
            current[active] = terminal
            history[0][active] = history[1][active]
            history[1][active] = history[2][active]
            history[2][active] = terminal
            sweeps[active] = sweep
            if sweep >= contract.min_sweeps:
                reached_budget[active[activity[active] >= contract.activity_budget]] = True

        total_sweeps += int(sweeps.sum())
        if generation == 0:
            first_generation_times[:] = sweeps

        monochrome = (~current.any(axis=1)) | current.all(axis=1)
        timed_out = ~reached_budget
        dead = timed_out | monochrome
        for timeout, mono in zip(timed_out, monochrome, strict=True):
            if timeout and mono:
                deaths["timeout_and_monochrome"] += 1
            elif timeout:
                deaths["timeout"] += 1
            elif mono:
                deaths["monochrome"] += 1

        # The draw is mandatory for every entrant, including dead and newly
        # strict-positive futures.  This is part of the subsequent RNG state.
        copy_masks = rng.random((batch_size, contract.width)) < contract.copy_error
        offspring = current ^ copy_masks

        if domain_codes is None:
            observed_states = current
            observer_empty = np.zeros(batch_size, dtype=np.bool_)
        else:
            observed_states = figure_states((history[0], history[1], history[2]), domain_codes)
            observer_empty = ~observed_states.any(axis=1)
            deaths["observer_empty"] += int(observer_empty.sum())
        stopped = dead | observer_empty
        divided_local = np.flatnonzero(~stopped)
        if len(divided_local):
            counts = final4_counts(observed_states[divided_local])
            for local, composition in zip(divided_local, counts, strict=True):
                future = int(batch[local])
                _update_observer_state(
                    compositions,
                    lengths,
                    first_break,
                    broke_by_8,
                    strict_positive,
                    future,
                    composition,
                    contract,
                )
                states[future] = offspring[local]
                if captures is not None:
                    captures[future].append(
                        {
                            "generation": generation,
                            "terminal_row_hex": _row_to_hex(current[local]),
                            "offspring_row_hex": _row_to_hex(offspring[local]),
                            "composition": composition.astype(float).tolist(),
                            "sweeps": int(sweeps[local]),
                            "activity": int(activity[local]),
                            "strict_after_generation": bool(strict_positive[future]),
                            "first_break": (
                                int(first_break[future]) if first_break[future] >= 0 else None
                            ),
                        }
                    )
        alive[batch[stopped]] = False

    qualifying: list[np.ndarray] = []
    for future in range(n):
        break_index = int(first_break[future])
        length = int(lengths[future])
        # The break-causing daughter at break_index + 1 is itself the required
        # later generation.  No additional post-break division is required.
        if break_index >= 0 and length >= break_index + 2:
            qualifying.append(compositions[future, length - 1])
    support = _mass_support(np.mean(qualifying, axis=0), contract.form_mass_quantile) if qualifying else 0
    result = {
        "seed_index": seed_index,
        "strict_count": int(strict_positive.sum()),
        "break_by_8_count": int(broke_by_8.sum()),
        "survival_sum": int(lengths.sum()),
        "first_generation_times": first_generation_times.astype(int).tolist(),
        "form_support": int(support),
        "form_n_futures": len(qualifying),
        "total_sweeps": total_sweeps,
        "death_counts": dict(sorted(deaths.items())),
    }
    if captures is not None:
        result["captures"] = captures
    return result


def capture_e19_seed(
    rule: int,
    seed_index: int,
    contract: E19Contract = E19Contract(),
    *,
    domain_codes: frozenset[int] | None = None,
) -> dict[str, Any]:
    """Run one frozen seed batch while retaining generation-level states.

    Capture is observational: the mandatory draw order and returned summary
    fields are identical to :func:`_simulate_seed` with capture disabled.
    """

    return _simulate_seed(rule, seed_index, contract, domain_codes, capture=True)


def evaluate_e19_rule(
    rule: int,
    contract: E19Contract = E19Contract(),
    *,
    domain_codes: frozenset[int] | None = None,
) -> E19RuleResult:
    if not 0 <= rule <= 255:
        raise ValueError("an ECA rule must lie in [0, 255]")
    require_pinned_numpy()
    seeds = tuple(
        _simulate_seed(rule, seed_index, contract, domain_codes)
        for seed_index in range(contract.n_seeds)
    )
    n_futures = contract.n_seeds * contract.futures_per_seed
    strict_count = sum(int(seed["strict_count"]) for seed in seeds)
    break_count = sum(int(seed["break_by_8_count"]) for seed in seeds)
    survival_sum = sum(int(seed["survival_sum"]) for seed in seeds)
    first_times = [int(value) for seed in seeds for value in seed["first_generation_times"]]
    death_counts: Counter[str] = Counter()
    for seed in seeds:
        death_counts.update(seed["death_counts"])
    return E19RuleResult(
        rule=rule,
        wolfram_class=wolfram_class(rule),
        disputed=rule in DISPUTED,
        strict=strict_count / n_futures,
        break_by_8=break_count / n_futures,
        median_gen_sweeps=float(median(first_times)),
        mean_survival=survival_sum / n_futures,
        form_supports=tuple(
            int(seed["form_support"]) for seed in seeds if int(seed["form_n_futures"]) > 0
        ),
        descriptors=rule_descriptors(rule),
        n_futures=n_futures,
        total_sweeps=sum(int(seed["total_sweeps"]) for seed in seeds),
        death_counts=dict(sorted(death_counts.items())),
        per_seed=seeds,
    )


def diagnostic_rules(
    rules: Sequence[int] = (8, 13, 35, 110, 172),
    contract: E19Contract = E19Contract(),
) -> list[dict[str, Any]]:
    return [evaluate_e19_rule(rule, contract).to_dict() for rule in rules]
