"""Two source-isolated cellular-automaton substrates.

The Evoloop state dynamics are the public nine-state Sayama/Golly rule.  The
provenance array used for genealogy is passive metadata and never enters the
transition lookup.  The protocell follows the random-order replication,
degradation, and diffusion contract in Kamimura and Kaneko (2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .core import connected_components_torus, crop_component


TASK_ROOT = Path(__file__).resolve().parent
EVOLOOP_TABLE_PATH = TASK_ROOT / "evoloop.table"

PROTOCELL_PY = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
PROTOCELL_AY = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
EVOLOOP_COUNTS = (4, 8, 16, 32)
EVOLOOP_IMMIGRATION = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    side: int
    mechanics_seeds: int
    mechanics_required: int
    mechanics_boundaries: int
    mechanics_cap: int
    boundary_cap: int
    protocell_persistence: int
    evoloop_persistence: int
    evoloop_arm_window: int
    calibration_blocks: int
    calibration_pairs: int
    pilot_blocks: int
    confirmation_blocks: int
    landmarks: tuple[int, ...]
    pilot_branches: int
    confirmation_branches: int


FULL_PROFILE = ModelProfile(
    name="full",
    side=256,
    mechanics_seeds=16,
    mechanics_required=12,
    mechanics_boundaries=20,
    mechanics_cap=2_000_000,
    boundary_cap=100_000,
    protocell_persistence=64,
    evoloop_persistence=512,
    evoloop_arm_window=4_096,
    calibration_blocks=24,
    calibration_pairs=100,
    pilot_blocks=32,
    confirmation_blocks=128,
    landmarks=(20, 35, 50, 65, 80),
    pilot_branches=32,
    confirmation_branches=64,
)

SMOKE_PROFILE = ModelProfile(
    name="smoke",
    side=64,
    mechanics_seeds=2,
    mechanics_required=1,
    mechanics_boundaries=1,
    mechanics_cap=2_500,
    boundary_cap=2_500,
    protocell_persistence=2,
    evoloop_persistence=8,
    evoloop_arm_window=32,
    calibration_blocks=2,
    calibration_pairs=4,
    pilot_blocks=2,
    confirmation_blocks=2,
    landmarks=(1, 2),
    pilot_branches=2,
    confirmation_branches=4,
)


def profile_named(name: str) -> ModelProfile:
    if name == "full":
        return FULL_PROFILE
    if name == "smoke":
        return SMOKE_PROFILE
    raise ValueError(f"unknown profile: {name}")


@dataclass(frozen=True)
class ProtocellParameters:
    p_y: float
    a_y: float
    p_x: float
    a_x: float = 0.01

    @classmethod
    def from_pair(cls, p_y: float, a_y: float) -> "ProtocellParameters":
        return cls(p_y=float(p_y), a_y=float(a_y), p_x=1.0 - float(p_y))

    @property
    def key(self) -> str:
        return f"py={self.p_y:.0e},ay={self.a_y:.0e}"


@dataclass(frozen=True)
class EvoloopParameters:
    initial_count: int
    immigration_per_10000: float

    @property
    def key(self) -> str:
        return f"n={self.initial_count},imm={self.immigration_per_10000:g}"


@dataclass
class ProtocellWorld:
    grid: NDArray[np.uint8]

    def copy(self) -> "ProtocellWorld":
        return ProtocellWorld(self.grid.copy())


@dataclass
class EvoloopWorld:
    grid: NDArray[np.uint8]
    provenance: NDArray[np.int32]
    focal_label: int
    next_label: int

    def copy(self) -> "EvoloopWorld":
        return EvoloopWorld(
            self.grid.copy(),
            self.provenance.copy(),
            int(self.focal_label),
            int(self.next_label),
        )


@dataclass(frozen=True)
class BoundaryTransition:
    parent: NDArray[np.uint8]
    child: NDArray[np.uint8]
    elapsed_updates: int
    ambiguous: bool
    extinct: bool
    occupancy_exceeded: bool


def protocell_initial(side: int) -> ProtocellWorld:
    if side < 16:
        raise ValueError("protocell arena is too small")
    grid = np.zeros((side, side), dtype=np.uint8)
    start = side // 2 - 5
    grid[start : start + 10, start : start + 10] = 1
    grid[side // 2, side // 2] = 2
    return ProtocellWorld(grid)


def _neighbors(row: int, col: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    return [
        ((row - 1) % rows, col),
        ((row + 1) % rows, col),
        (row, (col - 1) % cols),
        (row, (col + 1) % cols),
    ]


def _reaction_empty_sites(
    first: tuple[int, int], second: tuple[int, int], grid: NDArray[np.uint8]
) -> list[tuple[int, int]]:
    excluded = {first, second}
    sites = {
        site
        for source in (first, second)
        for site in _neighbors(source[0], source[1], grid.shape)
        if site not in excluded and grid[site] == 0
    }
    return sorted(sites)


def protocell_sweep(
    grid: NDArray[np.uint8], parameters: ProtocellParameters, rng: np.random.Generator
) -> None:
    """Apply one random-order replication/degradation/diffusion sweep in place."""

    occupied = np.argwhere(grid > 0)
    for order_index in rng.permutation(occupied.shape[0]):
        row, col = (int(value) for value in occupied[order_index])
        species = int(grid[row, col])
        if species not in (1, 2):
            continue
        opposite = 3 - species
        catalysts = [
            site
            for site in _neighbors(row, col, grid.shape)
            if int(grid[site]) == opposite
        ]
        probability = parameters.p_x if species == 1 else parameters.p_y
        if catalysts and rng.random() < probability:
            catalyst = catalysts[int(rng.integers(len(catalysts)))]
            empty = _reaction_empty_sites((row, col), catalyst, grid)
            if empty:
                target = empty[int(rng.integers(len(empty)))]
                grid[target] = species

    occupied = np.argwhere(grid > 0)
    for order_index in rng.permutation(occupied.shape[0]):
        row, col = (int(value) for value in occupied[order_index])
        species = int(grid[row, col])
        if species == 0:
            continue
        probability = parameters.a_x if species == 1 else parameters.a_y
        if rng.random() < probability:
            grid[row, col] = 0

    occupied = np.argwhere(grid > 0)
    for order_index in rng.permutation(occupied.shape[0]):
        row, col = (int(value) for value in occupied[order_index])
        if grid[row, col] == 0:
            continue
        destinations = _neighbors(row, col, grid.shape)
        target = destinations[int(rng.integers(4))]
        if grid[target] == 0:
            grid[target] = grid[row, col]
            grid[row, col] = 0


def qualifying_protocell_components(
    grid: NDArray[np.uint8], min_size: int = 20
) -> list[NDArray[np.int32]]:
    # Diffusion makes localized molecular clouds porous.  Radius-two proximity
    # recognizes a cloud without changing any CA state or bridging a genuine
    # division once the two daughters separate.
    occupied = {tuple(int(value) for value in point) for point in np.argwhere(grid > 0)}
    components: list[NDArray[np.int32]] = []
    rows, cols = grid.shape
    offsets = [
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if (dr or dc) and max(abs(dr), abs(dc)) <= 2
    ]
    while occupied:
        seed = occupied.pop()
        stack = [seed]
        points = [seed]
        while stack:
            row, col = stack.pop()
            for dr, dc in offsets:
                nxt = ((row + dr) % rows, (col + dc) % cols)
                if nxt in occupied:
                    occupied.remove(nxt)
                    points.append(nxt)
                    stack.append(nxt)
        if len(points) >= min_size:
            components.append(np.asarray(points, dtype=np.int32))
    components.sort(key=lambda item: -item.shape[0])
    return [
        component
        for component in components
        if np.any(grid[component[:, 0], component[:, 1]] == 2)
    ]


def _recenter_crop(crop: NDArray[np.uint8], side: int) -> NDArray[np.uint8]:
    if crop.shape[0] > side or crop.shape[1] > side:
        raise ValueError("individual does not fit in arena")
    grid = np.zeros((side, side), dtype=np.uint8)
    row = (side - crop.shape[0]) // 2
    col = (side - crop.shape[1]) // 2
    grid[row : row + crop.shape[0], col : col + crop.shape[1]] = crop
    return grid


def protocell_daughter_lobes(
    grid: NDArray[np.uint8], *, min_size: int = 20, min_separation: int = 8
) -> list[NDArray[np.int32]]:
    """Return two persistent-candidate lobes centred on separated Y molecules."""

    y_points = np.argwhere(grid == 2)
    if y_points.shape[0] < 2:
        return []
    shape = np.asarray(grid.shape, dtype=np.int32)
    best_pair: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None
    best_distance = -1.0
    for first in range(y_points.shape[0] - 1):
        for second in range(first + 1, y_points.shape[0]):
            delta = np.abs(y_points[first] - y_points[second])
            delta = np.minimum(delta, shape - delta)
            distance = float(np.sqrt(np.square(delta).sum()))
            if distance > best_distance:
                best_distance = distance
                best_pair = (y_points[first], y_points[second])
    if best_pair is None or best_distance < min_separation:
        return []
    occupied = np.argwhere(grid > 0)
    assignments: list[list[tuple[int, int]]] = [[], []]
    for point in occupied:
        distances = []
        for center in best_pair:
            delta = np.abs(point - center)
            delta = np.minimum(delta, shape - delta)
            distances.append(float(np.square(delta).sum()))
        assignments[int(np.argmin(distances))].append((int(point[0]), int(point[1])))
    if any(len(points) < min_size for points in assignments):
        return []
    return [np.asarray(points, dtype=np.int32) for points in assignments]


def advance_protocell_boundary(
    world: ProtocellWorld,
    parameters: ProtocellParameters,
    rng: np.random.Generator,
    *,
    cap: int,
    persistence: int,
    occupancy_limit: float = 0.25,
) -> tuple[ProtocellWorld, BoundaryTransition]:
    grid = world.grid.copy()
    components = qualifying_protocell_components(grid)
    if not components:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return world.copy(), BoundaryTransition(empty, empty, 0, False, True, False)
    parent_crop = crop_component(grid, components[0])
    last_single = parent_crop
    elapsed = 0
    while elapsed < cap:
        protocell_sweep(grid, parameters, rng)
        elapsed += 1
        if np.count_nonzero(grid) > occupancy_limit * grid.size:
            empty = np.zeros((0, 0), dtype=np.uint8)
            return world.copy(), BoundaryTransition(last_single, empty, elapsed, False, False, True)
        if elapsed % 4:
            continue
        if not np.any(grid == 2):
            empty = np.zeros((0, 0), dtype=np.uint8)
            return world.copy(), BoundaryTransition(last_single, empty, elapsed, False, True, False)
        components = protocell_daughter_lobes(grid)
        if len(components) < 2:
            if elapsed % 20 == 0:
                clusters = qualifying_protocell_components(grid)
                if clusters:
                    last_single = crop_component(grid, clusters[0])
            continue
        candidate_grid = grid.copy()
        stable = True
        persistence_completed = 0
        for _ in range(min(persistence, cap - elapsed)):
            protocell_sweep(candidate_grid, parameters, rng)
            elapsed += 1
            persistence_completed += 1
            if np.count_nonzero(candidate_grid) > occupancy_limit * candidate_grid.size:
                empty = np.zeros((0, 0), dtype=np.uint8)
                return world.copy(), BoundaryTransition(last_single, empty, elapsed, False, False, True)
            if not np.any(candidate_grid == 2):
                stable = False
                break
            stable_components = protocell_daughter_lobes(candidate_grid)
            if len(stable_components) < 2:
                stable = False
                break
        if not stable or persistence_completed < persistence:
            grid = candidate_grid
            continue
        stable_components = protocell_daughter_lobes(candidate_grid)[:2]
        choice = int(rng.integers(2))
        child_crop = crop_component(candidate_grid, stable_components[choice])
        child_world = ProtocellWorld(_recenter_crop(child_crop, grid.shape[0]))
        return child_world, BoundaryTransition(
            last_single,
            child_crop,
            elapsed,
            False,
            False,
            False,
        )
    empty = np.zeros((0, 0), dtype=np.uint8)
    return world.copy(), BoundaryTransition(last_single, empty, cap, False, True, False)


EVOLOOP_SEED = np.asarray(
    [
        [0, 2, 2, 2, 2, 2, 2, 2, 0],
        [2, 7, 0, 1, 7, 0, 1, 7, 2],
        [2, 1, 2, 2, 2, 2, 2, 0, 2],
        [2, 0, 2, 0, 0, 0, 2, 1, 2],
        [2, 7, 2, 0, 0, 0, 2, 7, 2],
        [2, 1, 2, 0, 0, 0, 2, 0, 2],
        [2, 0, 2, 2, 2, 2, 2, 1, 2],
        [2, 7, 1, 0, 4, 1, 0, 3, 2],
        [0, 2, 2, 2, 2, 2, 2, 5, 0],
    ],
    dtype=np.uint8,
)


def _parse_token(token: str, variables: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    token = token.strip()
    if token.isdigit():
        return (int(token),)
    if token not in variables:
        raise ValueError(f"undefined rule-table token: {token}")
    return variables[token]


def parse_evoloop_table(path: Path = EVOLOOP_TABLE_PATH) -> list[tuple[tuple[tuple[int, ...], ...], int]]:
    variables: dict[str, tuple[int, ...]] = {}
    rules: list[tuple[tuple[tuple[int, ...], ...], int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"var\s+(\w+)=\{([0-8,]+)\}", line)
        if match:
            variables[match.group(1)] = tuple(int(value) for value in match.group(2).split(","))
            continue
        if line.startswith(("n_states:", "neighborhood:", "symmetries:")):
            continue
        pieces = line.split(",")
        if len(pieces) != 6:
            raise ValueError(f"invalid Evoloop rule row: {line}")
        inputs = tuple(_parse_token(token, variables) for token in pieces[:5])
        if not pieces[5].isdigit():
            raise ValueError("Evoloop output tokens must be literal states")
        rules.append((inputs, int(pieces[5])))
    if len(rules) != 132:
        raise ValueError(f"expected 132 Evoloop rules, found {len(rules)}")
    return rules


def build_evoloop_lut(path: Path = EVOLOOP_TABLE_PATH) -> tuple[NDArray[np.uint8], int]:
    """Build the complete 9^5 lookup table; unmatched cells retain state."""

    rules = parse_evoloop_table(path)
    lut = np.empty((9, 9, 9, 9, 9), dtype=np.uint8)
    covered = np.zeros(lut.shape, dtype=bool)
    for center, north, east, south, west in product(range(9), repeat=5):
        output = center
        matched = False
        neighborhoods = (
            (north, east, south, west),
            (east, south, west, north),
            (south, west, north, east),
            (west, north, east, south),
        )
        for inputs, candidate in rules:
            if center not in inputs[0]:
                continue
            for rotated in neighborhoods:
                if all(rotated[index] in inputs[index + 1] for index in range(4)):
                    output = candidate
                    matched = True
                    break
            if matched:
                break
        lut[center, north, east, south, west] = output
        covered[center, north, east, south, west] = matched
    return lut, int(covered.sum())


class EvoloopRule:
    def __init__(self, path: Path = EVOLOOP_TABLE_PATH):
        self.lut, self.covered_neighborhoods = build_evoloop_lut(path)

    def step(self, grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
        north = np.roll(grid, 1, axis=0)
        east = np.roll(grid, -1, axis=1)
        south = np.roll(grid, -1, axis=0)
        west = np.roll(grid, 1, axis=1)
        return self.lut[grid, north, east, south, west]


def _passive_provenance_step(
    before: NDArray[np.uint8], after: NDArray[np.uint8], provenance: NDArray[np.int32]
) -> NDArray[np.int32]:
    output = provenance.copy()
    output[after == 0] = 0
    births = (before == 0) & (after != 0)
    for row, col in np.argwhere(births):
        labels = [
            int(provenance[site])
            for site in _neighbors(int(row), int(col), before.shape)
            if before[site] != 0 and provenance[site] > 0
        ]
        if labels:
            counts = {label: labels.count(label) for label in set(labels)}
            output[row, col] = min(counts, key=lambda label: (-counts[label], label))
        else:
            output[row, col] = 0
    return output


def place_pattern(
    grid: NDArray[np.uint8],
    pattern: NDArray[np.uint8],
    row: int,
    col: int,
    *,
    provenance: NDArray[np.int32] | None = None,
    label: int = 0,
    require_clear_box: bool = False,
) -> bool:
    rows, cols = grid.shape
    if require_clear_box:
        box = [
            ((row + rr) % rows, (col + cc) % cols)
            for rr in range(pattern.shape[0])
            for cc in range(pattern.shape[1])
        ]
        if any(grid[target] != 0 for target in box):
            return False
    coords = np.argwhere(pattern > 0)
    targets = [((row + int(rr)) % rows, (col + int(cc)) % cols) for rr, cc in coords]
    if any(grid[target] != 0 for target in targets):
        return False
    for (rr, cc), target in zip(coords, targets, strict=True):
        grid[target] = pattern[int(rr), int(cc)]
        if provenance is not None:
            provenance[target] = label
    return True


def evoloop_initial(
    side: int,
    count: int,
    rng: np.random.Generator,
) -> EvoloopWorld:
    if side < 32:
        raise ValueError("Evoloop arena is too small")
    grid = np.zeros((side, side), dtype=np.uint8)
    provenance = np.zeros((side, side), dtype=np.int32)
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 256:
        attempts += 1
        pattern = np.rot90(EVOLOOP_SEED, int(rng.integers(4)))
        row = int(rng.integers(side))
        col = int(rng.integers(side))
        if place_pattern(
            grid,
            pattern,
            row,
            col,
            provenance=provenance,
            label=placed + 1,
            require_clear_box=True,
        ):
            placed += 1
    if placed != count:
        raise RuntimeError(f"could place only {placed}/{count} Evoloops")
    return EvoloopWorld(grid, provenance, focal_label=1, next_label=count + 1)


def _component_holes(crop: NDArray[np.uint8]) -> int:
    if crop.size == 0:
        return 0
    padded = np.pad(crop != 0, 1, constant_values=False)
    background = ~padded
    structure = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
    )
    _, component_count = ndimage.label(background, structure=structure)
    # Padding guarantees exactly one exterior background component.
    return max(0, int(component_count) - 1)


def _component_label_fraction(
    component: NDArray[np.int32], provenance: NDArray[np.int32], label: int
) -> float:
    labels = provenance[component[:, 0], component[:, 1]]
    return float(np.count_nonzero(labels == label) / labels.size)


def _focal_components(
    grid: NDArray[np.uint8], provenance: NDArray[np.int32], label: int
) -> list[NDArray[np.int32]]:
    return [
        component
        for component in connected_components_torus(grid > 0, min_size=20)
        if _component_label_fraction(component, provenance, label) >= 0.5
    ]


def _try_immigration(
    world: EvoloopWorld,
    rng: np.random.Generator,
    rate_per_10000: float,
) -> None:
    if rate_per_10000 <= 0:
        return
    arrivals = int(rng.poisson(rate_per_10000 / 10_000.0))
    for _ in range(arrivals):
        for _ in range(128):
            pattern = np.rot90(EVOLOOP_SEED, int(rng.integers(4)))
            row = int(rng.integers(world.grid.shape[0]))
            col = int(rng.integers(world.grid.shape[1]))
            if place_pattern(
                world.grid,
                pattern,
                row,
                col,
                provenance=world.provenance,
                label=world.next_label,
                require_clear_box=True,
            ):
                world.next_label += 1
                break


def evoloop_tick(
    world: EvoloopWorld,
    rule: EvoloopRule,
    rng: np.random.Generator,
    immigration_per_10000: float,
) -> None:
    _try_immigration(world, rng, immigration_per_10000)
    before = world.grid
    after = rule.step(before)
    world.provenance = _passive_provenance_step(before, after, world.provenance)
    world.grid = after


def _nearest_component(
    components: Sequence[NDArray[np.int32]], target: NDArray[np.int32], shape: tuple[int, int]
) -> NDArray[np.int32] | None:
    if not components:
        return None
    target_center = target.mean(axis=0)
    distances: list[float] = []
    for component in components:
        center = component.mean(axis=0)
        delta = np.abs(center - target_center)
        delta = np.minimum(delta, np.asarray(shape) - delta)
        distances.append(float(np.square(delta).sum()))
    return components[int(np.argmin(distances))]


def advance_evoloop_boundary(
    world: EvoloopWorld,
    parameters: EvoloopParameters,
    rule: EvoloopRule,
    rng: np.random.Generator,
    *,
    cap: int,
    persistence: int,
    arm_window: int,
    occupancy_limit: float = 0.25,
) -> tuple[EvoloopWorld, BoundaryTransition]:
    current = world.copy()
    focal = _focal_components(current.grid, current.provenance, current.focal_label)
    if not focal:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return current, BoundaryTransition(empty, empty, 0, False, True, False)
    parent_component = focal[0]
    parent_crop = crop_component(current.grid, parent_component)
    previous_mask = np.zeros_like(current.grid, dtype=bool)
    previous_mask[parent_component[:, 0], parent_component[:, 1]] = True
    elapsed = 0
    while elapsed < cap:
        evoloop_tick(current, rule, rng, parameters.immigration_per_10000)
        elapsed += 1
        if np.count_nonzero(current.grid) > occupancy_limit * current.grid.size:
            empty = np.zeros((0, 0), dtype=np.uint8)
            return current, BoundaryTransition(parent_crop, empty, elapsed, False, False, True)
        if elapsed % 4:
            continue
        focal = _focal_components(current.grid, current.provenance, current.focal_label)
        if not focal:
            empty = np.zeros((0, 0), dtype=np.uint8)
            return current, BoundaryTransition(parent_crop, empty, elapsed, False, True, False)
        if len(focal) == 1:
            parent_component = focal[0]
            parent_crop = crop_component(current.grid, parent_component)
            previous_mask.fill(False)
            previous_mask[parent_component[:, 0], parent_component[:, 1]] = True
            continue
        overlaps = [int(np.count_nonzero(previous_mask[item[:, 0], item[:, 1]])) for item in focal]
        parent_index = int(np.argmax(overlaps))
        child_candidates = [item for index, item in enumerate(focal) if index != parent_index]
        if not child_candidates:
            continue
        eligible_children: list[NDArray[np.int32]] = []
        ambiguous_children: list[NDArray[np.int32]] = []
        for candidate in child_candidates:
            fraction = _component_label_fraction(
                candidate, current.provenance, current.focal_label
            )
            candidate_crop = crop_component(current.grid, candidate)
            if fraction < 0.80:
                ambiguous_children.append(candidate)
            elif _component_holes(candidate_crop) >= 1:
                eligible_children.append(candidate)
        if not eligible_children and ambiguous_children:
            child_crop = crop_component(current.grid, ambiguous_children[0])
            return current, BoundaryTransition(
                parent_crop, child_crop, elapsed, True, False, False
            )
        if not eligible_children:
            continue
        child = eligible_children[0]

        stable = True
        tracked = child
        starting_size = int(child.shape[0])
        arm_launched = False
        for lookahead in range(1, min(persistence + arm_window, cap - elapsed) + 1):
            evoloop_tick(current, rule, rng, parameters.immigration_per_10000)
            elapsed += 1
            if np.count_nonzero(current.grid) > occupancy_limit * current.grid.size:
                empty = np.zeros((0, 0), dtype=np.uint8)
                return current, BoundaryTransition(parent_crop, empty, elapsed, False, False, True)
            candidates = _focal_components(current.grid, current.provenance, current.focal_label)
            nearest = _nearest_component(candidates, tracked, current.grid.shape)
            if nearest is None:
                stable = False
                break
            tracked = nearest
            if _component_label_fraction(
                tracked, current.provenance, current.focal_label
            ) < 0.80:
                return current, BoundaryTransition(
                    parent_crop,
                    crop_component(current.grid, tracked),
                    elapsed,
                    True,
                    False,
                    False,
                )
            if lookahead <= persistence and _component_holes(crop_component(current.grid, tracked)) < 1:
                stable = False
                break
            if lookahead > persistence and tracked.shape[0] >= starting_size + 2:
                arm_launched = True
                break
        if not stable or not arm_launched:
            continue
        child_crop = crop_component(current.grid, tracked)
        new_label = current.next_label
        current.next_label += 1
        current.provenance[tracked[:, 0], tracked[:, 1]] = new_label
        current.focal_label = new_label
        return current, BoundaryTransition(
            parent_crop,
            child_crop,
            elapsed,
            False,
            False,
            False,
        )
    empty = np.zeros((0, 0), dtype=np.uint8)
    return current, BoundaryTransition(parent_crop, empty, cap, False, True, False)


def mechanics_cells(model: str) -> list[ProtocellParameters | EvoloopParameters]:
    if model == "protocell":
        return [
            ProtocellParameters.from_pair(p_y, a_y)
            for p_y, a_y in product(PROTOCELL_PY, PROTOCELL_AY)
        ]
    if model == "evoloop":
        return [
            EvoloopParameters(count, immigration)
            for count, immigration in product(EVOLOOP_COUNTS, EVOLOOP_IMMIGRATION)
        ]
    raise ValueError(f"unknown model: {model}")
