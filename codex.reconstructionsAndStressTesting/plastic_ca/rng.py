"""Domain-separated deterministic random streams."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable


def derive_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def stream(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))


def fixed_density_bits(namespace: str, index: int, width: int, density: float = 0.5) -> int:
    """Return a reproducible bit row with an exact rounded density."""

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    count = min(width, max(0, round(width * density)))
    positions = stream(namespace, "launch-row", index, width, density).sample(range(width), count)
    value = 0
    for position in positions:
        value |= 1 << position
    return value


def hash_bits(namespace: str, index: int, width: int) -> int:
    """Return raw deterministic hash bits, with expected rather than forced 1/2 density."""

    byte_count = (width + 7) // 8
    payload = f"{namespace}\x1fhash-row\x1f{index}\x1f{width}".encode("utf-8")
    value = int.from_bytes(hashlib.shake_256(payload).digest(byte_count), "little")
    value &= (1 << width) - 1
    if value == 0:
        value = 1
    elif value == (1 << width) - 1:
        value ^= 1
    return value


def bernoulli_mask(rng: random.Random, probability: float, width: int) -> int:
    """Draw independent Bernoulli bits using exact geometric skip sampling.

    For sparse probabilities this costs approximately the number of set bits,
    rather than ``width`` Python-level random draws.
    """

    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return (1 << width) - 1
    log_failure = math.log1p(-probability)
    position = -1
    result = 0
    while True:
        # log(1-U) / log(1-p) is the number of failures before a success.
        gap = int(math.log1p(-rng.random()) / log_failure)
        position += gap + 1
        if position >= width:
            return result
        result |= 1 << position


def stable_sample(items: Iterable[int], count: int, *parts: object) -> list[int]:
    values = list(items)
    return stream(*parts).sample(values, min(count, len(values)))
