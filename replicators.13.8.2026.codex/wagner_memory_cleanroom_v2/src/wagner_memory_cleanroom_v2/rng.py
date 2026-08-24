from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def semantic_bytes(master_seed: str, *coordinates: Any) -> bytes:
    return sha256("\x1f".join([master_seed, *(str(item) for item in coordinates)]).encode()).digest()


def seed64(master_seed: str, *coordinates: Any) -> int:
    return int.from_bytes(semantic_bytes(master_seed, *coordinates)[:8], "little")


def generator(master_seed: str, *coordinates: Any) -> np.random.Generator:
    return np.random.default_rng(seed64(master_seed, *coordinates))


def jax_key_data(master_seed: str, *coordinates: Any) -> np.ndarray:
    return np.frombuffer(semantic_bytes(master_seed, *coordinates)[:8], dtype=np.uint32).copy()


def stable_permutation(size: int, master_seed: str, *coordinates: Any) -> np.ndarray:
    return generator(master_seed, *coordinates).permutation(size).astype(np.int16)


def stable_derangement(size: int, master_seed: str, *coordinates: Any) -> np.ndarray:
    if size < 2:
        raise ValueError("a derangement requires at least two entries")
    rng = generator(master_seed, "derangement", *coordinates)
    identity = np.arange(size)
    for _ in range(10_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate.astype(np.int16)
    raise RuntimeError("deterministic derangement guard exhausted")
