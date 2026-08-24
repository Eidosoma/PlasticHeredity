from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def semantic_bytes(master_seed: str, *coordinates: Any) -> bytes:
    parts = [master_seed, *(str(item) for item in coordinates)]
    return sha256("\x1f".join(parts).encode()).digest()


def seed64(master_seed: str, *coordinates: Any) -> int:
    return int.from_bytes(semantic_bytes(master_seed, *coordinates)[:8], "little")


def generator(master_seed: str, *coordinates: Any) -> np.random.Generator:
    return np.random.default_rng(seed64(master_seed, *coordinates))


def jax_key_data(master_seed: str, *coordinates: Any) -> np.ndarray:
    raw = semantic_bytes(master_seed, *coordinates)
    return np.frombuffer(raw[:8], dtype=np.uint32).copy()


def stable_permutation(size: int, master_seed: str, *coordinates: Any) -> np.ndarray:
    return generator(master_seed, *coordinates).permutation(size).astype(np.int16)

