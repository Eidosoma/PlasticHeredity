from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _semantic_bytes(master: str, domain: str, coordinates: tuple[Any, ...]) -> bytes:
    value = [master, domain, *coordinates]
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=False).encode("utf-8")


def stable_digest(master: str, domain: str, *coordinates: Any) -> str:
    return hashlib.sha256(_semantic_bytes(master, domain, coordinates)).hexdigest()


def stable_seed(master: str, domain: str, *coordinates: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(_semantic_bytes(master, domain, coordinates)).digest()[:8],
        "little",
        signed=False,
    )


def generator(master: str, domain: str, *coordinates: Any) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(stable_seed(master, domain, *coordinates)))


def jax_key(master: str, domain: str, *coordinates: Any):
    import jax

    seed = stable_seed(master, domain, *coordinates)
    key = jax.random.PRNGKey(np.uint32(seed & 0xFFFFFFFF))
    return jax.random.fold_in(key, np.uint32(seed >> 32))


def array_digest(*arrays: np.ndarray, decimals: int | None = None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        if decimals is not None and np.issubdtype(value.dtype, np.floating):
            value = np.round(value.astype(np.float64), decimals=decimals)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()

