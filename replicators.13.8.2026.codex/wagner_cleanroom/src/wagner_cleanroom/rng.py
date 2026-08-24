from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _payload(master_label: str, coordinates: tuple[Any, ...]) -> bytes:
    return json.dumps(
        [master_label, *coordinates], separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def seed_int(master_label: str, *coordinates: Any) -> int:
    raw = hashlib.sha256(_payload(master_label, coordinates)).digest()
    return int.from_bytes(raw[:16], "little", signed=False)


def generator(master_label: str, *coordinates: Any) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(seed_int(master_label, *coordinates)))


def future_id(cohort: str, *coordinates: Any) -> str:
    raw = hashlib.sha256(_payload(cohort, coordinates)).hexdigest()
    return f"{cohort[:8]}-{raw[:24]}"


def digest_u64(values: np.ndarray) -> np.uint64:
    raw = hashlib.blake2b(np.ascontiguousarray(values).view(np.uint8), digest_size=8).digest()
    return np.uint64(int.from_bytes(raw, "little"))

