from __future__ import annotations

import hashlib


def derive_seed(master_seed: str, domain: str, *parts: object) -> int:
    """Derive an order-independent 256-bit seed for one stochastic domain."""

    material = "|".join((master_seed, domain, *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")

