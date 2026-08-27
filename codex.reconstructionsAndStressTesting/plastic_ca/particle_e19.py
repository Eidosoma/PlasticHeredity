"""Figure/ground observer on the corrected E19 batch lifecycle.

The trajectory engine is exact.  The code-free evidence did not disclose the
four particle-domain launch rows, so the dictionary recipe remains an explicit
clean-room choice and this stage is assessed at gate level rather than as a
bit-exact numerical reproduction.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean
import sys
import time
from typing import Any, Sequence

import numpy as np

from .e19 import E19Contract, e19_step, evaluate_e19_rule, require_pinned_numpy, spacetime_codes
from .eca import CLASS_3_CORE, CLASS_4, RAW_CHAMPIONS
from .particle import PARTICLE_GATE_RULES
from .rng import fixed_density_bits


DOMAIN_NAMESPACE = "plastic-ca-e19-particle-domain-v1"


@dataclass(frozen=True)
class E19DomainDictionary:
    codes: frozenset[int]
    coverage: float
    n_distinct_codes: int
    counts: tuple[tuple[int, int], ...]
    seed_rows_hex: tuple[str, ...]


def _position_bits(value: int, width: int = 64) -> np.ndarray:
    return np.array([bool((value >> index) & 1) for index in range(width)], dtype=np.bool_)


def build_e19_domain_dictionary(
    rule: int,
    *,
    n_seeds: int = 4,
    burnin: int = 64,
    collect: int = 16,
    coverage_target: float = 0.9,
    cap: int = 64,
) -> E19DomainDictionary:
    counts: Counter[int] = Counter()
    seed_rows: list[str] = []
    densities = (0.2, 0.4, 0.6, 0.8)
    for seed_index in range(n_seeds):
        integer = fixed_density_bits(
            DOMAIN_NAMESPACE,
            seed_index,
            64,
            densities[seed_index % len(densities)],
        )
        seed_rows.append(f"{integer:016x}")
        row = _position_bits(integer)[None, :]
        history = [row.copy(), row.copy(), row.copy()]
        for _ in range(burnin):
            row = e19_step(row, rule)
            history = [history[1], history[2], row.copy()]
        for _ in range(collect):
            row = e19_step(row, rule)
            history = [history[1], history[2], row.copy()]
            values, frequencies = np.unique(
                spacetime_codes((history[0], history[1], history[2])),
                return_counts=True,
            )
            counts.update({int(value): int(count) for value, count in zip(values, frequencies, strict=True)})
    total = sum(counts.values())
    selected: list[int] = []
    selected_mass = 0
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= cap:
            break
        selected.append(code)
        selected_mass += count
        if total and selected_mass / total >= coverage_target:
            break
    return E19DomainDictionary(
        codes=frozenset(selected),
        coverage=selected_mass / total if total else 0.0,
        n_distinct_codes=len(counts),
        counts=tuple(sorted(counts.items())),
        seed_rows_hex=tuple(seed_rows),
    )


def evaluate_e19_particle_rule(rule: int, contract: E19Contract) -> dict[str, Any]:
    dictionary = build_e19_domain_dictionary(rule)
    result = evaluate_e19_rule(rule, contract, domain_codes=dictionary.codes)
    return {
        "rule": rule,
        "wolfram_class": result.wolfram_class,
        "strict": result.strict,
        "break_by_8": result.break_by_8,
        "mean_survival": result.mean_survival,
        "median_gen_sweeps": result.median_gen_sweeps,
        "dict_coverage": dictionary.coverage,
        "dict_dict_size": len(dictionary.codes),
        "dict_n_distinct_codes": dictionary.n_distinct_codes,
        "dict_codes": sorted(dictionary.codes),
        "dict_seed_rows_hex": list(dictionary.seed_rows_hex),
        "n_futures": result.n_futures,
        "death_counts": result.death_counts,
    }


def _task(arguments: tuple[int, E19Contract]) -> dict[str, Any]:
    return evaluate_e19_particle_rule(*arguments)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_e19_particle(
    contract: E19Contract,
    output: Path,
    *,
    rules: Sequence[int] = PARTICLE_GATE_RULES,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    design = hashlib.sha256(
        json.dumps(
            {
                "contract": contract.to_dict(),
                "domain_namespace": DOMAIN_NAMESPACE,
                "domain": {"seeds": 4, "burnin": 64, "collect": 16, "coverage": 0.9, "cap": 64},
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    rows: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for rule in rules:
        path = checkpoints / f"rule-{rule:03d}.json"
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("design_digest") == design:
                rows[rule] = payload["result"]
                continue
        missing.append(rule)

    def save(row: dict[str, Any]) -> None:
        rule = int(row["rule"])
        rows[rule] = row
        _atomic_json(
            checkpoints / f"rule-{rule:03d}.json",
            {"design_digest": design, "result": row},
        )

    if workers <= 1:
        for rule in missing:
            save(evaluate_e19_particle_rule(rule, contract))
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, (rule, contract)): rule for rule in missing}
            for future in as_completed(futures):
                save(future.result())

    ordered = [rows[rule] for rule in sorted(rules)]
    by_rule = {int(row["rule"]): row for row in ordered}
    class4 = {str(rule): float(by_rule[rule]["strict"]) for rule in sorted(CLASS_4) if rule in by_rule}
    core = {str(rule): float(by_rule[rule]["strict"]) for rule in sorted(CLASS_3_CORE) if rule in by_rule}
    champions = {str(rule): float(by_rule[rule]["strict"]) for rule in RAW_CHAMPIONS if rule in by_rule}
    gates = {
        "gate_redemption_110": 110 in by_rule and 0.005 <= float(by_rule[110]["strict"]) <= 0.5,
        "gate_chaos_stays_chaos": bool(core) and all(value < 0.005 for value in core.values()),
        "gate_champions_stable": len(champions) == len(RAW_CHAMPIONS) and all(value >= 0.005 for value in champions.values()),
        "class4_strict": class4,
        "champions_strict": champions,
        "core3_strict": core,
        "core3_strict_max": max(core.values()) if core else None,
        "dict_coverage_by_class": {
            str(cls): mean(float(row["dict_coverage"]) for row in ordered if int(row["wolfram_class"]) == cls)
            for cls in (1, 2, 3, 4)
            if any(int(row["wolfram_class"]) == cls for row in ordered)
        },
    }
    summary = {
        "experiment": "e19_particle_observer",
        "elapsed_seconds": time.time() - started,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "contract_digest": contract.digest,
        "design_digest": design,
        "dictionary_recipe": {
            "status": "clean-room choice; four reference hash rows were not disclosed",
            "namespace": DOMAIN_NAMESPACE,
            "densities": [0.2, 0.4, 0.6, 0.8],
            "burnin": 64,
            "collect": 16,
            "coverage_target": 0.9,
            "cap": 64,
        },
        "numeric_claim": "gate-level only",
        "gates": gates,
        "rows": ordered,
    }
    _atomic_json(output / "particle_gates.json", summary)
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary
