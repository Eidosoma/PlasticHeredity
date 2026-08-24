#!/usr/bin/env python3
"""Value-preserving exact-regeneration repair for E01/S19-L10.

Human authorization was received after the initial regeneration compared a
schedule-dependent DataFrame column order.  This additive repair preserves the
failed attempt, changes no scientific implementation or value, reruns all 400
trajectories and all registered tables, and compares tables after a fixed
lexicographic column canonicalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.e01 import run_s19_l10 as locked

LOOP_ROOT = Path("/artifacts/research_steps/S19/loops/L10")
REPAIR_CACHE = Path("/cache/e01_s19_l10/regeneration_repair_001")
REPAIR_OUTPUT = Path("/cache/e01_s19_l10/regenerated_outputs_repair_001")
REPAIR_ID = "S19-L10-TECHNICAL-REPAIR-001"
REPAIR_SCRIPT = Path(__file__).resolve()
SCIENTIFIC_COMMIT = "e257cad4263ee63d733c37f041ee6994eeb7e385"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            locked.json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def canonical_frame_sha256_fixed(
    frame: pd.DataFrame, sort_columns: tuple[str, ...]
) -> str:
    """Canonicalize both row order and schema order before exact hashing."""

    row_keys = [column for column in sort_columns if column in frame.columns]
    ordered = (
        frame.sort_values(row_keys, kind="stable").reset_index(drop=True).copy()
        if row_keys
        else frame.reset_index(drop=True).copy()
    )
    ordered = ordered.reindex(columns=sorted(ordered.columns))
    for column in ordered.columns:
        if ordered[column].dtype == object:
            ordered[column] = ordered[column].map(
                lambda value: (
                    json.dumps(
                        locked.json_safe(value), sort_keys=True, separators=(",", ":")
                    )
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                )
            )
    payload = ordered.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def initial_failure_diagnostic() -> dict[str, Any]:
    validation = json.loads(
        (LOOP_ROOT / "regeneration_validation.json").read_text(encoding="utf-8")
    )
    comparisons = pd.read_csv(LOOP_ROOT / "result_regeneration_results.csv")
    failed = comparisons.loc[~comparisons.passed.astype(bool)]
    primary = read_table(LOOP_ROOT / "label_fingerprint_results.parquet")
    replay = read_table(
        Path("/cache/e01_s19_l10/regenerated_outputs/label_fingerprint_results.parquet")
    )
    keys = ["pipelineId", "candidateId", "matrixIndex"]
    primary = primary.sort_values(keys, kind="stable").reset_index(drop=True)
    replay = replay.sort_values(keys, kind="stable").reset_index(drop=True)
    same_columns = set(primary.columns) == set(replay.columns)
    aligned = replay.reindex(columns=primary.columns) if same_columns else replay
    same = (
        primary.eq(aligned) | (primary.isna() & aligned.isna())
        if same_columns
        else None
    )
    cell_differences = int((~same).to_numpy().sum()) if same is not None else None
    return {
        "initialValidation": validation,
        "failedTableCount": len(failed),
        "failedTables": failed.artifact.astype(str).tolist(),
        "columnSetsEqual": same_columns,
        "columnOrderExact": list(primary.columns) == list(replay.columns),
        "cellDifferencesAfterColumnAlignment": cell_differences,
        "cellValuesExactAfterColumnAlignment": cell_differences == 0,
    }


def preserve_initial_failure() -> None:
    copies = {
        "regeneration_validation.json": "regeneration_validation_failed_attempt_001.json",
        "result_regeneration_results.csv": "result_regeneration_results_failed_attempt_001.csv",
        "trajectory_regeneration_results.parquet": "trajectory_regeneration_results_failed_attempt_001.parquet",
        "regeneration_runtime.json": "regeneration_runtime_failed_attempt_001.json",
    }
    for source_name, destination_name in copies.items():
        destination = LOOP_ROOT / destination_name
        if destination.exists():
            raise RuntimeError(
                f"preserved repair evidence already exists: {destination}"
            )
        shutil.copy2(LOOP_ROOT / source_name, destination)


def prepare() -> None:
    if (LOOP_ROOT / "technical_repair_lock_001.json").exists():
        raise RuntimeError("technical repair lock already exists")
    diagnostic = initial_failure_diagnostic()
    if not (
        diagnostic["initialValidation"]["trajectoryReplayPassCount"] == 400
        and diagnostic["initialValidation"]["scientificTablePassCount"] == 13
        and diagnostic["failedTables"] == ["label_fingerprint_results.parquet"]
        and diagnostic["columnSetsEqual"]
        and not diagnostic["columnOrderExact"]
        and diagnostic["cellValuesExactAfterColumnAlignment"]
    ):
        raise RuntimeError(
            "observed failure is not the authorized technical column-order defect"
        )
    preserve_initial_failure()
    write_json(
        LOOP_ROOT / "technical_repair_001.json",
        {
            "schema": "eidosoma.e01.s19_l10.technical_repair.v1",
            "repairId": REPAIR_ID,
            "humanAuthorization": "If it was a technical problem, fix and rerun.",
            "failureObserved": diagnostic,
            "repair": "Rerun complete trajectory and table regeneration and canonicalize table columns lexicographically before exact hashing.",
            "scientificCodeChanged": False,
            "scientificMethodChanged": False,
            "scientificValueChanged": False,
            "thresholdSeedPipelineLabelControlOrStatisticChanged": False,
            "initialFailurePreserved": True,
            "newRepairCacheRequired": True,
            "recordedAtUtc": utc_now(),
        },
    )
    (LOOP_ROOT / "technical_repair_001.md").write_text(
        f"""# S19-L10 technical repair 001

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** additive technical repair locked before rerun; repaired outcome pending.
- **Artifacts written:** preserved initial regeneration validation/table/trajectory/runtime evidence, repair decision JSON, and this repair note.
- **Validation result:** the initial 400/400 trajectory replay passed; 13/14 table hashes passed; the sole failed table had the same columns and zero different cells after diagnostic alignment but a different column order.
- **Outcome classification:** pending complete repaired regeneration; no scientific classification is changed by this lock.
- **Caveats or blockers:** the repair is post-outcome but explicitly human-authorized; it may canonicalize schema order only and may not change any scientific code, method, value, seed, or gate.
- **Recommended next action:** commit and push this repair, verify the clean release gate, rerun all 400 trajectories and all tables in a fresh cache, then accept results only if every exact value/schema gate passes.

The initial failure is preserved under `*_failed_attempt_001.*`. Repair 001 fixes only the omission of column-order canonicalization in the exact table comparator. Lexicographic column order is deterministic and independent of scientific values. The original scientific runner/core/config at commit `{SCIENTIFIC_COMMIT}` remain unchanged.
""",
        encoding="utf-8",
    )
    scientific_lock = json.loads(
        (LOOP_ROOT / "implementation_lock.json").read_text(encoding="utf-8")
    )
    scientific_hashes = {row["path"]: row["sha256"] for row in scientific_lock["code"]}
    if not all(
        sha256_file(REPO / path) == digest for path, digest in scientific_hashes.items()
    ):
        raise RuntimeError("original scientific lock changed before technical repair")
    write_json(
        LOOP_ROOT / "technical_repair_lock_001.json",
        {
            "schema": "eidosoma.e01.s19_l10.technical_repair_lock.v1",
            "repairId": REPAIR_ID,
            "lockedAtUtc": utc_now(),
            "outcomeAccessedBeforeRepair": True,
            "humanAuthorized": True,
            "scientificCommit": SCIENTIFIC_COMMIT,
            "scientificCodeHashes": scientific_hashes,
            "repairScriptPath": str(REPAIR_SCRIPT.relative_to(REPO)),
            "repairScriptSha256": sha256_file(REPAIR_SCRIPT),
            "fixedSchemaCanonicalization": "LEXICOGRAPHIC_COLUMN_ORDER_THEN_LOCKED_ROW_ORDER",
            "freshRepairCache": str(REPAIR_CACHE),
            "freshRepairOutput": str(REPAIR_OUTPUT),
            "rerunScope": "ALL_100_MATRICES_ALL_400_TRAJECTORIES_ALL_14_AUTHORITATIVE_TABLES",
            "repairCount": 1,
            "anotherRepairPermitted": False,
            "headBeforeRepairCommit": git("rev-parse", "HEAD"),
        },
    )
    print(
        json.dumps(
            {"status": "TECHNICAL_REPAIR_001_LOCKED_PENDING_COMMIT_PUSH", **diagnostic},
            sort_keys=True,
        )
    )


def release_gate() -> dict[str, Any]:
    lock = json.loads(
        (LOOP_ROOT / "technical_repair_lock_001.json").read_text(encoding="utf-8")
    )
    scientific_hashes = lock["scientificCodeHashes"]
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    result = {
        "schema": "eidosoma.e01.s19_l10.technical_repair_release_gate.v1",
        "head": head,
        "remoteHead": remote,
        "branch": git("branch", "--show-current"),
        "cleanWorktree": not bool(git("status", "--porcelain=v1")),
        "repairScriptHashMatches": sha256_file(REPAIR_SCRIPT)
        == lock["repairScriptSha256"],
        "scientificCodeHashesMatch": all(
            sha256_file(REPO / path) == digest
            for path, digest in scientific_hashes.items()
        ),
        "originalScientificCommitIsAncestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SCIENTIFIC_COMMIT, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "validatedAtUtc": utc_now(),
    }
    result["passed"] = bool(
        result["head"] == result["remoteHead"]
        and result["branch"] == "eidosoma/groups/42"
        and result["cleanWorktree"]
        and result["repairScriptHashMatches"]
        and result["scientificCodeHashesMatch"]
        and result["originalScientificCommitIsAncestor"]
    )
    write_json(LOOP_ROOT / "technical_repair_release_gate_001.json", result)
    return result


def compare_trajectories(replay: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    keys = ["matrixIndex", "groupId", "candidateId"]
    fields = [
        "trajectoryId",
        "trajectorySha256",
        "betaSha256",
        "initialStateSha256",
        "terminalStatus",
        "completedFissions",
        "selectedClockLength",
        "postFissionBoundaryCount",
        "cacheSha256",
    ]
    merged = primary.merge(
        replay, on=keys, suffixes=("Primary", "Replay"), validate="one_to_one"
    )
    rows = []
    for item in merged.itertuples():
        exact = {
            field: getattr(item, f"{field}Primary") == getattr(item, f"{field}Replay")
            for field in fields
        }
        rows.append(
            {
                "matrixIndex": int(item.matrixIndex),
                "groupId": item.groupId,
                "candidateId": item.candidateId,
                **{f"{field}Exact": value for field, value in exact.items()},
                "passed": bool(all(exact.values())),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def compare_tables() -> pd.DataFrame:
    rows = []
    for filename, keys in locked.CORE_TABLES.items():
        primary = read_table(LOOP_ROOT / filename)
        replay = read_table(REPAIR_OUTPUT / filename)
        column_sets = set(primary.columns) == set(replay.columns)
        replay_aligned = (
            replay.reindex(columns=primary.columns) if column_sets else replay
        )
        row_keys = [column for column in keys if column in primary.columns]
        primary_ordered = primary.sort_values(row_keys, kind="stable").reset_index(
            drop=True
        )
        replay_ordered = replay_aligned.sort_values(
            row_keys, kind="stable"
        ).reset_index(drop=True)
        dtypes_exact = bool(
            column_sets
            and [str(value) for value in primary_ordered.dtypes]
            == [str(value) for value in replay_ordered.dtypes]
        )
        values_exact = bool(column_sets and primary_ordered.equals(replay_ordered))
        primary_hash = canonical_frame_sha256_fixed(primary, keys)
        replay_hash = canonical_frame_sha256_fixed(replay, keys)
        rows.append(
            {
                "artifact": filename,
                "primaryRows": len(primary),
                "replayRows": len(replay),
                "primaryCanonicalSha256": primary_hash,
                "replayCanonicalSha256": replay_hash,
                "columnSetsExact": column_sets,
                "rawColumnOrderExact": list(primary.columns) == list(replay.columns),
                "dtypesExactAfterAlignment": dtypes_exact,
                "cellValuesExactAfterAlignment": values_exact,
                "passed": bool(
                    len(primary) == len(replay)
                    and column_sets
                    and dtypes_exact
                    and values_exact
                    and primary_hash == replay_hash
                ),
            }
        )
    return pd.DataFrame(rows)


def run(workers: int) -> None:
    if workers != 8:
        raise ValueError("technical repair is locked to eight workers")
    gate = release_gate()
    if not gate["passed"]:
        raise RuntimeError(f"technical repair release gate failed: {gate}")
    if REPAIR_CACHE.exists() and any(REPAIR_CACHE.rglob("*")):
        raise RuntimeError("fresh technical-repair trajectory cache is not empty")
    if REPAIR_OUTPUT.exists() and any(REPAIR_OUTPUT.rglob("*")):
        raise RuntimeError("fresh technical-repair output cache is not empty")
    REPAIR_CACHE.mkdir(parents=True, exist_ok=True)
    REPAIR_OUTPUT.mkdir(parents=True, exist_ok=True)

    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    outputs = locked.run_simulation_batch(range(100), workers, REPAIR_CACHE)
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    replay = pd.DataFrame([row for output in outputs for row in output["trajectories"]])
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    if len(attempts) != 400 or len(replay) != 400 or len(failures):
        raise RuntimeError("technical repair trajectory rerun failed")
    trajectory_results = compare_trajectories(replay)
    if len(trajectory_results) != 400 or not trajectory_results.passed.all():
        raise RuntimeError("technical repair trajectory replay failed")

    frames = locked.build_scientific_outputs(REPAIR_CACHE, workers)
    if len(frames["failure"]):
        raise RuntimeError("technical repair scientific rerun raised an exception")
    locked.write_scientific_outputs(REPAIR_OUTPUT, frames)
    table_results = compare_tables()
    trajectory_results.to_parquet(
        LOOP_ROOT / "trajectory_regeneration_results.parquet",
        index=False,
        compression="zstd",
    )
    table_results.to_csv(
        LOOP_ROOT / "result_regeneration_results.csv", index=False, lineterminator="\n"
    )

    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime
        + child_after.ru_stime
        - child_before.ru_utime
        - child_before.ru_stime
    )
    repair_runtime = {
        "schema": "eidosoma.e01.s19_l10.technical_repair_regeneration_runtime.v1",
        "repairId": REPAIR_ID,
        "startedAtUtc": started,
        "completedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - wall_start,
        "coordinatorCpuSeconds": time.process_time() - cpu_start,
        "workerCpuSeconds": float(attempts.cpuSeconds.sum()),
        "childCpuSeconds": child_cpu,
        "workers": workers,
    }
    write_json(LOOP_ROOT / "regeneration_repair_runtime_001.json", repair_runtime)
    original_runtime = json.loads(
        (LOOP_ROOT / "regeneration_runtime_failed_attempt_001.json").read_text(
            encoding="utf-8"
        )
    )
    combined_runtime = {
        "schema": "eidosoma.e01.s19_l10.regeneration_runtime.v2_combined_attempts",
        "startedAtUtc": original_runtime["startedAtUtc"],
        "completedAtUtc": repair_runtime["completedAtUtc"],
        "wallSeconds": float(original_runtime["wallSeconds"])
        + float(repair_runtime["wallSeconds"]),
        "coordinatorCpuSeconds": float(original_runtime["coordinatorCpuSeconds"])
        + float(repair_runtime["coordinatorCpuSeconds"]),
        "workerCpuSeconds": float(original_runtime["workerCpuSeconds"])
        + float(repair_runtime["workerCpuSeconds"]),
        "childCpuSeconds": float(original_runtime["childCpuSeconds"])
        + float(repair_runtime["childCpuSeconds"]),
        "workers": workers,
        "attemptCount": 2,
        "failedAttemptPreserved": True,
        "repairId": REPAIR_ID,
    }
    write_json(LOOP_ROOT / "regeneration_runtime.json", combined_runtime)
    validation = {
        "schema": "eidosoma.e01.s19_l10.regeneration_validation.v2_technical_repair",
        "repairId": REPAIR_ID,
        "initialFailedAttemptPreserved": True,
        "initialFailure": "SCHEDULE_DEPENDENT_COLUMN_ORDER_IN_FINGERPRINT_TABLE",
        "fixedCanonicalization": "LEXICOGRAPHIC_COLUMN_ORDER_THEN_LOCKED_ROW_ORDER",
        "trajectoryReplayRows": len(trajectory_results),
        "trajectoryReplayPassCount": int(trajectory_results.passed.sum()),
        "scientificTableCount": len(table_results),
        "scientificTablePassCount": int(table_results.passed.sum()),
        "all400TrajectoriesExact": bool(
            len(trajectory_results) == 400 and trajectory_results.passed.all()
        ),
        "allScientificTablesExact": bool(
            len(table_results) == len(locked.CORE_TABLES) and table_results.passed.all()
        ),
        "scientificValueChangePermitted": False,
        "scientificValueChangeObserved": False,
        "passed": bool(
            len(trajectory_results) == 400
            and trajectory_results.passed.all()
            and len(table_results) == len(locked.CORE_TABLES)
            and table_results.passed.all()
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(
            "technical repair exact regeneration failed; no second repair permitted"
        )
    print(
        json.dumps(
            {"status": "TECHNICAL_REPAIR_001_COMPLETE", **validation}, sort_keys=True
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
