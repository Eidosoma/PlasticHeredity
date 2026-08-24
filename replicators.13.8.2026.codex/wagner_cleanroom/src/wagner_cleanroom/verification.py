from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment import (
    ARM_NAMES,
    CHALLENGE_CODE,
    CONDITION_CODE,
    _simulate_cell,
    expected_rows,
)
from .protocol import PROJECT_ROOT, digest, load_protocol, write_json_atomic
from .storage import load_rulebook, verify_checksums


def cleanroom_violations() -> list[str]:
    forbidden_module = "plastic_" + "heredity"
    forbidden_path = "New" + "Ideas"
    failures: list[str] = []
    source_root = PROJECT_ROOT / "src"
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            failures.append(f"syntax error {path.relative_to(PROJECT_ROOT)}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                if name == forbidden_module or name.startswith(forbidden_module + "."):
                    failures.append(f"forbidden import in {path.relative_to(PROJECT_ROOT)}")
        if forbidden_path in text:
            failures.append(f"forbidden runtime path literal in {path.relative_to(PROJECT_ROOT)}")
    return failures


def validate_environment() -> dict[str, Any]:
    import numpy
    import scipy
    import sklearn

    primary = load_protocol("primary")
    predictor = load_protocol("predictor")
    failures = cleanroom_violations()
    if expected_rows(primary) != 3_194_880:
        failures.append("registered primary record count is not 3,194,880")
    if predictor["development_sources"] != 96 or predictor["evaluation_sources"] != 128:
        failures.append("predictor cohort sizes changed")
    stat = PROJECT_ROOT.stat()
    result = {
        "ok": not failures,
        "failures": failures,
        "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "protocol_digests": {"primary": digest(primary), "predictor": digest(predictor)},
        "project_device": stat.st_dev,
    }
    return result


def verify_run(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    checksum_path = root / "SHA256SUMS"
    if checksum_path.exists():
        failures.extend(verify_checksums(root))
    for registration_path in root.rglob("registration.json"):
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if digest(registration["protocol"]) != registration.get("protocol_digest"):
            failures.append(f"registration digest mismatch: {registration_path.relative_to(root)}")
    for summary_path in root.rglob("simulation_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("complete"):
            failures.append(f"incomplete simulation: {summary_path.parent.relative_to(root)}")
        if "expected_rows" in summary and summary.get("rows") != summary.get("expected_rows"):
            failures.append(f"record mismatch: {summary_path.parent.relative_to(root)}")
    campaign_status_path = root / "STATUS.json"
    if (root / "campaign_registration.json").exists() and campaign_status_path.exists():
        campaign_status = json.loads(campaign_status_path.read_text(encoding="utf-8"))
        if campaign_status.get("complete"):
            replay_path = root / "replay_summary.json"
            if not replay_path.exists():
                failures.append("completed campaign is missing replay_summary.json")
            else:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                if not replay.get("all_exact"):
                    failures.append("one or more registered replay audits failed")
    failures.extend(cleanroom_violations())
    result = {"ok": not failures, "failures": failures}
    write_json_atomic(root / "verification.json", result)
    return result


def format_primary_future_id(
    source: int,
    condition: int,
    arm: int,
    history: int,
    midpoint: int,
    challenge: int,
    age: int,
    future: int,
) -> str:
    return f"P{source:04d}-c{condition}-a{arm}-h{history}-m{midpoint}-k{challenge}-g{age}-f{future:03d}"


def parse_primary_future_id(value: str) -> tuple[int, int, int, int, int, int, int, int]:
    try:
        segments = value.split("-")
        return (
            int(segments[0][1:]), int(segments[1][1:]), int(segments[2][1:]),
            int(segments[3][1:]), int(segments[4][1:]), int(segments[5][1:]),
            int(segments[6][1:]), int(segments[7][1:]),
        )
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid primary future ID: {value}") from error


def replay_primary(root: Path, future_id: str) -> dict[str, Any]:
    source, condition_code, arm_code, history, midpoint, challenge_code, age, future = parse_primary_future_id(future_id)
    run_dir = root / "primary" if (root / "primary").exists() else root
    registration = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
    protocol = registration["protocol"]
    source_path = run_dir / "sources" / f"source_{source:04d}.npz"
    shard_path = run_dir / "shards" / f"source_{source:04d}.npz"
    rulebook = load_rulebook(source_path)
    with np.load(shard_path, allow_pickle=False) as data:
        rows = data["rows"]
        mask = (
            (rows["condition"] == condition_code) & (rows["arm"] == arm_code)
            & (rows["history"] == history) & (rows["midpoint"] == midpoint)
            & (rows["challenge"] == challenge_code) & (rows["age"] == age)
            & (rows["future"] == future)
        )
        retained = rows[mask]
    if len(retained) != 1:
        raise RuntimeError(f"future ID resolves to {len(retained)} retained rows")
    condition_name = next(name for name, code in CONDITION_CODE.items() if code == condition_code)
    challenge_name = next(name for name, code in CHALLENGE_CODE.items() if code == challenge_code)
    condition = next(cell for cell in protocol["conditions"] if cell["name"] == condition_name)
    replayed = _simulate_cell(
        rulebook, protocol, condition_name, ARM_NAMES[arm_code], history, midpoint,
        challenge_name, age, int(condition["futures"]),
    )
    candidate = replayed[replayed["future"] == future]
    exact = bool(np.array_equal(retained, candidate))
    result = {
        "future_id": future_id,
        "exact": exact,
        "retained_digest": int(retained["trajectory_digest"][0]),
        "replayed_digest": int(candidate["trajectory_digest"][0]),
    }
    audit_dir = root / "replay_audits"
    write_json_atomic(audit_dir / f"{future_id}.json", result)
    return result
