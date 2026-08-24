from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/e01/run_s20_b_closeout.py"
SPEC = importlib.util.spec_from_file_location("s20_b_closeout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
S20 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S20)


def test_contract_selects_only_closeout_mode_and_forbids_science() -> None:
    contract = S20.load_contract()
    assert contract["mode"] == "S20_B_CLOSEOUT_ONLY"
    assert contract["scientificOutcomeGenerationAuthorized"] is False
    assert contract["reportBundleGenerationAuthorized"] is False
    assert contract["e02ExecutionAuthorizedInsideE01"] is False
    assert contract["expectedS18StatusCounts"] == {
        "DIRECTIONALLY_SUPPORTED": 17,
        "NOT_EVALUATED": 16,
        "NOT_SUPPORTED_WITHIN_TESTED_SCOPE": 21,
        "SUPPORTED": 3,
        "UNDERDETERMINED": 2,
    }


def test_v3_matrix_addition_preserves_every_s18_field() -> None:
    source = S20.read_csv(S20.S18_MATRIX_A)
    generated, fields = S20.append_v3_fields(source, "matrix_a")
    assert len(generated) == 59
    assert fields[: len(source[0])] == list(source[0])
    for before, after in zip(source, generated, strict=True):
        assert all(after[key] == value for key, value in before.items())
        assert after["s20ConfirmationStatus"] == S20.S20_NA
        assert after["finalV3AddendumStatus"] == before["finalStatusCode"]


def test_confirmation_parquet_is_explicit_zero_row_schema(tmp_path: Path) -> None:
    path = tmp_path / "confirmation.parquet"
    S20.write_empty_parquet(
        path,
        [("matrixId", pa.string()), ("value", pa.float64()), ("status", pa.string())],
    )
    table = pq.read_table(path)
    assert table.num_rows == 0
    assert table.schema.names == ["matrixId", "value", "status"]


def test_handover_and_discovery_keep_claim_boundaries() -> None:
    contract = S20.load_contract()
    handover = S20.handover_notes(contract)
    discovery = S20.discovery_report(contract)
    for phrase in [
        "candidate replacement causal-architecture variable",
        "not PhiID",
        "not a replication of the paper",
        "not causal control",
    ]:
        assert phrase.lower() in handover.lower()
    for phrase in [
        "plastic hereditary-regime switching",
        "failed Phi/PhiID replications",
        "does not reproduce the paper",
        "40 new shared catalytic matrices",
        "25,600 branch futures",
    ]:
        assert phrase.lower() in discovery.lower()


def test_l54_contract_matches_frozen_classification() -> None:
    contract = S20.load_contract()
    frozen = json.loads((S20.L54_ROOT / "classification.json").read_text())
    assert set(contract["requiredL54Classifications"]) == set(
        frozen["classifications"]
    )
    assert frozen["phiComputed"] is False
    assert frozen["paperReplicationClaim"] is False
    assert frozen["interventionRun"] is False


def test_e02_authorization_is_released_only_after_s20_validation_and_hashing(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.json"
    preauthorization = tmp_path / "preauthorization.json"
    S20.write_json(validation, {"validationResult": "PASS"})
    S20.write_json(preauthorization, {"aggregateSha256": "frozen-core"})
    payload = S20.e02_authorization_payload(
        S20.load_contract(), validation, preauthorization
    )
    assert payload["authorizationEffectiveAfterS20ValidationAndHashing"] is True
    assert payload["s20ValidationResult"] == "PASS"
    assert payload["s20ValidationSha256"] == S20.sha256_file(validation)
    assert payload["preauthorizationManifestSha256"] == S20.sha256_file(
        preauthorization
    )
    assert payload["e02ExecutedInE01"] is False
