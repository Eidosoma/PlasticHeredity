from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from e01_s13r_schema_normalization.core import (
    ADAPTER_FIELD_TYPES,
    EXPECTED_ADAPTED_TASK_IDS,
    VERSION,
    array_value_sha256,
    normalize_label_table,
    null_mask_sha256,
    table_logical_sha256,
    table_schema_signature,
)

REPO = Path(__file__).resolve().parents[2]


def _table(*, null_typed: bool) -> pa.Table:
    fields = [
        pa.field("candidateId", pa.string()),
        pa.field("generation", pa.int64()),
        pa.field("clusterId", pa.null() if null_typed else pa.string()),
        pa.field("referenceObservationId", pa.null() if null_typed else pa.string()),
        pa.field("metricToReference", pa.null() if null_typed else pa.float64()),
        pa.field("isReplicator", pa.bool_()),
    ]
    schema = pa.schema(fields)
    values = [
        pa.array(["C2", "C2"], type=pa.string()),
        pa.array([1, 2], type=pa.int64()),
        pa.nulls(2, type=fields[2].type),
        pa.nulls(2, type=fields[3].type),
        pa.nulls(2, type=fields[4].type),
        pa.array([False, True], type=pa.bool_()),
    ]
    return pa.Table.from_arrays(values, schema=schema)


def test_preregistration_freezes_exactly_one_schema_repair() -> None:
    config = yaml.safe_load(
        (
            REPO
            / "configs/e01/s13r_schema_normalization_confirmation_preregistration.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["versionedStepId"] == VERSION
    assert config["inputs"]["taskCount"] == 200
    assert config["adapter"]["expectedCanonicalTaskCount"] == 195
    assert config["adapter"]["expectedAdaptedTaskCount"] == 5
    assert set(config["adapter"]["exactAffectedTasks"]) == EXPECTED_ADAPTED_TASK_IDS
    assert set(config["adapter"]["exactPermittedFields"]) == set(ADAPTER_FIELD_TYPES)
    assert config["strictOneRepairRule"]["furtherSchemaChangePermitted"] is False
    assert config["immutability"]["partialS13ConcatenationUsePermitted"] is False


def test_null_typed_optional_fields_gain_only_declared_physical_types() -> None:
    source = _table(null_typed=True)
    logical_before = table_logical_sha256(source)
    masks_before = {name: null_mask_sha256(source[name]) for name in source.column_names}
    normalized, adapted = normalize_label_table(source)
    assert adapted == tuple(ADAPTER_FIELD_TYPES)
    assert normalized.column_names == source.column_names
    assert normalized.num_rows == source.num_rows
    assert table_logical_sha256(normalized) == logical_before
    assert {name: null_mask_sha256(normalized[name]) for name in normalized.column_names} == masks_before
    for name, target in ADAPTER_FIELD_TYPES.items():
        assert normalized.schema.field(name).type.equals(target)
        assert normalized[name].null_count == normalized.num_rows
    for name in set(source.column_names) - set(ADAPTER_FIELD_TYPES):
        assert normalized[name].equals(source[name])
        assert normalized.schema.field(name).type.equals(source.schema.field(name).type)


def test_canonical_typed_table_is_value_identical_and_unadapted() -> None:
    source = _table(null_typed=False)
    normalized, adapted = normalize_label_table(source)
    assert adapted == ()
    assert normalized.equals(source)
    assert table_schema_signature(normalized) == table_schema_signature(source)
    for name in source.column_names:
        assert array_value_sha256(normalized[name]) == array_value_sha256(source[name])


def test_adapter_rejects_any_noncanonical_physical_type() -> None:
    source = _table(null_typed=False)
    index = source.schema.get_field_index("metricToReference")
    forbidden = source.set_column(
        index,
        pa.field("metricToReference", pa.string()),
        pa.array([None, None], type=pa.string()),
    )
    with pytest.raises(TypeError, match="forbidden source type"):
        normalize_label_table(forbidden)


def test_adapter_rejects_missing_declared_field() -> None:
    source = _table(null_typed=True).drop(["clusterId"])
    with pytest.raises(ValueError, match="missing authorized label fields"):
        normalize_label_table(source)
