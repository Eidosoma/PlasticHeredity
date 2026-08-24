from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from e01_s13rr_downstream_schema_canonicalization.core import (
    DOWNSTREAM_AFFECTED_TASKS,
    PREFIX_FIELD_TYPES,
    RESEARCH_STEP_ID,
    SEED_FIELD_TYPES,
    TABLE_FAMILIES,
    VERSION,
    canonicalize_table,
    schema_signature,
)

REPO = Path(__file__).resolve().parents[2]
TASK = "S12F-CANDIDATE-02/M72"


def _prefix(*, null_typed: bool) -> pa.Table:
    fields = [pa.field("candidateId", pa.string()), pa.field("matrixIndex", pa.int64())]
    fields += [
        pa.field(name, pa.null() if null_typed else data_type)
        for name, data_type in PREFIX_FIELD_TYPES.items()
    ]
    arrays = [pa.array(["S12F-CANDIDATE-02"]), pa.array([72], type=pa.int64())]
    arrays += [pa.nulls(1, type=field.type) for field in fields[2:]]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _seeds(*, null_typed: bool) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("candidateId", pa.string()),
            pa.field("matrixIndex", pa.int64()),
            pa.field("endpointGeneration", pa.null() if null_typed else pa.float64()),
            pa.field("seed", pa.int64()),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array(["S12F-CANDIDATE-02"]),
            pa.array([72]),
            pa.nulls(1, type=schema.field(2).type),
            pa.array([1]),
        ],
        schema=schema,
    )


def test_preregistration_freezes_exact_scope_and_original_replay_count() -> None:
    config = yaml.safe_load(
        (
            REPO
            / "configs/e01/s13rr_downstream_schema_canonicalization_preregistration.yaml"
        ).read_text()
    )
    assert config["researchStepId"] == RESEARCH_STEP_ID
    assert config["versionedStepId"] == VERSION
    assert config["inputs"]["taskCount"] == 200
    assert config["derivedViewContract"]["sourceTables"] == list(TABLE_FAMILIES)
    assert (
        set(config["derivedViewContract"]["exactNewAffectedTasks"])
        == DOWNSTREAM_AFFECTED_TASKS
    )
    assert config["frozenSourceReplayGate"]["executedSuffixExactCount"] == 3600
    assert config["statisticsContract"]["executions"] == 2
    assert config["permanentStopRule"]["furtherSchemaChangePermitted"] is False


def test_prefix_adapter_types_only_exact_all_null_fields() -> None:
    source = _prefix(null_typed=True)
    canonical = _prefix(null_typed=False).schema
    view, adapted, operation = canonicalize_table(
        family="prefix.parquet", task_id=TASK, source=source, canonical_schema=canonical
    )
    assert set(adapted) == set(PREFIX_FIELD_TYPES)
    assert operation == "ALL_INELIGIBLE_PREFIX_NULL_FIELD_TYPING"
    assert view.num_rows == source.num_rows
    assert view.column_names == source.column_names
    for name, target in PREFIX_FIELD_TYPES.items():
        assert view.schema.field(name).type.equals(target)
        assert view[name].null_count == 1


def test_seed_adapter_types_only_endpoint_generation() -> None:
    source = _seeds(null_typed=True)
    canonical = _seeds(null_typed=False).schema
    view, adapted, _ = canonicalize_table(
        family="seeds.parquet", task_id=TASK, source=source, canonical_schema=canonical
    )
    assert tuple(adapted) == tuple(SEED_FIELD_TYPES)
    assert view["endpointGeneration"].null_count == 1
    assert view["seed"].equals(source["seed"])


def test_zero_row_suffix_gets_schema_without_rows() -> None:
    source = pa.table({})
    canonical = pa.schema(
        [pa.field("candidateId", pa.string()), pa.field("status", pa.string())]
    )
    view, adapted, operation = canonicalize_table(
        family="suffix.parquet", task_id=TASK, source=source, canonical_schema=canonical
    )
    assert view.num_rows == source.num_rows == 0
    assert view.num_columns == 2
    assert adapted == ("candidateId", "status")
    assert operation == "ZERO_ROW_SUFFIX_CANONICAL_SCHEMA_ASSIGNMENT"
    assert schema_signature(view.schema) == schema_signature(canonical)


def test_adapter_rejects_undeclared_task_and_nonnull_or_other_types() -> None:
    canonical = _prefix(null_typed=False).schema
    with pytest.raises(ValueError, match="undeclared prefix variant"):
        canonicalize_table(
            family="prefix.parquet",
            task_id="S12F-CANDIDATE-02/M71",
            source=_prefix(null_typed=True),
            canonical_schema=canonical,
        )
    bad = _prefix(null_typed=True)
    idx = bad.schema.get_field_index("synergy")
    bad = bad.set_column(idx, "synergy", pa.array([1], type=pa.int64()))
    with pytest.raises(TypeError, match="undeclared source type"):
        canonicalize_table(
            family="prefix.parquet",
            task_id=TASK,
            source=bad,
            canonical_schema=canonical,
        )


def test_canonical_input_is_exact_identity_view() -> None:
    source = _prefix(null_typed=False)
    view, adapted, operation = canonicalize_table(
        family="prefix.parquet",
        task_id="S12F-CANDIDATE-02/M00",
        source=source,
        canonical_schema=source.schema,
    )
    assert adapted == ()
    assert operation == "IDENTITY_CANONICAL_VIEW"
    assert view.equals(source)
