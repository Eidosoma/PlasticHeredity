"""Prospectively locked downstream schema views for E01 S13RR.

This module has no simulator, source estimator, or statistics implementation.
It can only reproduce the S13R label view and normalize the exact all-ineligible
matrix-72 prefix, suffix, and seed schemas named by the human authorization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pyarrow as pa
from pyarrow import ipc

from e01_s13r_schema_normalization.core import normalize_label_table

VERSION = "E01-S13RR-DOWNSTREAM-SCHEMA-CANONICALIZATION-v1.0.0"
RESEARCH_STEP_ID = "S13RR"
ADAPTER_ID = "S13RR_EXACT_EIGHT_FAMILY_CANONICAL_VIEW-v1.0.0"

TABLE_FAMILIES = (
    "labels.parquet",
    "preprocessing.parquet",
    "full.parquet",
    "prefix.parquet",
    "partition.parquet",
    "diagnostic.parquet",
    "suffix.parquet",
    "seeds.parquet",
)

LABEL_AFFECTED_TASKS = frozenset(
    {
        "S12F-CANDIDATE-02/M30",
        "S12F-CANDIDATE-02/M35",
        "S12F-CANDIDATE-02/M63",
        "S12F-CANDIDATE-03/M35",
        "S12F-CANDIDATE-03/M63",
    }
)
DOWNSTREAM_AFFECTED_TASKS = frozenset(
    {"S12F-CANDIDATE-02/M72", "S12F-CANDIDATE-03/M72"}
)

PREFIX_FIELD_TYPES: Mapping[str, pa.DataType] = {
    "synergy": pa.float64(),
    "downwardCausation": pa.float64(),
    "emergence": pa.float64(),
    "localPhiR": pa.float64(),
    "exactReplayPassed": pa.bool_(),
    "futureSuffixStructuralGatePassed": pa.bool_(),
    "futureSuffixExecutedSentinelPassed": pa.bool_(),
}
SEED_FIELD_TYPES: Mapping[str, pa.DataType] = {
    "endpointGeneration": pa.float64(),
}


def schema_signature(schema: pa.Schema) -> tuple[tuple[str, str], ...]:
    """Return the ordered, metadata-independent physical schema."""

    return tuple((field.name, str(field.type)) for field in schema)


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(repr(schema_signature(schema)).encode()).hexdigest()


def physical_field_sha256(table: pa.Table, name: str) -> str:
    """Hash a single field including its physical type and values."""

    field = table.schema.field(name)
    one = pa.Table.from_arrays([table[name]], schema=pa.schema([field]))
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, one.schema) as writer:
        writer.write_table(one)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _replace_all_null_fields(
    table: pa.Table, field_types: Mapping[str, pa.DataType]
) -> tuple[pa.Table, tuple[str, ...]]:
    """Replace only Arrow-null, logically all-null fields with exact types."""

    result = table
    adapted: list[str] = []
    for name, target in field_types.items():
        if name not in result.column_names:
            raise ValueError(f"missing declared field {name}")
        index = result.schema.get_field_index(name)
        field = result.schema.field(index)
        column = result.column(index)
        if field.type.equals(target):
            continue
        if not pa.types.is_null(field.type):
            raise TypeError(f"undeclared source type for {name}: {field.type}")
        if column.null_count != len(column):
            raise ValueError(f"null-typed field {name} is not all null")
        chunks = [pa.nulls(len(chunk), type=target) for chunk in column.chunks]
        replacement = pa.chunked_array(chunks, type=target)
        result = result.set_column(index, pa.field(name, target), replacement)
        adapted.append(name)
    return result, tuple(adapted)


def canonicalize_table(
    *,
    family: str,
    task_id: str,
    source: pa.Table,
    canonical_schema: pa.Schema,
) -> tuple[pa.Table, tuple[str, ...], str]:
    """Construct the only allowed canonical derived view.

    Returns ``(view, adapted_fields, operation)``. Any unregistered variant is
    rejected. The caller separately verifies logical equality and source hashes.
    """

    if family not in TABLE_FAMILIES:
        raise ValueError(f"undeclared table family: {family}")

    canonical_signature = schema_signature(canonical_schema)
    source_signature = schema_signature(source.schema)
    provisional = source
    adapted_fields: tuple[str, ...] = ()
    operation = "IDENTITY_CANONICAL_VIEW"

    if family == "labels.parquet":
        provisional, adapted_fields = normalize_label_table(source)
        if adapted_fields:
            if task_id not in LABEL_AFFECTED_TASKS:
                raise ValueError(f"undeclared label variant task: {task_id}")
            operation = "INHERITED_S13R_ALL_NULL_FIELD_TYPING"
        elif task_id in LABEL_AFFECTED_TASKS:
            raise ValueError(f"declared label variant is already canonical: {task_id}")
    elif family == "prefix.parquet":
        provisional, adapted_fields = _replace_all_null_fields(
            source, PREFIX_FIELD_TYPES
        )
        if adapted_fields:
            if task_id not in DOWNSTREAM_AFFECTED_TASKS:
                raise ValueError(f"undeclared prefix variant task: {task_id}")
            if set(adapted_fields) != set(PREFIX_FIELD_TYPES):
                raise ValueError("partial prefix adaptation is forbidden")
            operation = "ALL_INELIGIBLE_PREFIX_NULL_FIELD_TYPING"
        elif task_id in DOWNSTREAM_AFFECTED_TASKS:
            raise ValueError(f"declared prefix variant is already canonical: {task_id}")
    elif family == "seeds.parquet":
        provisional, adapted_fields = _replace_all_null_fields(source, SEED_FIELD_TYPES)
        if adapted_fields:
            if task_id not in DOWNSTREAM_AFFECTED_TASKS:
                raise ValueError(f"undeclared seed variant task: {task_id}")
            if set(adapted_fields) != set(SEED_FIELD_TYPES):
                raise ValueError("partial seed adaptation is forbidden")
            operation = "ALL_INELIGIBLE_SEED_NULL_FIELD_TYPING"
        elif task_id in DOWNSTREAM_AFFECTED_TASKS:
            raise ValueError(f"declared seed variant is already canonical: {task_id}")
    elif family == "suffix.parquet" and source_signature != canonical_signature:
        if task_id not in DOWNSTREAM_AFFECTED_TASKS:
            raise ValueError(f"undeclared suffix variant task: {task_id}")
        if source.num_rows != 0 or source.num_columns != 0:
            raise ValueError("suffix schema assignment is allowed only for 0x0 tables")
        provisional = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in canonical_schema],
            schema=canonical_schema,
        )
        adapted_fields = tuple(field.name for field in canonical_schema)
        operation = "ZERO_ROW_SUFFIX_CANONICAL_SCHEMA_ASSIGNMENT"

    if schema_signature(provisional.schema) != canonical_signature:
        raise ValueError(f"{family} did not reach the exact canonical schema")

    # Use canonical field ordering and metadata without changing any array.
    view = pa.Table.from_arrays(
        [provisional[name] for name in canonical_schema.names],
        schema=canonical_schema,
    )
    return view, adapted_fields, operation
