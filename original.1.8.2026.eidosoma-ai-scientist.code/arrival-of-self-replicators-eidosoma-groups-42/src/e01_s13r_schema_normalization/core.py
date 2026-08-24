"""The single, narrow physical-schema adapter authorized for E01 S13R.

Only three label columns may change physical Arrow type, and only when the
input field has Arrow's null type.  Values, null masks, rows, order, metadata,
and every other field remain logically unchanged.  This module deliberately
contains no scientific calculation and no generic schema-promotion facility.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa

VERSION = "E01-S13R-SCHEMA-NORMALIZATION-CONFIRMATION-v1.0.0"
RESEARCH_STEP_ID = "S13R"
ADAPTER_ID = "S13R_LABEL_OPTIONAL_NULL_TYPE_ADAPTER-v1.0.0"

ADAPTER_FIELD_TYPES: Mapping[str, pa.DataType] = {
    "clusterId": pa.string(),
    "referenceObservationId": pa.string(),
    "metricToReference": pa.float64(),
}

EXPECTED_ADAPTED_TASK_IDS = frozenset(
    {
        "S12F-CANDIDATE-02/M30",
        "S12F-CANDIDATE-02/M35",
        "S12F-CANDIDATE-02/M63",
        "S12F-CANDIDATE-03/M35",
        "S12F-CANDIDATE-03/M63",
    }
)


def table_schema_signature(table: pa.Table) -> list[tuple[str, str]]:
    """Return a metadata-independent ordered field signature."""

    return [(field.name, str(field.type)) for field in table.schema]


def schema_sha256(table: pa.Table) -> str:
    payload = json.dumps(table_schema_signature(table), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__nonfinite__": "NaN"}
        return {"__nonfinite__": "+Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    return value


def array_value_sha256(array: pa.ChunkedArray) -> str:
    """Hash logical values without incorporating the Arrow physical type."""

    payload = [_json_value(value) for value in array.to_pylist()]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def null_mask_sha256(array: pa.ChunkedArray) -> str:
    mask = bytes(1 if value is None else 0 for value in array.to_pylist())
    return hashlib.sha256(mask).hexdigest()


def table_logical_sha256(table: pa.Table) -> str:
    payload = [
        {
            "name": name,
            "values": array_value_sha256(table[name]),
            "nullMask": null_mask_sha256(table[name]),
        }
        for name in table.column_names
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_label_table(
    table: pa.Table,
) -> tuple[pa.Table, tuple[str, ...]]:
    """Return the authorized typed view and the fields physically adapted.

    A canonical field already having its target type is left untouched.  An
    Arrow-null field may be replaced only when every row is null.  Any other
    physical type is rejected.  No generic cast is intentionally exposed.
    """

    missing = [name for name in ADAPTER_FIELD_TYPES if name not in table.column_names]
    if missing:
        raise ValueError(f"missing authorized label fields: {missing}")
    result = table
    adapted: list[str] = []
    for name, target_type in ADAPTER_FIELD_TYPES.items():
        index = result.schema.get_field_index(name)
        field = result.schema.field(index)
        column = result.column(index)
        if field.type.equals(target_type):
            continue
        if not pa.types.is_null(field.type):
            raise TypeError(
                f"field {name} has forbidden source type {field.type}; "
                f"expected null or {target_type}"
            )
        if column.null_count != len(column):
            raise ValueError(f"null-typed field {name} contains a nonnull value")
        replacement = pa.chunked_array(
            [pa.nulls(len(chunk), type=target_type) for chunk in column.chunks],
            type=target_type,
        )
        result = result.set_column(index, pa.field(name, target_type), replacement)
        adapted.append(name)
    return result, tuple(adapted)
