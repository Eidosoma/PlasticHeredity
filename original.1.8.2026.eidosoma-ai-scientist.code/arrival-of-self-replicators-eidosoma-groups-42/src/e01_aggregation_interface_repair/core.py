"""The complete and only data adapter authorized for S12J.

The adapter creates one derived column in an in-memory statistical view.  It
does not mutate S12I's Parquet file, change any existing value, reorder a row,
or add any other semantic behavior.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import pyarrow as pa
from pandas.api.types import is_integer_dtype
from pyarrow import ipc

VERSION = "E01-S12J-AGGREGATION-INTERFACE-REPAIR-CONFIRMATION-v1.0.0"
RESEARCH_STEP_ID = "S12J"
EVIDENCE_CLASS = "HUMAN_AUTHORIZED_AGGREGATION_INTERFACE_REPAIR_CONFIRMATION"
ADAPTER_ID = "S12J-PREFIX-ENDPOINT-RAW-INDEX-ALIAS-v1.0.0"
CANDIDATE_IDS = (
    "S12F-CANDIDATE-01",
    "S12F-CANDIDATE-02",
    "S12F-CANDIDATE-03",
)
SOURCE_FIELD = "endpointRawObservationIndex"
DERIVED_FIELD = "rawObservationIndex"
UNIQUE_KEYS = (
    "candidateId",
    "trajectoryId",
    "implementationId",
    "generation",
)
MONOTONIC_GROUP_KEYS = (
    "candidateId",
    "trajectoryId",
    "implementationId",
)
ENDPOINT_JOIN_KEYS = (
    "candidateId",
    "trajectoryId",
    "matrixIndex",
    "generation",
)
HISTORICAL_LABEL_ID = "HISTORICAL_H090_REPLICATOR"


def _arrow_hash(frame: pd.DataFrame) -> str:
    """Hash a canonical Arrow stream for an ordered frame."""

    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def adapt_prefix_statistical_view(prefix: pd.DataFrame) -> pd.DataFrame:
    """Return a copied view with the single exact index alias."""

    if SOURCE_FIELD not in prefix.columns:
        raise ValueError(f"missing required source field: {SOURCE_FIELD}")
    if DERIVED_FIELD in prefix.columns:
        raise ValueError(f"derived field already exists: {DERIVED_FIELD}")
    if prefix[SOURCE_FIELD].isna().any():
        raise ValueError(f"source field contains nulls: {SOURCE_FIELD}")
    if not is_integer_dtype(prefix[SOURCE_FIELD].dtype):
        raise TypeError(f"source field is not integer typed: {prefix[SOURCE_FIELD].dtype}")
    adapted = prefix.copy(deep=True)
    adapted[DERIVED_FIELD] = adapted[SOURCE_FIELD].copy(deep=True)
    return adapted


def validate_prefix_adapter(
    prefix: pd.DataFrame,
    adapted: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate every preregistered row, identity, and non-adapter field gate."""

    original_columns = list(prefix.columns)
    required = set(UNIQUE_KEYS) | set(ENDPOINT_JOIN_KEYS) | {SOURCE_FIELD}
    missing = sorted(required - set(original_columns))
    if missing:
        raise ValueError(f"prefix adapter validation missing columns: {missing}")
    if DERIVED_FIELD in original_columns:
        raise ValueError("prefix source unexpectedly already contains derived field")
    expected_columns = [*original_columns, DERIVED_FIELD]
    row_count_unchanged = len(prefix) == len(adapted)
    column_contract = list(adapted.columns) == expected_columns

    exact_original_fields = False
    try:
        pd.testing.assert_frame_equal(
            prefix,
            adapted[original_columns],
            check_exact=True,
            check_dtype=True,
            check_like=False,
            check_names=True,
        )
        exact_original_fields = True
    except AssertionError:
        exact_original_fields = False

    original_column_hashes = {
        column: _arrow_hash(prefix[[column]]) for column in original_columns
    }
    adapted_original_column_hashes = {
        column: _arrow_hash(adapted[[column]]) for column in original_columns
    }
    unchanged_column_hashes = {
        column: original_column_hashes[column]
        == adapted_original_column_hashes[column]
        for column in original_columns
    }
    row_order_hash_before = _arrow_hash(prefix)
    row_order_hash_after = _arrow_hash(adapted[original_columns])

    nonnull = bool(
        adapted[SOURCE_FIELD].notna().all()
        and adapted[DERIVED_FIELD].notna().all()
    )
    integer_typed = bool(
        is_integer_dtype(adapted[SOURCE_FIELD].dtype)
        and is_integer_dtype(adapted[DERIVED_FIELD].dtype)
    )
    exact_alias_identity = bool(
        adapted[SOURCE_FIELD].array.equals(adapted[DERIVED_FIELD].array)
    )
    unique_keys = bool(not adapted.duplicated(list(UNIQUE_KEYS), keep=False).any())

    monotonic_rows: list[dict[str, Any]] = []
    for keys, group in adapted.groupby(
        list(MONOTONIC_GROUP_KEYS), sort=False, dropna=False
    ):
        values = group[SOURCE_FIELD]
        strictly_increasing = bool(
            values.is_monotonic_increasing
            and values.is_unique
            and (values.diff().dropna() > 0).all()
        )
        monotonic_rows.append(
            {
                "candidateId": keys[0],
                "trajectoryId": keys[1],
                "implementationId": keys[2],
                "rowCount": len(group),
                "strictlyIncreasing": strictly_increasing,
            }
        )
    monotonic = bool(
        len(monotonic_rows) == 192
        and all(item["strictlyIncreasing"] for item in monotonic_rows)
    )

    historical = labels[labels["labelId"] == HISTORICAL_LABEL_ID][
        [*ENDPOINT_JOIN_KEYS, "postFissionRawObservationIndex"]
    ].copy()
    label_keys_unique = bool(
        len(historical) == 9600
        and not historical.duplicated(list(ENDPOINT_JOIN_KEYS), keep=False).any()
    )
    endpoint_check = adapted[
        [*ENDPOINT_JOIN_KEYS, SOURCE_FIELD, DERIVED_FIELD]
    ].merge(
        historical,
        on=list(ENDPOINT_JOIN_KEYS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    endpoint_rows_matched = bool(
        len(endpoint_check) == len(adapted)
        and endpoint_check["postFissionRawObservationIndex"].notna().all()
    )
    endpoint_identity = bool(
        endpoint_rows_matched
        and (
            endpoint_check[SOURCE_FIELD]
            == endpoint_check["postFissionRawObservationIndex"]
        ).all()
        and (
            endpoint_check[DERIVED_FIELD]
            == endpoint_check["postFissionRawObservationIndex"]
        ).all()
    )

    audit_view = adapted[
        [
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "implementationId",
            "generation",
            SOURCE_FIELD,
            DERIVED_FIELD,
        ]
    ].copy()
    audit_view.insert(0, "rowOrdinal", range(len(audit_view)))

    gates = {
        "sourceFieldPresent": SOURCE_FIELD in prefix.columns,
        "derivedFieldAbsentBeforeAdapter": DERIVED_FIELD not in prefix.columns,
        "rowCountUnchanged": row_count_unchanged,
        "rowOrderAndOriginalColumnHashUnchanged": row_order_hash_before
        == row_order_hash_after,
        "columnContractExact": column_contract,
        "everyOriginalFieldExactEquality": exact_original_fields,
        "everyOriginalFieldCanonicalArrowHashUnchanged": all(
            unchanged_column_hashes.values()
        ),
        "sourceAndDerivedNonnull": nonnull,
        "sourceAndDerivedInteger": integer_typed,
        "exactRowwiseAliasIdentity": exact_alias_identity,
        "frozenKeyUniqueness": unique_keys,
        "strictMonotonicityByTrajectoryImplementation": monotonic,
        "historicalLabelEndpointKeysUnique": label_keys_unique,
        "everyPrefixRowMatchedToEndpointIdentity": endpoint_rows_matched,
        "exactEndpointIdentityAgreement": endpoint_identity,
    }
    validation = {
        "schema": "eidosoma.e01.s12j_adapter_validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSION,
        "adapterId": ADAPTER_ID,
        "sourceField": SOURCE_FIELD,
        "derivedField": DERIVED_FIELD,
        "inputRowCount": len(prefix),
        "adaptedRowCount": len(adapted),
        "inputColumnCount": len(prefix.columns),
        "adaptedColumnCount": len(adapted.columns),
        "monotonicGroupCount": len(monotonic_rows),
        "endpointMatchedRowCount": int(
            endpoint_check["postFissionRawObservationIndex"].notna().sum()
        ),
        "rowOrderHashBefore": row_order_hash_before,
        "rowOrderHashAfterWithoutAdapter": row_order_hash_after,
        "originalColumnHashesBefore": original_column_hashes,
        "originalColumnHashesAfter": adapted_original_column_hashes,
        "unchangedOriginalColumnHashes": unchanged_column_hashes,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return validation, audit_view
