"""Narrow S13R schema-normalization confirmation contract."""

from .core import (
    ADAPTER_FIELD_TYPES,
    EXPECTED_ADAPTED_TASK_IDS,
    RESEARCH_STEP_ID,
    VERSION,
    normalize_label_table,
    table_schema_signature,
)

__all__ = [
    "ADAPTER_FIELD_TYPES",
    "EXPECTED_ADAPTED_TASK_IDS",
    "RESEARCH_STEP_ID",
    "VERSION",
    "normalize_label_table",
    "table_schema_signature",
]
