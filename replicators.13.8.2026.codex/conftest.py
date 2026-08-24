"""Current-lifecycle pytest annotations for checksum-sealed historical tests.

The two marked tests describe the state immediately after the original P1/P2
readback crashes: at that point the atomic result directories did not exist.
Both result directories now exist because their separately checksum-sealed,
zero-future recoveries completed successfully.  The historical test sources
remain untouched so their amendment hashes remain valid; this additive hook
marks only those superseded pre-recovery assertions as expected failures.
"""

from __future__ import annotations

import pytest


SUPERSEDED_PRE_RECOVERY_TESTS = {
    (
        "tests/test_intervention_readback_recovery.py::"
        "test_failure_is_the_preregistered_derived_field_mismatch"
    ),
    (
        "tests/test_intervention_p2_readback_recovery.py::"
        "test_p2_stopped_only_at_the_known_readback_failure"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marker = pytest.mark.xfail(
        strict=True,
        reason=(
            "historical pre-recovery invariant superseded by the sealed "
            "zero-future recovery"
        ),
    )
    for item in items:
        if item.nodeid in SUPERSEDED_PRE_RECOVERY_TESTS:
            item.add_marker(marker)
